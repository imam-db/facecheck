"""Face detection and embedding with OpenCV Zoo YuNet + SFace.

``FaceEmbedder`` is the seam the rest of the service depends on, which keeps
tests and evaluation independent of the ONNX models. The concrete
implementation wraps ``cv2.FaceDetectorYN`` and ``cv2.FaceRecognizerSF`` and
applies the quality gate in :mod:`facecheck.quality`.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import cv2
import numpy as np

from facecheck.config import Settings
from facecheck.quality import (
    INVALID_IMAGE,
    FaceCandidate,
    QualityError,
    select_face,
)
from facecheck.similarity import l2_normalize


class EmbedderError(RuntimeError):
    """Base class for embedder failures that are not quality rejections."""


class ModelNotFoundError(EmbedderError):
    """Raised when the ONNX model files are not available on disk."""


@dataclass(frozen=True)
class EmbeddingResult:
    """A normalized face embedding plus the detection that produced it."""

    embedding: np.ndarray
    bbox: tuple[float, float, float, float]
    landmarks: np.ndarray | None
    model_version: str
    detector_score: float | None = None


@runtime_checkable
class FaceEmbedder(Protocol):
    """Minimal interface for a detect-and-embed backend."""

    @property
    def model_version(self) -> str: ...

    @property
    def embedding_dim(self) -> int: ...

    def detect_and_embed(self, bgr_image: np.ndarray) -> EmbeddingResult: ...


def decode_image_bgr(data: bytes) -> np.ndarray:
    """Decode encoded image bytes (JPEG/PNG/...) into a BGR uint8 array."""

    if not data:
        raise QualityError(INVALID_IMAGE, "empty image payload")
    buffer = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise QualityError(INVALID_IMAGE, "could not decode image")
    return image


class OpenCVYuNetSFaceEmbedder:
    """YuNet detection followed by SFace alignment, crop, and embedding."""

    def __init__(
        self,
        detector_path: str | Path,
        recognizer_path: str | Path,
        *,
        min_face_size: int = 80,
        detector_score_threshold: float = 0.9,
        nms_threshold: float = 0.3,
        top_k: int = 5000,
        model_version: str | None = None,
    ) -> None:
        detector_path = Path(detector_path)
        recognizer_path = Path(recognizer_path)
        missing = [str(path) for path in (detector_path, recognizer_path) if not path.is_file()]
        if missing:
            raise ModelNotFoundError(
                "missing model file(s): "
                + ", ".join(missing)
                + ". Run `python scripts/download_models.py` first."
            )

        self._detector = cv2.FaceDetectorYN.create(
            str(detector_path),
            "",
            (320, 320),
            detector_score_threshold,
            nms_threshold,
            top_k,
        )
        self._recognizer = cv2.FaceRecognizerSF.create(str(recognizer_path), "")
        self._min_face_size = min_face_size
        self._model_version = model_version or (
            f"{detector_path.stem}+{recognizer_path.stem}"
        )
        self._embedding_dim = 128
        self._lock = threading.Lock()

    @classmethod
    def from_settings(cls, settings: Settings) -> OpenCVYuNetSFaceEmbedder:
        return cls(
            settings.detector_path,
            settings.recognizer_path,
            min_face_size=settings.min_face_size,
            detector_score_threshold=settings.detector_score_threshold,
            model_version=settings.model_version,
        )

    @property
    def model_version(self) -> str:
        return self._model_version

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    def detect_and_embed(self, bgr_image: np.ndarray) -> EmbeddingResult:
        image = np.asarray(bgr_image)
        if image.ndim != 3 or image.shape[2] not in (3, 4):
            raise QualityError(INVALID_IMAGE, "expected an HxWx3 BGR image")
        if image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)

        height, width = image.shape[:2]
        if height == 0 or width == 0:
            raise QualityError(INVALID_IMAGE, "empty image")

        # FaceDetectorYN keeps the input size as mutable state, so detection
        # and recognition are serialized across FastAPI worker threads.
        with self._lock:
            self._detector.setInputSize((width, height))
            _, raw_faces = self._detector.detect(image)
            candidates = _to_candidates(raw_faces)
            face = select_face(candidates, min_face_size=self._min_face_size)

            face_row = _face_row(face)
            aligned = self._recognizer.alignCrop(image, face_row)
            if aligned is None or aligned.size == 0:
                raise EmbedderError("face alignment failed")
            feature = self._recognizer.feature(aligned)

        embedding = l2_normalize(np.asarray(feature, dtype=np.float32).reshape(-1))
        self._embedding_dim = int(embedding.shape[0])
        return EmbeddingResult(
            embedding=embedding,
            bbox=face.bbox,
            landmarks=face.landmarks,
            model_version=self._model_version,
            detector_score=face.score,
        )


def _to_candidates(raw_faces: np.ndarray | None) -> list[FaceCandidate]:
    if raw_faces is None or len(raw_faces) == 0:
        return []

    candidates: list[FaceCandidate] = []
    for row in np.asarray(raw_faces, dtype=np.float32):
        row = row.reshape(-1)
        bbox = (float(row[0]), float(row[1]), float(row[2]), float(row[3]))
        landmarks = row[4:14].reshape(5, 2).copy()
        score = float(row[14]) if row.size > 14 else None
        candidates.append(FaceCandidate(bbox=bbox, landmarks=landmarks, score=score))
    return candidates


def _face_row(face: FaceCandidate) -> np.ndarray:
    """Rebuild the 15-element YuNet row that ``FaceRecognizerSF.alignCrop`` expects."""

    x, y, w, h = face.bbox
    if face.landmarks is None:
        raise EmbedderError("alignment requires YuNet landmarks")
    landmarks = np.asarray(face.landmarks, dtype=np.float32).reshape(-1)
    score = face.score if face.score is not None else 0.0
    return np.concatenate(
        [np.array([x, y, w, h], dtype=np.float32), landmarks, np.array([score], dtype=np.float32)]
    )


def build_embedder(settings: Settings) -> FaceEmbedder:
    """Construct the production embedder from settings."""

    return OpenCVYuNetSFaceEmbedder.from_settings(settings)
