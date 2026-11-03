"""MiniFASNet passive liveness detector using the exported ONNX models.

The runtime scores a full-resolution BGR image and a face box. It crops the
face to 80x80 with the upstream ``CropImage`` arithmetic at each model's scale,
feeds **raw BGR float32 in [0, 255]** as an NCHW blob (no ``/255``, no RGB
swap), softmaxes the three logits, and averages the real-class (index ``1``)
probability across the two models.

It does not apply a pass threshold; the caller interprets ``score``.
"""

from __future__ import annotations

import threading
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np

from facecheck.config import Settings, apply_opencv_thread_settings
from facecheck.embeddings import SHA_PREFIX_LENGTH, EmbeddingResult, file_sha256
from facecheck.liveness.base import (
    LivenessInputError,
    LivenessModelNotFoundError,
    LivenessResult,
)
from facecheck.liveness.crops import CropImage

__all__ = ["MiniFASNetLivenessDetector", "validate_liveness_inputs"]

INPUT_SIZE = 80
REAL_CLASS_INDEX = 1
V2_SCALE = 2.7
V1SE_SCALE = 4.0


def _softmax(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64).reshape(-1)
    shifted = values - values.max()
    exp = np.exp(shifted)
    return exp / exp.sum()


def _blob(crop: np.ndarray) -> np.ndarray:
    """Build a raw BGR float32 NCHW blob in [0, 255]."""

    array = np.asarray(crop, dtype=np.float32)
    return np.ascontiguousarray(array.transpose(2, 0, 1)[None])


def validate_liveness_inputs(
    image: np.ndarray, bbox_xywh: Sequence[int] | list[int]
) -> tuple[np.ndarray, list[int]]:
    """Validate a PAD image and box, returning the array and integer box.

    The image must be an HxWx3 uint8 BGR array. Float images (including
    normalized ``[0, 1]`` input), grayscale, and BGRA are rejected instead of
    being silently coerced. The box must have four components and a positive
    width and height.
    """

    if not isinstance(image, np.ndarray):
        raise LivenessInputError("liveness image must be a numpy array")
    if image.dtype != np.uint8:
        raise LivenessInputError(
            f"liveness image must be uint8, got {image.dtype}; "
            "pass raw BGR pixels, not float [0, 1]"
        )
    if image.ndim != 3 or image.shape[2] != 3:
        raise LivenessInputError(
            f"liveness image must be HxWx3 BGR, got shape {image.shape}"
        )
    if image.shape[0] == 0 or image.shape[1] == 0:
        raise LivenessInputError("liveness image is empty")

    if len(bbox_xywh) != 4:
        raise LivenessInputError(
            f"liveness box must have 4 components, got {len(bbox_xywh)}"
        )
    try:
        bbox = [int(value) for value in bbox_xywh]
    except (TypeError, ValueError) as exc:
        raise LivenessInputError("liveness box components must be numbers") from exc
    if bbox[2] < 1 or bbox[3] < 1:
        raise LivenessInputError(
            f"liveness box must have positive width and height, got {bbox}"
        )
    return image, bbox


class MiniFASNetLivenessDetector:
    """Two-model MiniFASNet ensemble behind the ``LivenessDetector`` protocol."""

    def __init__(self, v2_path: str | Path, v1se_path: str | Path) -> None:
        v2_path = Path(v2_path)
        v1se_path = Path(v1se_path)
        missing = [str(path) for path in (v2_path, v1se_path) if not path.is_file()]
        if missing:
            raise LivenessModelNotFoundError(
                "missing model file(s): "
                + ", ".join(missing)
                + ". Run `python scripts/download_models.py` first."
            )

        self._models = (
            (v2_path.name, V2_SCALE, cv2.dnn.readNetFromONNX(str(v2_path))),
            (v1se_path.name, V1SE_SCALE, cv2.dnn.readNetFromONNX(str(v1se_path))),
        )
        self._model_version = (
            f"{v2_path.stem}@{file_sha256(v2_path)[:SHA_PREFIX_LENGTH]}+"
            f"{v1se_path.stem}@{file_sha256(v1se_path)[:SHA_PREFIX_LENGTH]}"
        )
        self._cropper = CropImage()
        # cv2.dnn.Net keeps mutable state across setInput/forward, so all
        # inference is serialized through a single lock shared by both nets.
        self._lock = threading.Lock()

    @classmethod
    def from_settings(cls, settings: Settings) -> MiniFASNetLivenessDetector:
        # Interim placement: applying here covers detector construction. The app
        # lifespan should call ``apply_opencv_thread_settings`` once at startup.
        apply_opencv_thread_settings(settings)
        return cls(settings.pad_v2_path, settings.pad_v1se_path)

    @property
    def model_version(self) -> str:
        return self._model_version

    def score(
        self, bgr_image: np.ndarray, bbox_xywh: Sequence[int] | list[int]
    ) -> LivenessResult:
        image, bbox = validate_liveness_inputs(bgr_image, bbox_xywh)

        per_model: dict[str, float] = {}
        with self._lock:
            for name, scale, net in self._models:
                crop = self._cropper.crop(image, bbox, scale, INPUT_SIZE, INPUT_SIZE)
                try:
                    net.setInput(_blob(crop))
                    logits = net.forward()
                except cv2.error as exc:
                    raise LivenessInputError(
                        f"MiniFASNet inference failed for {name}: {exc}"
                    ) from exc
                per_model[name] = float(_softmax(logits)[REAL_CLASS_INDEX])

        score = float(np.mean(list(per_model.values())))
        height, width = image.shape[:2]
        aspect_ratio = width / height if height else None
        return LivenessResult(
            score=score,
            per_model=per_model,
            model_version=self._model_version,
            aspect_ratio=aspect_ratio,
        )

    def score_face(
        self, full_res_image: np.ndarray, embedding_result: EmbeddingResult
    ) -> LivenessResult:
        """Score the face from an ``EmbeddingResult`` in its full-res frame.

        ``full_res_image`` must be the **exact** array that was passed to
        ``detect_and_embed`` (the same object or a decode of the same
        full-resolution bytes), not a re-downscaled copy. The box is mapped
        back with ``embedding_result.detection_scale``, so any other frame
        puts the crop in the wrong pixels.
        """

        return self.score(full_res_image, embedding_result.full_res_bbox_xywh)
