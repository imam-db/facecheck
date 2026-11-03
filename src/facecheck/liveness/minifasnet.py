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

from facecheck.config import Settings
from facecheck.embeddings import SHA_PREFIX_LENGTH, ModelNotFoundError, file_sha256
from facecheck.liveness.base import LivenessResult
from facecheck.liveness.crops import CropImage

__all__ = ["MiniFASNetLivenessDetector"]

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


class MiniFASNetLivenessDetector:
    """Two-model MiniFASNet ensemble behind the ``LivenessDetector`` protocol."""

    def __init__(
        self,
        v2_path: str | Path,
        v1se_path: str | Path,
        *,
        model_version: str | None = None,
    ) -> None:
        v2_path = Path(v2_path)
        v1se_path = Path(v1se_path)
        missing = [str(path) for path in (v2_path, v1se_path) if not path.is_file()]
        if missing:
            raise ModelNotFoundError(
                "missing model file(s): "
                + ", ".join(missing)
                + ". Run `python scripts/download_models.py` first."
            )

        self._models = (
            (v2_path.name, V2_SCALE, cv2.dnn.readNetFromONNX(str(v2_path))),
            (v1se_path.name, V1SE_SCALE, cv2.dnn.readNetFromONNX(str(v1se_path))),
        )
        self._model_version = model_version or (
            f"{v2_path.stem}@{file_sha256(v2_path)[:SHA_PREFIX_LENGTH]}+"
            f"{v1se_path.stem}@{file_sha256(v1se_path)[:SHA_PREFIX_LENGTH]}"
        )
        self._cropper = CropImage()
        # cv2.dnn.Net keeps mutable state across setInput/forward, so all
        # inference is serialized through a single lock shared by both nets.
        self._lock = threading.Lock()

    @classmethod
    def from_settings(cls, settings: Settings) -> MiniFASNetLivenessDetector:
        return cls(settings.pad_v2_path, settings.pad_v1se_path)

    @property
    def model_version(self) -> str:
        return self._model_version

    def score(
        self, bgr_image: np.ndarray, bbox_xywh: Sequence[int] | list[int]
    ) -> LivenessResult:
        image = np.asarray(bgr_image)
        bbox = [int(value) for value in bbox_xywh]

        per_model: dict[str, float] = {}
        with self._lock:
            for name, scale, net in self._models:
                crop = self._cropper.crop(image, bbox, scale, INPUT_SIZE, INPUT_SIZE)
                net.setInput(_blob(crop))
                per_model[name] = float(_softmax(net.forward())[REAL_CLASS_INDEX])

        score = float(np.mean(list(per_model.values())))
        aspect_ratio = bbox[2] / bbox[3] if bbox[3] else None
        return LivenessResult(
            score=score,
            per_model=per_model,
            model_version=self._model_version,
            aspect_ratio=aspect_ratio,
        )
