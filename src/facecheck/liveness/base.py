"""Liveness detector protocol and result type.

The detector reports a real/bona-fide probability. It deliberately performs no
thresholding or pass/fail decision; that belongs to a later integration step.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    from facecheck.embeddings import EmbeddingResult


class LivenessError(Exception):
    """Base for PAD/liveness failures. Must not subclass ``EmbedderError``."""


class LivenessInputError(LivenessError):
    """Bad image or box passed to the detector."""


class LivenessModelNotFoundError(LivenessError):
    """PAD ONNX files missing on disk."""


@dataclass(frozen=True)
class LivenessResult:
    """Scores from a PAD detector for one face crop.

    ``score`` is the average real-class probability across the underlying
    models. ``per_model`` maps each model filename to its own real-class
    softmax probability. ``aspect_ratio`` is the **frame** width/height
    (``image.shape[1] / image.shape[0]``), matching the upstream
    ``check_image`` 3:4 frame check; it is not the face box ratio.
    """

    score: float
    per_model: dict[str, float]
    model_version: str
    aspect_ratio: float | None = None


@runtime_checkable
class LivenessDetector(Protocol):
    """Minimal interface for a passive liveness backend."""

    @property
    def model_version(self) -> str: ...

    def score(
        self, bgr_image: np.ndarray, bbox_xywh: Sequence[int] | list[int]
    ) -> LivenessResult: ...

    def score_face(
        self, full_res_image: np.ndarray, embedding_result: EmbeddingResult
    ) -> LivenessResult:
        """Score the face described by ``embedding_result``.

        ``full_res_image`` must be the exact array passed to
        ``detect_and_embed``; the detection box is mapped back with
        ``embedding_result.detection_scale``.
        """
        ...
