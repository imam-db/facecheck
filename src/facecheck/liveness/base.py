"""Liveness detector protocol and result type.

The detector reports a real/bona-fide probability. It deliberately performs no
thresholding or pass/fail decision; that belongs to a later integration step.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class LivenessResult:
    """Scores from a PAD detector for one face crop.

    ``score`` is the average real-class probability across the underlying
    models. ``per_model`` maps each model filename to its own real-class
    softmax probability. ``aspect_ratio`` is the crop (or box) width/height,
    logged because the upstream models were trained on roughly 3:4 boxes.
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
