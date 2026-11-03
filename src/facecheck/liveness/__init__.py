"""Passive presentation-attack detection building blocks.

This package holds the runtime detector module only. It is not wired into the
enroll or verify flow and applies no pass/fail threshold; callers score a crop
and interpret the probability themselves.
"""

from __future__ import annotations

from facecheck.liveness.base import (
    LivenessDetector,
    LivenessError,
    LivenessInputError,
    LivenessModelNotFoundError,
    LivenessResult,
)
from facecheck.liveness.boxes import map_detection_box_to_full_res
from facecheck.liveness.minifasnet import MiniFASNetLivenessDetector

__all__ = [
    "LivenessDetector",
    "LivenessError",
    "LivenessInputError",
    "LivenessModelNotFoundError",
    "LivenessResult",
    "MiniFASNetLivenessDetector",
    "map_detection_box_to_full_res",
]
