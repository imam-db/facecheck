"""Passive presentation-attack detection building blocks.

This package holds the runtime detector module only. It is not wired into the
enroll or verify flow and applies no pass/fail threshold; callers score a crop
and interpret the probability themselves.
"""

from __future__ import annotations

from facecheck.liveness.base import LivenessDetector, LivenessResult
from facecheck.liveness.boxes import map_detection_box_to_full_res
from facecheck.liveness.crops import CropImage, crop_face
from facecheck.liveness.minifasnet import MiniFASNetLivenessDetector

__all__ = [
    "CropImage",
    "LivenessDetector",
    "LivenessResult",
    "MiniFASNetLivenessDetector",
    "crop_face",
    "map_detection_box_to_full_res",
]
