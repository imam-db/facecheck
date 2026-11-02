"""Face quality gate: exactly one face, large enough to be useful."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

NO_FACE = "no_face"
MULTIPLE_FACES = "multiple_faces"
FACE_TOO_SMALL = "face_too_small"
INVALID_IMAGE = "invalid_image"


class QualityError(ValueError):
    """Raised when an image cannot be used for enrollment or verification."""

    code = "quality_error"

    def __init__(self, reason: str, message: str | None = None) -> None:
        self.reason = reason
        super().__init__(message or reason)


@dataclass(frozen=True)
class FaceCandidate:
    """A detected face, in the coordinate space of the input image."""

    bbox: tuple[float, float, float, float]
    landmarks: np.ndarray | None = None
    score: float | None = None

    @property
    def width(self) -> float:
        return self.bbox[2]

    @property
    def height(self) -> float:
        return self.bbox[3]

    @property
    def min_side(self) -> float:
        return min(self.width, self.height)


def select_face(
    faces: Sequence[FaceCandidate],
    *,
    min_face_size: int = 80,
    image_min_side: float | None = None,
    min_face_frac: float = 0.0,
) -> FaceCandidate:
    """Return the single usable face or raise :class:`QualityError`.

    The policy is deliberately strict: attendance verification is 1:1, so a
    frame containing zero faces, several faces, or a face smaller than
    ``min_face_size`` pixels on its shortest side is rejected rather than
    guessed at. When ``image_min_side`` is provided, the face must also cover
    at least ``min_face_frac`` of the image's shortest side, which keeps the
    gate meaningful after the image is downscaled for detection.
    """

    if not faces:
        raise QualityError(NO_FACE, "no face detected")

    if len(faces) > 1:
        raise QualityError(
            MULTIPLE_FACES,
            f"expected exactly one face, detected {len(faces)}",
        )

    face = faces[0]
    if face.min_side < min_face_size:
        raise QualityError(
            FACE_TOO_SMALL,
            f"face is {face.width:.0f}x{face.height:.0f}px; "
            f"minimum side is {min_face_size}px",
        )

    if image_min_side and min_face_frac > 0.0:
        relative_minimum = min_face_frac * image_min_side
        if face.min_side < relative_minimum:
            raise QualityError(
                FACE_TOO_SMALL,
                f"face is {face.width:.0f}x{face.height:.0f}px; "
                f"minimum is {min_face_frac:.0%} of the image's shortest side "
                f"({relative_minimum:.0f}px)",
            )

    return face
