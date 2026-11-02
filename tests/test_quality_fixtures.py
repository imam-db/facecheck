"""Quality-gate behaviour on synthetic images.

No ONNX models are needed: detections are expressed as ``FaceCandidate``
values, which is exactly what a detector produces before the gate runs.
"""

from __future__ import annotations

import numpy as np
import pytest

from facecheck.embeddings import decode_image_bgr
from facecheck.quality import (
    FACE_TOO_SMALL,
    INVALID_IMAGE,
    MULTIPLE_FACES,
    NO_FACE,
    FaceCandidate,
    QualityError,
    select_face,
)
from tests.conftest import image_with_marker, png_bytes, solid_image


def _candidate(x: float, y: float, width: float, height: float) -> FaceCandidate:
    return FaceCandidate(bbox=(x, y, width, height), landmarks=None, score=0.98)


def test_blank_image_decodes_and_has_no_faces() -> None:
    image = decode_image_bgr(png_bytes(solid_image()))
    assert image.shape == (320, 320, 3)
    with pytest.raises(QualityError) as excinfo:
        select_face([])
    assert excinfo.value.reason == NO_FACE


def test_empty_payload_is_invalid_image() -> None:
    with pytest.raises(QualityError) as excinfo:
        decode_image_bgr(b"")
    assert excinfo.value.reason == INVALID_IMAGE


def test_garbage_bytes_are_invalid_image() -> None:
    with pytest.raises(QualityError) as excinfo:
        decode_image_bgr(b"this is definitely not an image")
    assert excinfo.value.reason == INVALID_IMAGE


def test_two_drawn_boxes_are_multiple_faces() -> None:
    image = image_with_marker()
    person_a = _candidate(10, 10, 90, 90)
    person_b = _candidate(180, 140, 90, 90)
    assert image.shape == (320, 320, 3)
    with pytest.raises(QualityError) as excinfo:
        select_face([person_a, person_b])
    assert excinfo.value.reason == MULTIPLE_FACES
    assert "2" in str(excinfo.value)


def test_small_face_is_rejected() -> None:
    with pytest.raises(QualityError) as excinfo:
        select_face([_candidate(10, 10, 64, 64)], min_face_size=80)
    assert excinfo.value.reason == FACE_TOO_SMALL


def test_min_face_size_boundary_is_inclusive() -> None:
    assert select_face([_candidate(0, 0, 80, 80)], min_face_size=80).min_side == 80
    assert select_face([_candidate(0, 0, 120, 90)], min_face_size=80).width == 120
    with pytest.raises(QualityError):
        select_face([_candidate(0, 0, 79.9, 200)], min_face_size=80)


def test_single_large_face_is_accepted() -> None:
    face = _candidate(40, 50, 160, 150)
    assert select_face([face], min_face_size=80) is face


def test_candidate_min_side_uses_shortest_dimension() -> None:
    assert _candidate(0, 0, 200, 60).min_side == 60


def test_decoded_synthetic_image_matches_drawn_colours() -> None:
    drawn = image_with_marker(color_bgr=(180, 120, 90), marker_bgr=(5, 6, 7))
    decoded = decode_image_bgr(png_bytes(drawn))
    assert np.array_equal(decoded[0, 0], np.array([180, 120, 90], dtype=np.uint8))
    assert np.array_equal(decoded[20, 20], np.array([5, 6, 7], dtype=np.uint8))
