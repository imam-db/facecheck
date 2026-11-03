"""Unit tests for mapping detection boxes back to full resolution."""

from __future__ import annotations

import pytest

from facecheck.liveness.boxes import map_detection_box_to_full_res


def test_scale_one_still_truncates_each_component() -> None:
    mapped = map_detection_box_to_full_res([10.9, 20.4, 30.7, 40.2], 1.0)
    assert mapped == [10, 20, 30, 40]
    assert all(isinstance(value, int) for value in mapped)


def test_scale_below_one_expands_and_truncates() -> None:
    # 960 / 2000 = 0.48; upstream-style int() truncation, not round().
    mapped = map_detection_box_to_full_res([100.0, 50.0, 60.0, 90.0], 0.48)
    assert mapped == [208, 104, 125, 187]


def test_truncation_pins_known_floats() -> None:
    # 1.9 / 0.5 = 3.8 -> 3 (int truncation, not 4).
    assert map_detection_box_to_full_res([1.9, 3.1, 7.9, 9.9], 0.5) == [3, 6, 15, 19]


def test_non_integer_scale_round_trip_shape() -> None:
    original = [106.0, 147.0, 207.0, 213.0]
    mapped = map_detection_box_to_full_res(original, 1.0)
    assert mapped == [106, 147, 207, 213]


def test_zero_or_negative_scale_rejected() -> None:
    with pytest.raises(ValueError, match="positive"):
        map_detection_box_to_full_res([1, 2, 3, 4], 0.0)
    with pytest.raises(ValueError, match="positive"):
        map_detection_box_to_full_res([1, 2, 3, 4], -0.5)
