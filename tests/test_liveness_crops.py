"""Runtime crop must stay bit-identical to the scripts reference crop."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from facecheck.liveness.crops import CropImage

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts._crop_image import CropImage as ReferenceCropImage  # noqa: E402

SCALES = (2.7, 4.0)
OUT = 80


def _random_case(seed: int) -> tuple[np.ndarray, list[int], float]:
    rng = np.random.default_rng(seed)
    height = int(rng.integers(40, 400))
    width = int(rng.integers(40, 400))
    image = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)

    x = int(rng.integers(-width // 2, width + width // 2))
    y = int(rng.integers(-height // 2, height + height // 2))
    box_w = int(rng.integers(1, max(2, width)))
    box_h = int(rng.integers(1, max(2, height)))
    scale = SCALES[int(rng.integers(0, len(SCALES)))]
    return image, [x, y, box_w, box_h], scale


@pytest.mark.parametrize("seed", range(40))
def test_runtime_crop_matches_reference_on_random_boxes(seed: int) -> None:
    image, bbox, scale = _random_case(seed)
    runtime = CropImage().crop(image, bbox, scale, OUT, OUT)
    reference = ReferenceCropImage().crop(image, bbox, scale, OUT, OUT)
    assert np.array_equal(runtime, reference), (
        f"seed {seed}: runtime crop differs from reference for bbox={bbox} scale={scale}"
    )


@pytest.mark.parametrize(
    ("bbox", "scale"),
    [
        ([0, 0, 20, 20], 2.7),  # top-left corner
        ([-50, -50, 30, 30], 4.0),  # negative origin
        ([90, 90, 40, 40], 2.7),  # box past the far edge
        ([10, 10, 1, 1], 4.0),  # tiny box, scale clamped
        ([40, 0, 20, 80], 2.7),  # full height, off-frame vertically
        ([120, 120, 80, 80], 4.0),  # fully outside the frame
    ],
)
def test_runtime_crop_matches_reference_on_border_cases(
    bbox: list[int], scale: float
) -> None:
    rng = np.random.default_rng(20261103)
    image = rng.integers(0, 256, size=(160, 160, 3), dtype=np.uint8)
    runtime = CropImage().crop(image, bbox, scale, OUT, OUT)
    reference = ReferenceCropImage().crop(image, bbox, scale, OUT, OUT)
    assert np.array_equal(runtime, reference)
