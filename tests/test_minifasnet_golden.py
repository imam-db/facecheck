"""Semantic golden tests for the exported MiniFASNet PAD models.

The reference images live outside the repository and are supplied through
``FACECHECK_PAD_GOLDEN_DIR`` (schema: ``docs/pad_golden_manifest.md``). When the
variable is unset, the directory or ``manifest.json`` is missing, or the
MiniFASNet ONNX files have not been exported, these tests skip. No image is
ever committed.

The correct-score path goes through the **runtime**
``MiniFASNetLivenessDetector`` so the golden numbers exercise the shipped
preprocessing (runtime ``CropImage``, raw BGR [0, 255] NCHW). The scripts crop
port stays only as a bit-identical reference in ``tests/test_liveness_crops.py``.

For each case the test checks:

- the real probability matches the manifest within ``1e-3`` for **every** case,
  using the detector's averaged ``score`` and its per-model probabilities;
- dividing by 255 drives *discriminative* cases below ``0.1`` (and, for real
  cases, well away from the expected value), because the models were trained on
  raw pixels;
- swapping BGR/RGB is fail-open for discriminative spoof cases: the swapped crop
  looks at least ``0.5`` more real than the correct crop.

The wrong-preprocessing paths build intentional bad blobs locally from the
runtime ``CropImage`` so a regression is still caught without adding wrong-path
options to the detector API.

``discriminative`` defaults to ``False`` when absent (see
``docs/pad_golden_manifest.md``), so the contrast assertions only run where the
model already classifies the correct crop correctly.
"""

from __future__ import annotations

import json
import os
from functools import cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

from facecheck.config import Settings
from facecheck.liveness.crops import CropImage
from facecheck.liveness.minifasnet import (
    INPUT_SIZE,
    REAL_CLASS_INDEX,
    MiniFASNetLivenessDetector,
)
from tests.pad_golden_policy import (
    missing_discriminative_labels,
    should_assert_div255,
    should_assert_rgb_swap_gain,
)

pytestmark = pytest.mark.minifasnet

GOLDEN_ENV = "FACECHECK_PAD_GOLDEN_DIR"
GOLDEN_TOLERANCE = 1e-3
DIV255_MAX = 0.1
DIV255_MIN_GAP = 0.5
FAIL_OPEN_MARGIN = 0.5

MODEL_SCALES = {
    "2.7_80x80_MiniFASNetV2.onnx": 2.7,
    "4_0_0_80x80_MiniFASNetV1SE.onnx": 4.0,
}

_golden_value = os.environ.get(GOLDEN_ENV)
if not _golden_value:
    pytest.skip(
        f"{GOLDEN_ENV} is not set; semantic golden images are kept out of the repo",
        allow_module_level=True,
    )

GOLDEN_DIR = Path(_golden_value)
MANIFEST_PATH = GOLDEN_DIR / "manifest.json"
if not MANIFEST_PATH.is_file():
    pytest.skip(f"missing golden manifest {MANIFEST_PATH}", allow_module_level=True)

MANIFEST: dict[str, Any] = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
CASES: list[dict[str, Any]] = MANIFEST.get("cases") or []
if not CASES:
    pytest.skip(f"{MANIFEST_PATH} has no cases", allow_module_level=True)

MODEL_NAMES = sorted(MODEL_SCALES)


def _models_dir() -> Path:
    return Settings(_env_file=None).models_dir


def _missing_models() -> list[str]:
    models_dir = _models_dir()
    return [name for name in MODEL_NAMES if not (models_dir / name).is_file()]


requires_models = pytest.mark.skipif(
    bool(_missing_models()),
    reason=f"MiniFASNet ONNX models not exported: {', '.join(_missing_models())}",
)


@cache
def _detector() -> MiniFASNetLivenessDetector:
    return MiniFASNetLivenessDetector.from_settings(Settings(_env_file=None))


@cache
def _cv2_net(path: str):
    return cv2.dnn.readNetFromONNX(path)


def _softmax(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64).reshape(-1)
    shifted = values - values.max()
    exp = np.exp(shifted)
    return exp / exp.sum()


def _blob(crop: np.ndarray, *, div255: bool = False, rgb_swap: bool = False) -> np.ndarray:
    image = crop[:, :, ::-1] if rgb_swap else crop
    array = np.asarray(image, dtype=np.float32)
    if div255:
        array = array / 255.0
    return np.ascontiguousarray(array.transpose(2, 0, 1)[None])


def _wrong_preprocessing_probs(
    image: np.ndarray,
    bbox_xywh: list[int],
    *,
    div255: bool = False,
    rgb_swap: bool = False,
) -> dict[str, float]:
    """Score intentionally wrong blobs built from the runtime crop."""

    cropper = CropImage()
    probs: dict[str, float] = {}
    for name, scale in MODEL_SCALES.items():
        crop = cropper.crop(image, bbox_xywh, scale, INPUT_SIZE, INPUT_SIZE)
        net = _cv2_net(str(_models_dir() / name))
        net.setInput(_blob(crop, div255=div255, rgb_swap=rgb_swap))
        probs[name] = float(_softmax(net.forward())[REAL_CLASS_INDEX])
    return probs


def _average(probs: dict[str, float]) -> float:
    return float(np.mean(list(probs.values())))


def test_golden_set_has_discriminative_coverage() -> None:
    """The manifest must exercise both labels with a discriminative case."""

    missing = missing_discriminative_labels(CASES)
    assert not missing, (
        f"{MANIFEST_PATH}: no discriminative case for label(s) "
        f"{', '.join(missing)}; mark at least one case per label with "
        '"discriminative": true'
    )


@pytest.mark.parametrize("case", CASES, ids=[case["id"] for case in CASES])
@requires_models
def test_golden_real_probability(case: dict[str, Any]) -> None:
    image_path = GOLDEN_DIR / case["image"]
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    assert image is not None, f"could not read {image_path}"

    bbox = [int(value) for value in case["bbox_xywh"]]
    expected = float(case["expected_real_prob"])

    result = _detector().score(image, bbox)
    assert abs(result.score - expected) < GOLDEN_TOLERANCE, (
        f"{case['id']}: expected_real_prob {expected:.6f}, got {result.score:.6f}"
    )

    for name, probability in (case.get("expected_per_model") or {}).items():
        assert abs(result.per_model[name] - float(probability)) < GOLDEN_TOLERANCE


@pytest.mark.parametrize("case", CASES, ids=[case["id"] for case in CASES])
@requires_models
def test_golden_div255_is_wrong(case: dict[str, Any]) -> None:
    if not should_assert_div255(case):
        pytest.skip("case is not marked discriminative; /255 contrast is not observable")

    image_path = GOLDEN_DIR / case["image"]
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    assert image is not None, f"could not read {image_path}"

    bbox = [int(value) for value in case["bbox_xywh"]]
    expected = float(case["expected_real_prob"])
    average = _average(_wrong_preprocessing_probs(image, bbox, div255=True))

    # Dividing by 255 collapses every face toward the attack classes.
    assert average < DIV255_MAX, (
        f"{case['id']}: /255 real probability {average:.6f} >= {DIV255_MAX}"
    )
    if case["label"] == "real":
        assert abs(average - expected) > DIV255_MIN_GAP, (
            f"{case['id']}: /255 real probability {average:.6f} stayed close "
            f"to expected {expected:.6f}"
        )


@pytest.mark.parametrize("case", CASES, ids=[case["id"] for case in CASES])
@requires_models
def test_golden_rgb_swap_is_fail_open(case: dict[str, Any]) -> None:
    if not should_assert_rgb_swap_gain(case):
        pytest.skip("RGB-swap gain applies only to discriminative spoof cases")

    image_path = GOLDEN_DIR / case["image"]
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    assert image is not None, f"could not read {image_path}"

    bbox = [int(value) for value in case["bbox_xywh"]]
    expected = float(case["expected_real_prob"])
    average = _average(_wrong_preprocessing_probs(image, bbox, rgb_swap=True))

    # A channel swap makes attacks look real: the failure mode is fail-open.
    assert average - expected > FAIL_OPEN_MARGIN, (
        f"{case['id']}: RGB-swapped real probability {average:.6f} was not "
        f"{FAIL_OPEN_MARGIN} above expected {expected:.6f}"
    )
