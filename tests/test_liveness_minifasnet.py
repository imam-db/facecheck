"""Tests for the runtime MiniFASNet liveness detector.

Model-backed tests need the two exported ONNX files and skip when they are
missing. They check the model version format, input validation, the two-model
softmax plumbing, and that ``score_face`` maps the detection box. Semantic
golden numbers live in ``tests/test_minifasnet_golden.py``.

Input validation and the missing-model path do not need the ONNX files, so they
always run.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

import numpy as np
import pytest

from facecheck.config import Settings
from facecheck.embeddings import SHA_PREFIX_LENGTH, EmbeddingResult, file_sha256
from facecheck.liveness.base import (
    LivenessDetector,
    LivenessInputError,
    LivenessModelNotFoundError,
    LivenessResult,
)
from facecheck.liveness.minifasnet import (
    MiniFASNetLivenessDetector,
    validate_liveness_inputs,
)

pytestmark = pytest.mark.minifasnet

SETTINGS = Settings(_env_file=None)
MISSING = [
    str(path)
    for path in (SETTINGS.pad_v2_path, SETTINGS.pad_v1se_path)
    if not path.is_file()
]
requires_models = pytest.mark.skipif(
    bool(MISSING), reason=f"MiniFASNet ONNX models not exported: {', '.join(MISSING)}"
)


@pytest.fixture(scope="module")
def detector() -> MiniFASNetLivenessDetector:
    return MiniFASNetLivenessDetector.from_settings(SETTINGS)


def _assert_bounded_result(result: LivenessResult, expected_names: Sequence[str]) -> None:
    assert 0.0 <= result.score <= 1.0
    assert set(result.per_model) == set(expected_names)
    for probability in result.per_model.values():
        assert 0.0 <= probability <= 1.0
    assert result.score == pytest.approx(float(np.mean(list(result.per_model.values()))))


def test_validate_accepts_uint8_bgr_image() -> None:
    image = np.zeros((40, 60, 3), dtype=np.uint8)
    validated, bbox = validate_liveness_inputs(image, (1.9, 2.9, 10.9, 20.1))
    assert validated is image
    assert bbox == [1, 2, 10, 20]


def test_validate_rejects_non_array() -> None:
    with pytest.raises(LivenessInputError, match="numpy array"):
        validate_liveness_inputs([[0, 0, 0]], [0, 0, 10, 10])  # type: ignore[arg-type]


def test_validate_rejects_float_image() -> None:
    image = np.zeros((40, 40, 3), dtype=np.float32)
    with pytest.raises(LivenessInputError, match="uint8"):
        validate_liveness_inputs(image, [0, 0, 10, 10])


def test_validate_rejects_grayscale_image() -> None:
    image = np.zeros((40, 40), dtype=np.uint8)
    with pytest.raises(LivenessInputError, match="HxWx3"):
        validate_liveness_inputs(image, [0, 0, 10, 10])


def test_validate_rejects_bgra_image() -> None:
    image = np.zeros((40, 40, 4), dtype=np.uint8)
    with pytest.raises(LivenessInputError, match="HxWx3"):
        validate_liveness_inputs(image, [0, 0, 10, 10])


@pytest.mark.parametrize("bbox", [[0, 0, 0, 10], [0, 0, 10, 0], [0, 0, -5, 10]])
def test_validate_rejects_zero_or_negative_box(bbox: list[int]) -> None:
    image = np.zeros((40, 40, 3), dtype=np.uint8)
    with pytest.raises(LivenessInputError, match="positive width and height"):
        validate_liveness_inputs(image, bbox)


def test_validate_rejects_wrong_box_length() -> None:
    image = np.zeros((40, 40, 3), dtype=np.uint8)
    with pytest.raises(LivenessInputError, match="4 components"):
        validate_liveness_inputs(image, [0, 0, 10])


def test_missing_model_raises(tmp_path) -> None:
    with pytest.raises(LivenessModelNotFoundError, match="download_models"):
        MiniFASNetLivenessDetector(
            tmp_path / "missing_v2.onnx", tmp_path / "missing_v1.onnx"
        )


@requires_models
def test_model_version_uses_both_sha_prefixes(
    detector: MiniFASNetLivenessDetector,
) -> None:
    v2 = SETTINGS.pad_v2_path
    v1se = SETTINGS.pad_v1se_path
    expected = (
        f"{v2.stem}@{file_sha256(v2)[:SHA_PREFIX_LENGTH]}+"
        f"{v1se.stem}@{file_sha256(v1se)[:SHA_PREFIX_LENGTH]}"
    )
    assert detector.model_version == expected
    assert re.fullmatch(r"[^@]+@[0-9a-f]{8}\+[^@]+@[0-9a-f]{8}", detector.model_version)


@requires_models
def test_detector_satisfies_protocol(detector: MiniFASNetLivenessDetector) -> None:
    assert isinstance(detector, LivenessDetector)


@requires_models
def test_score_is_bounded_and_averages_both_models(
    detector: MiniFASNetLivenessDetector,
) -> None:
    rng = np.random.default_rng(20261103)
    image = rng.integers(0, 256, size=(240, 240, 3), dtype=np.uint8)
    result = detector.score(image, [60, 60, 120, 140])
    _assert_bounded_result(result, [SETTINGS.pad_v2_file, SETTINGS.pad_v1se_file])
    assert result.aspect_ratio == pytest.approx(image.shape[1] / image.shape[0])
    assert result.model_version == detector.model_version


@requires_models
def test_score_face_maps_box(detector: MiniFASNetLivenessDetector) -> None:
    rng = np.random.default_rng(20261104)
    full_res_image = rng.integers(0, 256, size=(480, 640, 3), dtype=np.uint8)
    embedding_result = EmbeddingResult(
        embedding=np.zeros(3, dtype=np.float32),
        bbox=(60.0, 80.0, 120.0, 140.0),
        landmarks=None,
        model_version="fake",
        detection_scale=0.5,
    )

    expected = detector.score(full_res_image, [120, 160, 240, 280])
    result = detector.score_face(full_res_image, embedding_result)

    assert result.score == pytest.approx(expected.score)
    assert result.per_model == pytest.approx(expected.per_model)
    assert result.aspect_ratio == pytest.approx(640 / 480)
