"""Tests for the runtime MiniFASNet liveness detector.

These need the two exported ONNX files and skip when they are missing. They
check the model version format and that the two-model softmax plumbing yields a
bounded probability; semantic golden numbers live in
``tests/test_minifasnet_golden.py``.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

import numpy as np
import pytest

from facecheck.config import Settings
from facecheck.embeddings import SHA_PREFIX_LENGTH, file_sha256
from facecheck.liveness.base import LivenessDetector, LivenessResult
from facecheck.liveness.minifasnet import MiniFASNetLivenessDetector

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
    assert result.aspect_ratio == pytest.approx(120 / 140)
    assert result.model_version == detector.model_version


@requires_models
def test_missing_model_raises(tmp_path) -> None:
    from facecheck.embeddings import ModelNotFoundError

    with pytest.raises(ModelNotFoundError, match="download_models"):
        MiniFASNetLivenessDetector(tmp_path / "missing_v2.onnx", tmp_path / "missing_v1.onnx")
