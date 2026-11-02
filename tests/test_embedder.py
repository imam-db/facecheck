"""Smoke tests for the real YuNet + SFace embedder.

These require downloaded ONNX models and are skipped otherwise. A real face
fixture is used when an LFW cache is present; the model-loading and quality
paths are tested without biometric data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from facecheck.config import Settings
from facecheck.embeddings import (
    ModelNotFoundError,
    OpenCVYuNetSFaceEmbedder,
    decode_image_bgr,
)
from facecheck.quality import NO_FACE, QualityError
from facecheck.similarity import cosine_similarity
from tests.conftest import models_available, png_bytes, solid_image

pytestmark = [
    pytest.mark.model,
    pytest.mark.skipif(not models_available(), reason="ONNX models not downloaded"),
]


@pytest.fixture(scope="module")
def embedder() -> OpenCVYuNetSFaceEmbedder:
    return OpenCVYuNetSFaceEmbedder.from_settings(Settings(_env_file=None))


def _lfw_samples(limit: int = 4) -> list[Path]:
    roots = [
        Path("data/lfw_home/lfw_deepfunneled"),
        Path("data/lfw_home/lfw_funneled"),
        Path.home() / "scikit_learn_data/lfw_home/lfw_funneled",
    ]
    for root in roots:
        if root.is_dir():
            images = sorted(root.glob("**/*.jpg"))[:limit]
            if images:
                return images
    return []


def test_blank_image_has_no_face(embedder: OpenCVYuNetSFaceEmbedder) -> None:
    image = decode_image_bgr(png_bytes(solid_image()))
    with pytest.raises(QualityError) as excinfo:
        embedder.detect_and_embed(image)
    assert excinfo.value.reason == NO_FACE


def test_missing_models_raise_model_not_found(tmp_path: Path) -> None:
    with pytest.raises(ModelNotFoundError, match="download_models"):
        OpenCVYuNetSFaceEmbedder(tmp_path / "a.onnx", tmp_path / "b.onnx")


def test_embedding_is_dim_128_and_normalized(embedder: OpenCVYuNetSFaceEmbedder) -> None:
    samples = _lfw_samples(limit=1)
    if not samples:
        pytest.skip("no LFW image available for a real-face embedding")

    result = embedder.detect_and_embed(decode_image_bgr(samples[0].read_bytes()))
    assert result.embedding.dtype == np.float32
    assert result.embedding.shape == (128,)
    assert np.isclose(np.linalg.norm(result.embedding), 1.0, atol=1e-5)
    assert result.model_version == (
        "face_detection_yunet_2023mar+face_recognition_sface_2021dec"
    )
    assert result.bbox[2] >= 80 and result.bbox[3] >= 80


def test_same_image_scores_one(embedder: OpenCVYuNetSFaceEmbedder) -> None:
    samples = _lfw_samples(limit=1)
    if not samples:
        pytest.skip("no LFW image available for a real-face embedding")

    image = decode_image_bgr(samples[0].read_bytes())
    first = embedder.detect_and_embed(image)
    second = embedder.detect_and_embed(image)
    assert cosine_similarity(first.embedding, second.embedding) == pytest.approx(1.0, abs=1e-6)
