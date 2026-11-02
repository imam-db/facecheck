from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from facecheck.api import create_app
from facecheck.config import Settings
from facecheck.embeddings import EmbeddingResult
from facecheck.quality import FACE_TOO_SMALL, INVALID_IMAGE, NO_FACE, QualityError
from facecheck.similarity import l2_normalize

TEST_API_KEY = "test-api-key"
TEST_TEMPLATE_KEY = "ZGV2LW9ubHktZmFjZWNoZWNrLXRlbXBsYXRlLWtleSE="
AUTH_HEADERS = {"X-API-Key": TEST_API_KEY}


def models_available() -> bool:
    settings = Settings(_env_file=None)
    return settings.detector_path.is_file() and settings.recognizer_path.is_file()


def png_bytes(image_bgr: np.ndarray) -> bytes:
    rgb = np.asarray(image_bgr, dtype=np.uint8)[:, :, ::-1]
    buffer = io.BytesIO()
    Image.fromarray(rgb, mode="RGB").save(buffer, format="PNG")
    return buffer.getvalue()


def solid_image(
    color_bgr: tuple[int, int, int] = (180, 120, 90),
    size: tuple[int, int] = (320, 320),
) -> np.ndarray:
    width, height = size
    return np.full((height, width, 3), color_bgr, dtype=np.uint8)


def image_with_marker(
    color_bgr: tuple[int, int, int] = (180, 120, 90),
    marker_bgr: tuple[int, int, int] = (10, 10, 10),
    offset: tuple[int, int] = (20, 20),
    marker_size: int = 48,
) -> np.ndarray:
    image = solid_image(color_bgr)
    x, y = offset
    image[y : y + marker_size, x : x + marker_size] = marker_bgr
    return image


class FakeEmbedder:
    """Deterministic embedder: mean colour relative to mid-grey, L2-normalized."""

    model_version = "fake_v1"
    embedding_dim = 3

    def detect_and_embed(self, bgr_image: np.ndarray) -> EmbeddingResult:
        image = np.asarray(bgr_image)
        if image.ndim != 3 or image.shape[2] != 3:
            raise QualityError(INVALID_IMAGE, "expected an HxWx3 image")
        height, width = image.shape[:2]
        if height < 80 or width < 80:
            raise QualityError(FACE_TOO_SMALL, "face too small for fake embedder")
        colors = np.unique(image.reshape(-1, 3), axis=0)
        if len(colors) < 2:
            raise QualityError(NO_FACE, "no face detected")
        mean_bgr = image.reshape(-1, 3).mean(axis=0).astype(np.float32)
        return EmbeddingResult(
            embedding=l2_normalize(mean_bgr - 128.0),
            bbox=(10.0, 10.0, float(width - 20), float(height - 20)),
            landmarks=None,
            model_version=self.model_version,
        )


def make_settings(tmp_path: Path, **overrides) -> Settings:
    values: dict[str, object] = {
        "db_url": f"sqlite:///{tmp_path / 'facecheck-test.db'}",
        "embedding_dim": 3,
        "match_threshold": 0.363,
        "api_key": TEST_API_KEY,
        "template_key": TEST_TEMPLATE_KEY,
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return make_settings(tmp_path)


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def client(settings: Settings, fake_embedder: FakeEmbedder):
    app = create_app(settings, embedder=fake_embedder)
    with TestClient(app, headers=AUTH_HEADERS) as test_client:
        yield test_client
