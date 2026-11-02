"""API tests with a deterministic fake embedder.

These run everywhere. Tests that need the real ONNX models live in
``TestModelBackedApi`` and skip when the model files are absent.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from facecheck.api import create_app
from facecheck.config import ConfigurationError
from facecheck.embeddings import EmbedderError, ModelNotFoundError
from facecheck.service import FaceVerificationService
from tests.conftest import (
    AUTH_HEADERS,
    TEST_API_KEY,
    FakeEmbedder,
    image_with_marker,
    make_settings,
    models_available,
    png_bytes,
    solid_image,
)

ENROLL_COLORS = (180, 120, 90)
CONSENT = {
    "consent_version": "2026-11",
    "consent_at": "2026-11-01T09:00:00Z",
}

PLACEHOLDER_API_KEY = "dev-api-key"
PLACEHOLDER_TEMPLATE_KEY = "ZGV2LW9ubHktZmFjZWNoZWNrLXRlbXBsYXRlLWtleSE="
ALT_TEMPLATE_KEY = "a1b2c3d4e5f60718293a4b5c6d7e8f90" * 2


def _enroll_images(count: int = 3) -> list[tuple[str, tuple[str, bytes, str]]]:
    return [
        (
            "files",
            (
                "face.png",
                png_bytes(
                    image_with_marker(
                        color_bgr=ENROLL_COLORS,
                        offset=(20 + 10 * index, 20 + 10 * index),
                    )
                ),
                "image/png",
            ),
        )
        for index in range(count)
    ]


def _file(name: str, data: bytes) -> tuple[str, tuple[str, bytes, str]]:
    return ("files", (name, data, "image/png"))


def _create_person(client: TestClient, **payload) -> str:
    body = {**CONSENT, **payload}
    response = client.post("/v1/persons", json=body)
    assert response.status_code == 201, response.text
    return response.json()["id"]


class RaisingEmbedder:
    def __init__(self, error: Exception) -> None:
        self.error = error
        self.model_version = "broken_v1"
        self.embedding_dim = 3

    def detect_and_embed(self, bgr_image: np.ndarray):
        raise self.error


def test_meta_reports_model_and_threshold(client: TestClient) -> None:
    response = client.get("/v1/meta")
    assert response.status_code == 200
    body = response.json()
    assert body == {
        "model_version": "fake_v1",
        "threshold": 0.363,
        "enroll_threshold": 0.363,
        "embedding_dim": 3,
        "min_face_size": 80,
        "min_face_frac": 0.15,
        "detect_max_side": 960,
    }


def test_healthz_is_open_without_api_key(settings) -> None:
    app = create_app(settings, embedder=FakeEmbedder())
    with TestClient(app) as anonymous:
        assert anonymous.get("/healthz").json() == {"status": "ok"}
        unauthorized = anonymous.get("/v1/meta")
        assert unauthorized.status_code == 401
        assert unauthorized.json()["code"] == "unauthorized"


def test_api_key_accepts_header_and_bearer(settings) -> None:
    app = create_app(settings, embedder=FakeEmbedder())
    with TestClient(app) as anonymous:
        assert anonymous.get("/v1/meta", headers=AUTH_HEADERS).status_code == 200
        bearer = {"Authorization": f"Bearer {TEST_API_KEY}"}
        assert anonymous.get("/v1/meta", headers=bearer).status_code == 200
        wrong = {"X-API-Key": "not-the-key"}
        assert anonymous.get("/v1/meta", headers=wrong).status_code == 401


@pytest.mark.parametrize("overrides", [{"api_key": None}, {"template_key": None}])
def test_startup_refuses_without_keys(tmp_path: Path, overrides: dict) -> None:
    settings = make_settings(tmp_path, **overrides)
    app = create_app(settings, embedder=FakeEmbedder())
    with pytest.raises(ConfigurationError, match="not set"), TestClient(app):
        pass


def test_startup_refuses_placeholders_in_production(tmp_path: Path) -> None:
    settings = make_settings(
        tmp_path,
        dev_mode=False,
        api_key=PLACEHOLDER_API_KEY,
        template_key=PLACEHOLDER_TEMPLATE_KEY,
    )
    app = create_app(settings, embedder=FakeEmbedder())
    with pytest.raises(ConfigurationError, match="placeholder"), TestClient(app):
        pass


def test_dev_mode_allows_placeholder_keys(tmp_path: Path) -> None:
    settings = make_settings(
        tmp_path,
        dev_mode=True,
        api_key=PLACEHOLDER_API_KEY,
        template_key=PLACEHOLDER_TEMPLATE_KEY,
    )
    app = create_app(settings, embedder=FakeEmbedder())
    with TestClient(app, headers={"X-API-Key": PLACEHOLDER_API_KEY}) as client:
        assert client.get("/v1/meta").status_code == 200


def test_startup_accepts_strong_keys_in_production(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, dev_mode=False)
    app = create_app(settings, embedder=FakeEmbedder())
    with TestClient(app, headers=AUTH_HEADERS) as client:
        assert client.get("/v1/meta").status_code == 200


def test_non_ascii_api_key_is_unauthorized_not_server_error(settings) -> None:
    app = create_app(settings, embedder=FakeEmbedder())
    with TestClient(app) as anonymous:
        utf8 = anonymous.get(
            "/v1/meta", headers={"X-API-Key": "café-key".encode()}
        )
        assert utf8.status_code == 401
        assert utf8.json()["code"] == "unauthorized"

        latin1 = anonymous.get("/v1/meta", headers={"X-API-Key": b"caf\xe9-key"})
        assert latin1.status_code == 401
        assert latin1.json()["code"] == "unauthorized"


def test_validation_error_uses_unified_envelope(client: TestClient) -> None:
    response = client.post("/v1/persons", json={"external_id": "missing-consent"})
    assert response.status_code == 422
    body = response.json()
    assert body["code"] == "validation_error"
    assert isinstance(body["message"], str) and body["message"]
    assert isinstance(body["details"], list) and body["details"]


def test_template_key_change_returns_409(tmp_path: Path) -> None:
    first = make_settings(tmp_path)
    first_app = create_app(first, embedder=FakeEmbedder())
    with TestClient(first_app, headers=AUTH_HEADERS) as client:
        person_id = _create_person(client)
        enrolled = client.post(
            f"/v1/persons/{person_id}/enroll", files=_enroll_images(3)
        )
        assert enrolled.status_code == 200, enrolled.text

    second = make_settings(tmp_path, template_key=ALT_TEMPLATE_KEY)
    second_app = create_app(second, embedder=FakeEmbedder())
    with TestClient(second_app, headers=AUTH_HEADERS) as client:
        response = client.post(
            f"/v1/persons/{person_id}/verify",
            files={"file": ("probe.png", png_bytes(image_with_marker()))},
        )

    assert response.status_code == 409
    body = response.json()
    assert body["code"] == "template_decrypt_failed"
    assert "re-enroll" in body["message"].lower()
    assert body["details"]["reason"] == "invalid_key"


def test_create_person_records_consent(client: TestClient) -> None:
    response = client.post(
        "/v1/persons", json={**CONSENT, "external_id": "emp-1"}
    )
    assert response.status_code == 201
    body = response.json()
    assert body["external_id"] == "emp-1"
    assert body["consent_version"] == "2026-11"
    assert body["consent_at"].startswith("2026-11-01")
    assert len(body["id"]) == 36


def test_create_person_requires_consent(client: TestClient) -> None:
    assert client.post("/v1/persons").status_code == 422
    response = client.post("/v1/persons", json={"external_id": "emp-2"})
    assert response.status_code == 422


def test_duplicate_external_id_is_conflict(client: TestClient) -> None:
    _create_person(client, external_id="emp-42")
    response = client.post("/v1/persons", json={**CONSENT, "external_id": "emp-42"})
    assert response.status_code == 409
    body = response.json()
    assert body["code"] == "duplicate_person"
    assert "emp-42" in body["message"]


def test_enroll_stores_requested_templates(client: TestClient) -> None:
    person_id = _create_person(client)
    response = client.post(
        f"/v1/persons/{person_id}/enroll", files=_enroll_images(3)
    )
    assert response.status_code == 200, response.text
    assert response.json() == {"enrolled": 3, "model_version": "fake_v1"}

    service: FaceVerificationService = client.app.state.service
    assert len(service.storage.list_templates(person_id)) == 3


def test_enroll_requires_three_to_five_images(client: TestClient) -> None:
    person_id = _create_person(client)

    too_few = client.post(f"/v1/persons/{person_id}/enroll", files=_enroll_images(2))
    assert too_few.status_code == 400
    assert too_few.json()["code"] == "invalid_enrollment"
    assert "between 3 and 5" in too_few.json()["message"]

    too_many = client.post(f"/v1/persons/{person_id}/enroll", files=_enroll_images(6))
    assert too_many.status_code == 400


def test_enroll_rejects_blank_image_with_indexed_reason(client: TestClient) -> None:
    person_id = _create_person(client)
    files = [
        _file("face.png", png_bytes(image_with_marker())),
        _file("blank.png", png_bytes(solid_image())),
        _file("face2.png", png_bytes(image_with_marker())),
    ]
    response = client.post(f"/v1/persons/{person_id}/enroll", files=files)
    assert response.status_code == 400
    body = response.json()
    assert body["code"] == "invalid_enrollment"
    assert body["details"]["reasons"] == [
        {"index": 1, "reason": "no_face", "message": "no face detected"}
    ]


def test_enroll_rejects_mixed_faces_as_inconsistent(client: TestClient) -> None:
    person_id = _create_person(client)
    files = [
        _file("a.png", png_bytes(image_with_marker())),
        _file("b.png", png_bytes(image_with_marker(offset=(40, 40)))),
        _file("c.png", png_bytes(image_with_marker(color_bgr=(30, 200, 60)))),
    ]
    response = client.post(f"/v1/persons/{person_id}/enroll", files=files)
    assert response.status_code == 400
    body = response.json()
    assert body["code"] == "inconsistent_enrollment"
    assert body["details"]["reason"] == "inconsistent_enrollment"
    assert body["details"]["min_similarity"] < body["details"]["threshold"]

    service: FaceVerificationService = client.app.state.service
    assert service.storage.list_templates(person_id) == []


def test_inconsistent_re_enrollment_keeps_previous_templates(client: TestClient) -> None:
    person_id = _create_person(client)
    assert client.post(
        f"/v1/persons/{person_id}/enroll", files=_enroll_images(3)
    ).status_code == 200
    response = client.post(
        f"/v1/persons/{person_id}/enroll",
        files=[
            _file("a.png", png_bytes(image_with_marker())),
            _file("b.png", png_bytes(image_with_marker())),
            _file("c.png", png_bytes(image_with_marker(color_bgr=(30, 200, 60)))),
        ],
    )
    assert response.status_code == 400
    service: FaceVerificationService = client.app.state.service
    assert len(service.storage.list_templates(person_id)) == 3


def test_enroll_requires_recorded_consent(client: TestClient) -> None:
    service: FaceVerificationService = client.app.state.service
    person = service.storage.create_person()
    response = client.post(
        f"/v1/persons/{person.id}/enroll", files=_enroll_images(3)
    )
    assert response.status_code == 409
    assert response.json()["code"] == "consent_required"


def test_enroll_unknown_person_is_404(client: TestClient) -> None:
    response = client.post(
        "/v1/persons/00000000-0000-0000-0000-000000000000/enroll",
        files=_enroll_images(3),
    )
    assert response.status_code == 404
    assert response.json()["code"] == "person_not_found"


def test_enroll_invalid_image_bytes(client: TestClient) -> None:
    person_id = _create_person(client)
    response = client.post(
        f"/v1/persons/{person_id}/enroll",
        files=[
            _file("face.png", b"not-an-image"),
            _file("face.png", png_bytes(image_with_marker())),
            _file("face.png", png_bytes(image_with_marker())),
        ],
    )
    assert response.status_code == 400
    assert response.json()["details"]["reason"] == "invalid_image"


def test_enroll_replaces_previous_templates(client: TestClient) -> None:
    person_id = _create_person(client)
    assert client.post(
        f"/v1/persons/{person_id}/enroll", files=_enroll_images(3)
    ).status_code == 200
    assert client.post(
        f"/v1/persons/{person_id}/enroll",
        files=[
            _file("face.png", png_bytes(image_with_marker(offset=(100, 100)))),
            _file("face.png", png_bytes(image_with_marker(offset=(110, 110)))),
            _file("face.png", png_bytes(image_with_marker(offset=(120, 120)))),
        ],
    ).status_code == 200

    service: FaceVerificationService = client.app.state.service
    assert len(service.storage.list_templates(person_id)) == 3


def test_verify_same_face_matches(client: TestClient) -> None:
    person_id = _create_person(client)
    client.post(f"/v1/persons/{person_id}/enroll", files=_enroll_images(3))

    response = client.post(
        f"/v1/persons/{person_id}/verify",
        files={"file": ("probe.png", png_bytes(image_with_marker(offset=(60, 60))))},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["decision"] == "match"
    assert body["score"] > 0.9
    assert body["threshold"] == 0.363
    assert body["model_version"] == "fake_v1"


def test_verify_different_face_does_not_match(client: TestClient) -> None:
    person_id = _create_person(client)
    client.post(f"/v1/persons/{person_id}/enroll", files=_enroll_images(3))

    response = client.post(
        f"/v1/persons/{person_id}/verify",
        files={"file": ("other.png", png_bytes(image_with_marker(color_bgr=(30, 200, 60))))},
    )
    assert response.status_code == 200
    assert response.json()["decision"] == "no_match"
    assert response.json()["score"] < 0.363


def test_verify_person_without_templates_is_conflict(client: TestClient) -> None:
    person_id = _create_person(client)
    response = client.post(
        f"/v1/persons/{person_id}/verify",
        files={"file": ("probe.png", png_bytes(image_with_marker()))},
    )
    assert response.status_code == 409
    assert response.json()["code"] == "verification_failed"
    assert "no enrolled templates" in response.json()["message"]


def test_verify_model_version_mismatch_is_conflict(client: TestClient) -> None:
    person_id = _create_person(client)
    service: FaceVerificationService = client.app.state.service
    service.storage.replace_templates(
        person_id, [np.array([0.1, 0.2, 0.3], dtype=np.float32)], model_version="old_v0"
    )

    response = client.post(
        f"/v1/persons/{person_id}/verify",
        files={"file": ("probe.png", png_bytes(image_with_marker()))},
    )
    assert response.status_code == 409
    assert "re-enroll" in response.json()["message"]


def test_verify_unknown_person_is_404(client: TestClient) -> None:
    response = client.post(
        "/v1/persons/00000000-0000-0000-0000-000000000000/verify",
        files={"file": ("probe.png", png_bytes(image_with_marker()))},
    )
    assert response.status_code == 404


def test_verify_quality_failure_is_400(client: TestClient) -> None:
    person_id = _create_person(client)
    client.post(f"/v1/persons/{person_id}/enroll", files=_enroll_images(3))
    response = client.post(
        f"/v1/persons/{person_id}/verify",
        files={"file": ("blank.png", png_bytes(solid_image()))},
    )
    assert response.status_code == 400
    assert response.json()["code"] == "quality_error"
    assert response.json()["details"]["reason"] == "no_face"


def test_delete_person_cascades_and_returns_204(client: TestClient) -> None:
    person_id = _create_person(client)
    client.post(f"/v1/persons/{person_id}/enroll", files=_enroll_images(3))
    service: FaceVerificationService = client.app.state.service

    assert client.delete(f"/v1/persons/{person_id}").status_code == 204
    assert service.storage.get_person(person_id) is None
    assert service.storage.list_templates(person_id) == []
    assert client.delete(f"/v1/persons/{person_id}").status_code == 404


def test_upload_limit_is_enforced(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, max_upload_bytes=100)
    app = create_app(settings, embedder=FakeEmbedder())
    with TestClient(app, headers=AUTH_HEADERS) as client:
        person_id = _create_person(client)
        response = client.post(
            f"/v1/persons/{person_id}/verify",
            files={"file": ("probe.png", png_bytes(image_with_marker()))},
        )
    assert response.status_code == 400
    assert response.json()["code"] == "quality_error"
    assert "upload limit" in response.json()["message"]


def test_pixel_limit_is_checked_from_the_header(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, max_pixels=10_000)
    app = create_app(settings, embedder=FakeEmbedder())
    with TestClient(app, headers=AUTH_HEADERS) as client:
        person_id = _create_person(client)
        response = client.post(
            f"/v1/persons/{person_id}/verify",
            files={"file": ("probe.png", png_bytes(image_with_marker()))},
        )
    assert response.status_code == 400
    assert response.json()["details"]["reason"] == "invalid_image"
    assert "pixels" in response.json()["message"]


def test_alignment_failure_maps_to_422(client: TestClient) -> None:
    service: FaceVerificationService = client.app.state.service
    person_id = _create_person(client)
    service.embedder = RaisingEmbedder(EmbedderError("face alignment failed"))
    response = client.post(
        f"/v1/persons/{person_id}/enroll", files=_enroll_images(3)
    )
    assert response.status_code == 422
    assert response.json()["code"] == "embedder_error"


def test_missing_model_maps_to_503(client: TestClient) -> None:
    service: FaceVerificationService = client.app.state.service
    person_id = _create_person(client)
    service.embedder = RaisingEmbedder(ModelNotFoundError("missing model file"))
    response = client.post(
        f"/v1/persons/{person_id}/enroll", files=_enroll_images(3)
    )
    assert response.status_code == 503
    assert response.json()["code"] == "model_not_found"


@pytest.mark.model
@pytest.mark.skipif(not models_available(), reason="ONNX models not downloaded")
class TestModelBackedApi:
    @pytest.fixture
    def real_client(self, tmp_path: Path):
        settings = make_settings(tmp_path, embedding_dim=128)
        app = create_app(settings)
        with TestClient(app, headers=AUTH_HEADERS) as client:
            yield client

    def test_meta_reports_real_model_version(self, real_client: TestClient) -> None:
        body = real_client.get("/v1/meta").json()
        version = body["model_version"]
        assert version.startswith("face_detection_yunet_2023mar@")
        assert "+face_recognition_sface_2021dec@" in version
        assert body["embedding_dim"] == 128

    def test_blank_enrollment_is_rejected_by_real_detector(
        self, real_client: TestClient
    ) -> None:
        person_id = _create_person(real_client)
        response = real_client.post(
            f"/v1/persons/{person_id}/enroll",
            files=[
                _file("face.png", png_bytes(solid_image())) for _ in range(3)
            ],
        )
        assert response.status_code == 400
        assert {entry["reason"] for entry in response.json()["details"]["reasons"]} == {
            "no_face"
        }
