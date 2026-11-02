"""API tests with a deterministic fake embedder.

These run everywhere. Tests that need the real ONNX models live in
``ModelBackedApiTests`` and skip when the model files are absent.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from facecheck.api import create_app
from facecheck.service import FaceVerificationService
from tests.conftest import (
    FakeEmbedder,
    image_with_marker,
    make_settings,
    models_available,
    png_bytes,
    solid_image,
)

ENROLL_COLORS = (180, 120, 90)


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
    response = client.post("/v1/persons", json=payload or None)
    assert response.status_code == 201, response.text
    return response.json()["id"]


def test_meta_reports_model_and_threshold(client: TestClient) -> None:
    response = client.get("/v1/meta")
    assert response.status_code == 200
    body = response.json()
    assert body == {
        "model_version": "fake_v1",
        "threshold": 0.363,
        "embedding_dim": 3,
        "min_face_size": 80,
    }


def test_healthz(client: TestClient) -> None:
    assert client.get("/healthz").json() == {"status": "ok"}


def test_create_person_returns_uuid(client: TestClient) -> None:
    response = client.post("/v1/persons", json={"external_id": "emp-1", "display_name": "Siti"})
    assert response.status_code == 201
    body = response.json()
    assert body["external_id"] == "emp-1"
    assert body["display_name"] == "Siti"
    assert len(body["id"]) == 36


def test_create_person_without_body(client: TestClient) -> None:
    response = client.post("/v1/persons")
    assert response.status_code == 201
    assert response.json()["external_id"] is None


def test_duplicate_external_id_is_conflict(client: TestClient) -> None:
    _create_person(client, external_id="emp-42")
    response = client.post("/v1/persons", json={"external_id": "emp-42"})
    assert response.status_code == 409
    assert "emp-42" in response.json()["detail"]


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
    assert "between 3 and 5" in too_few.json()["detail"]

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
    assert body["reasons"] == [
        {"index": 1, "reason": "no_face", "message": "no face detected"}
    ]


def test_enroll_unknown_person_is_404(client: TestClient) -> None:
    response = client.post(
        "/v1/persons/00000000-0000-0000-0000-000000000000/enroll",
        files=_enroll_images(3),
    )
    assert response.status_code == 404


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
    assert response.json()["reason"] == "invalid_image"


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
    assert "no enrolled templates" in response.json()["detail"]


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
    assert "re-enroll" in response.json()["detail"]


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
    assert response.json()["reason"] == "no_face"


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
    with TestClient(app) as client:
        person_id = _create_person(client)
        response = client.post(
            f"/v1/persons/{person_id}/verify",
            files={"file": ("probe.png", png_bytes(image_with_marker()))},
        )
    assert response.status_code == 400
    assert "upload limit" in response.json()["detail"]


@pytest.mark.model
@pytest.mark.skipif(not models_available(), reason="ONNX models not downloaded")
class ModelBackedApiTests:
    @pytest.fixture
    def real_client(self, tmp_path: Path):
        settings = make_settings(tmp_path, embedding_dim=128)
        app = create_app(settings)
        with TestClient(app) as client:
            yield client

    def test_meta_reports_real_model_version(self, real_client: TestClient) -> None:
        body = real_client.get("/v1/meta").json()
        assert body["model_version"] == (
            "face_detection_yunet_2023mar+face_recognition_sface_2021dec"
        )
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
        assert {entry["reason"] for entry in response.json()["reasons"]} == {"no_face"}
