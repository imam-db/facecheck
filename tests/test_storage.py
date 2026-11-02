"""Storage tests: template encryption, consent fields, and secure deletion."""

from __future__ import annotations

import base64
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from facecheck.storage import (
    ENVELOPE_PREFIX,
    SchemaMigrationError,
    Storage,
    TemplateDecryptError,
    embedding_to_bytes,
    parse_template_key,
)
from tests.conftest import TEST_TEMPLATE_KEY


def _storage(path: Path, key: str = TEST_TEMPLATE_KEY) -> Storage:
    return Storage(f"sqlite:///{path}", template_key=key)


def _embedding(dim: int = 128, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vector = rng.normal(size=dim).astype(np.float32)
    return vector / np.linalg.norm(vector)


def test_parse_template_key_accepts_base64_and_hex() -> None:
    raw = parse_template_key(TEST_TEMPLATE_KEY)
    assert len(raw) == 32
    assert parse_template_key(raw.hex()) == raw
    assert parse_template_key(base64.urlsafe_b64encode(raw)) == raw
    with pytest.raises(ValueError, match="32 bytes"):
        parse_template_key("not-a-key")
    with pytest.raises(ValueError, match="FACECHECK_TEMPLATE_KEY"):
        parse_template_key(None)


def test_templates_round_trip_and_are_encrypted(tmp_path: Path) -> None:
    db_path = tmp_path / "encrypted.db"
    storage = _storage(db_path)
    person = storage.create_person(
        external_id="emp-1",
        consent_version="2026-11",
        consent_at=datetime.now(UTC),
    )
    vector = _embedding()
    storage.replace_templates(person.id, [vector], model_version="fake_v1")

    templates = storage.list_templates(person.id)
    assert len(templates) == 1
    assert np.allclose(
        np.frombuffer(templates[0].embedding, dtype=np.float32), vector
    )
    assert templates[0].model_version == "fake_v1"

    with sqlite3.connect(db_path) as connection:
        blob = connection.execute("SELECT embedding FROM templates").fetchone()[0]
    assert blob.startswith(ENVELOPE_PREFIX)
    assert embedding_to_bytes(vector) not in blob
    storage.close()


def test_decryption_requires_the_same_key(tmp_path: Path) -> None:
    db_path = tmp_path / "wrong-key.db"
    storage = _storage(db_path)
    person = storage.create_person()
    storage.replace_templates(person.id, [_embedding()], model_version="fake_v1")
    storage.close()

    other_key = base64.urlsafe_b64encode(b"x" * 32).decode()
    other = _storage(db_path, key=other_key)
    with pytest.raises(TemplateDecryptError) as excinfo:
        other.list_templates(person.id)
    assert excinfo.value.details["reason"] == "invalid_key"
    assert excinfo.value.details["template_id"]
    assert "re-enroll" in str(excinfo.value).lower()
    other.close()


def test_legacy_plaintext_template_is_a_decrypt_error(tmp_path: Path) -> None:
    db_path = tmp_path / "legacy.db"
    storage = _storage(db_path)
    person = storage.create_person()
    vector = _embedding()
    storage.replace_templates(person.id, [vector], model_version="fake_v1")

    with sqlite3.connect(db_path) as connection:
        connection.execute("UPDATE templates SET embedding = ?", (embedding_to_bytes(vector),))
        connection.commit()

    with pytest.raises(TemplateDecryptError) as excinfo:
        storage.list_templates(person.id)
    assert excinfo.value.details["reason"] == "legacy_plaintext"
    storage.close()


def test_startup_rejects_old_schema_with_display_name(tmp_path: Path) -> None:
    db_path = tmp_path / "v0.db"
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            "CREATE TABLE persons ("
            "id TEXT PRIMARY KEY, external_id TEXT, display_name TEXT, "
            "consent_version TEXT, consent_at DATETIME, created_at DATETIME)"
        )
        connection.commit()

    with pytest.raises(SchemaMigrationError, match="display_name"):
        _storage(db_path)


def test_startup_rejects_schema_without_consent_columns(tmp_path: Path) -> None:
    db_path = tmp_path / "no-consent.db"
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            "CREATE TABLE persons ("
            "id TEXT PRIMARY KEY, external_id TEXT, created_at DATETIME)"
        )
        connection.commit()

    with pytest.raises(SchemaMigrationError, match="consent_version"):
        _storage(db_path)


def test_startup_accepts_current_schema(tmp_path: Path) -> None:
    storage = _storage(tmp_path / "current.db")
    assert storage.get_person("missing") is None
    storage.close()


def test_consent_fields_round_trip(tmp_path: Path) -> None:
    storage = _storage(tmp_path / "consent.db")
    when = datetime(2026, 11, 1, 9, 0, tzinfo=UTC)
    person = storage.create_person(
        external_id="emp-9", consent_version="v2", consent_at=when
    )
    loaded = storage.get_person(person.id)
    assert loaded is not None
    assert loaded.consent_version == "v2"
    assert loaded.consent_at.replace(tzinfo=UTC) == when
    storage.close()


def test_secure_delete_removes_ciphertext_bytes(tmp_path: Path) -> None:
    db_path = tmp_path / "secure-delete.db"
    storage = _storage(db_path)
    person = storage.create_person()
    storage.replace_templates(person.id, [_embedding()], model_version="fake_v1")

    with sqlite3.connect(db_path) as connection:
        blob = connection.execute("SELECT embedding FROM templates").fetchone()[0]

    assert storage.delete_person(person.id)
    storage.close()

    assert blob not in db_path.read_bytes()
