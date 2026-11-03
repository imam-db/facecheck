"""Unit tests for the out-of-repo PAD capture encryption helpers.

These use random bytes in ``tmp_path`` and need no images or models.
"""

from __future__ import annotations

import base64
import os
from pathlib import Path

import pytest
from eval.pad_crypto import (
    ENCRYPTED_SUFFIX,
    ENVELOPE_PREFIX,
    PAD_CAPTURE_KEY_ENV,
    PadCaptureCryptoError,
    PadCaptureKeyError,
    decrypt_bytes,
    decrypt_file,
    decrypt_tree,
    encrypt_bytes,
    encrypt_file,
    encrypt_tree,
    parse_capture_key,
)

KEY = os.urandom(32)
OTHER_KEY = os.urandom(32)


def test_parse_capture_key_accepts_hex_and_base64() -> None:
    raw = os.urandom(32)
    assert parse_capture_key(raw) == raw
    assert parse_capture_key(raw.hex()) == raw
    assert parse_capture_key(base64.urlsafe_b64encode(raw).decode()) == raw
    assert parse_capture_key(base64.b64encode(raw).decode()) == raw


def test_parse_capture_key_rejects_missing_and_wrong_length() -> None:
    with pytest.raises(PadCaptureKeyError, match=PAD_CAPTURE_KEY_ENV):
        parse_capture_key(None)
    with pytest.raises(PadCaptureKeyError, match=PAD_CAPTURE_KEY_ENV):
        parse_capture_key("")
    with pytest.raises(PadCaptureKeyError, match="32 bytes"):
        parse_capture_key(b"too-short")
    with pytest.raises(PadCaptureKeyError, match="32 bytes"):
        parse_capture_key("not-a-valid-key")


def test_bytes_round_trip_uses_an_envelope() -> None:
    payload = os.urandom(4096)
    envelope = encrypt_bytes(KEY, payload)
    assert envelope.startswith(ENVELOPE_PREFIX)
    assert payload not in envelope
    assert decrypt_bytes(KEY, envelope) == payload


def test_decrypt_rejects_wrong_key_and_tampering() -> None:
    envelope = encrypt_bytes(KEY, b"consented capture bytes")
    with pytest.raises(PadCaptureCryptoError, match="authentication"):
        decrypt_bytes(OTHER_KEY, envelope)

    tampered = bytearray(envelope)
    tampered[-1] ^= 0x01
    with pytest.raises(PadCaptureCryptoError, match="authentication"):
        decrypt_bytes(KEY, bytes(tampered))

    with pytest.raises(PadCaptureCryptoError, match="envelope"):
        decrypt_bytes(KEY, b"not-an-envelope")


def test_associated_data_is_required() -> None:
    envelope = encrypt_bytes(KEY, b"payload", aad=b"subj_01/image.jpg")
    assert decrypt_bytes(KEY, envelope, aad=b"subj_01/image.jpg") == b"payload"
    with pytest.raises(PadCaptureCryptoError):
        decrypt_bytes(KEY, envelope, aad=b"subj_02/image.jpg")


def test_encrypt_and_decrypt_file(tmp_path: Path) -> None:
    src = tmp_path / "capture.bin"
    envelope = tmp_path / "capture.bin.aesgcm"
    restored = tmp_path / "restored.bin"
    payload = os.urandom(1024)
    src.write_bytes(payload)

    encrypt_file(src, envelope, KEY)
    assert envelope.read_bytes().startswith(ENVELOPE_PREFIX)
    decrypt_file(envelope, restored, KEY)
    assert restored.read_bytes() == payload


def test_tree_round_trip_preserves_names(tmp_path: Path) -> None:
    src = tmp_path / "captures"
    (src / "subj_01").mkdir(parents=True)
    first = os.urandom(256)
    second = b"synthetic notes"
    (src / "subj_01" / "presentation.bin").write_bytes(first)
    (src / "notes.txt").write_bytes(second)

    encrypted = tmp_path / "captures.encrypted"
    assert encrypt_tree(src, encrypted, KEY) == 2
    assert not (encrypted / "subj_01" / "presentation.bin").exists()
    assert (encrypted / "subj_01" / f"presentation.bin{ENCRYPTED_SUFFIX}").is_file()
    assert second not in (encrypted / f"notes.txt{ENCRYPTED_SUFFIX}").read_bytes()

    restored = tmp_path / "restored"
    assert decrypt_tree(encrypted, restored, KEY) == 2
    assert (restored / "subj_01" / "presentation.bin").read_bytes() == first
    assert (restored / "notes.txt").read_bytes() == second


def test_tree_aad_binds_ciphertext_to_the_relative_path(tmp_path: Path) -> None:
    src = tmp_path / "captures"
    src.mkdir()
    (src / "original.bin").write_bytes(os.urandom(64))
    encrypted = tmp_path / "encrypted"
    encrypt_tree(src, encrypted, KEY)

    (encrypted / f"original.bin{ENCRYPTED_SUFFIX}").rename(
        encrypted / f"renamed.bin{ENCRYPTED_SUFFIX}"
    )
    with pytest.raises(PadCaptureCryptoError, match="authentication"):
        decrypt_tree(encrypted, tmp_path / "restored", KEY)


def test_tree_refuses_a_non_empty_destination(tmp_path: Path) -> None:
    src = tmp_path / "captures"
    src.mkdir()
    (src / "image.bin").write_bytes(os.urandom(32))
    destination = tmp_path / "destination"
    destination.mkdir()
    (destination / "existing.txt").write_text("keep me", encoding="utf-8")

    with pytest.raises(PadCaptureCryptoError, match="not an empty directory"):
        encrypt_tree(src, destination, KEY)
    assert (destination / "existing.txt").read_text(encoding="utf-8") == "keep me"


def test_tree_requires_an_existing_source(tmp_path: Path) -> None:
    with pytest.raises(PadCaptureCryptoError, match="does not exist"):
        encrypt_tree(tmp_path / "missing", tmp_path / "out", KEY)


def test_key_is_not_embedded_in_ciphertext() -> None:
    key = bytes(range(32))
    envelope = encrypt_bytes(key, b"plaintext")
    assert key not in envelope
    assert key.hex().encode() not in envelope
