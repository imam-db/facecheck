"""AES-GCM helpers for out-of-repo PAD capture directories.

PAD evaluation images are biometric data and must never be committed. The
capture protocol stores them outside the repository and encrypts them at rest
with a 32-byte key supplied through ``FACECHECK_PAD_CAPTURE_KEY``.

The on-disk envelope is::

    b"FCPAD1" + 12-byte nonce + AES-GCM ciphertext

The associated data is the file's path relative to the capture root, so an
envelope cannot be silently moved to a different capture name. The key is
never logged, printed, or written to a report. This is intentionally separate
from the template-encryption AAD in :mod:`facecheck.storage`.
"""

from __future__ import annotations

import base64
import os
from pathlib import Path

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

PAD_CAPTURE_KEY_ENV = "FACECHECK_PAD_CAPTURE_KEY"
ENVELOPE_PREFIX = b"FCPAD1"
NONCE_BYTES = 12
KEY_BYTES = 32
ENCRYPTED_SUFFIX = ".aesgcm"

__all__ = [
    "ENCRYPTED_SUFFIX",
    "ENVELOPE_PREFIX",
    "KEY_BYTES",
    "NONCE_BYTES",
    "PAD_CAPTURE_KEY_ENV",
    "PadCaptureCryptoError",
    "PadCaptureKeyError",
    "decrypt_bytes",
    "decrypt_file",
    "decrypt_tree",
    "encrypt_bytes",
    "encrypt_file",
    "encrypt_tree",
    "parse_capture_key",
]


class PadCaptureKeyError(ValueError):
    """Raised when the capture key is missing or malformed."""


class PadCaptureCryptoError(RuntimeError):
    """Raised when a capture envelope cannot be authenticated or parsed."""


def parse_capture_key(value: str | bytes | None) -> bytes:
    """Parse a 32-byte AES key from urlsafe base64, standard base64, or hex.

    This mirrors :func:`facecheck.storage.parse_template_key` but names the
    PAD capture variable in its errors and never includes the value itself.
    """

    if value is None or value == "":
        raise PadCaptureKeyError(
            f"{PAD_CAPTURE_KEY_ENV} is required: provide a 32-byte key as "
            "urlsafe base64 or 64 hex characters"
        )

    if isinstance(value, bytes):
        if len(value) == KEY_BYTES:
            return value
        try:
            value = value.decode("ascii")
        except UnicodeDecodeError as exc:
            raise PadCaptureKeyError(
                f"{PAD_CAPTURE_KEY_ENV} must be ASCII text or 32 raw bytes"
            ) from exc

    text = value.strip()
    if len(text) == KEY_BYTES * 2:
        try:
            decoded = bytes.fromhex(text)
        except ValueError:
            decoded = b""
        if len(decoded) == KEY_BYTES:
            return decoded

    padded = text + "=" * (-len(text) % 4)
    for decoder in (base64.urlsafe_b64decode, base64.b64decode):
        try:
            decoded = decoder(padded.encode("ascii"))
        except (ValueError, UnicodeEncodeError):
            continue
        if len(decoded) == KEY_BYTES:
            return decoded

    raise PadCaptureKeyError(
        f"{PAD_CAPTURE_KEY_ENV} must decode to exactly 32 bytes "
        "(urlsafe base64 or 64 hex characters)"
    )


def _check_key(key: bytes) -> None:
    if len(key) != KEY_BYTES:
        raise PadCaptureKeyError(
            f"capture key must be {KEY_BYTES} bytes, got {len(key)}"
        )


def encrypt_bytes(key: bytes, plaintext: bytes, *, aad: bytes = b"") -> bytes:
    """Encrypt bytes into a capture envelope. The key is never logged."""

    _check_key(key)
    nonce = os.urandom(NONCE_BYTES)
    ciphertext = AESGCM(key).encrypt(nonce, plaintext, aad)
    return ENVELOPE_PREFIX + nonce + ciphertext


def decrypt_bytes(key: bytes, envelope: bytes, *, aad: bytes = b"") -> bytes:
    """Authenticate and decrypt a capture envelope."""

    _check_key(key)
    header = len(ENVELOPE_PREFIX) + NONCE_BYTES
    if not envelope.startswith(ENVELOPE_PREFIX) or len(envelope) <= header:
        raise PadCaptureCryptoError("not a PAD capture envelope")
    nonce = envelope[len(ENVELOPE_PREFIX) : header]
    ciphertext = envelope[header:]
    try:
        return AESGCM(key).decrypt(nonce, ciphertext, aad)
    except InvalidTag as exc:
        raise PadCaptureCryptoError(
            "capture envelope failed authentication: wrong key, wrong path, "
            "or modified ciphertext"
        ) from exc


def encrypt_file(src: str | Path, dst: str | Path, key: bytes, *, aad: bytes = b"") -> None:
    src = Path(src)
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(encrypt_bytes(key, src.read_bytes(), aad=aad))


def decrypt_file(src: str | Path, dst: str | Path, key: bytes, *, aad: bytes = b"") -> None:
    src = Path(src)
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(decrypt_bytes(key, src.read_bytes(), aad=aad))


def _relative_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file())


def _prepare_destination(dst: Path) -> None:
    if dst.exists() and (not dst.is_dir() or any(dst.iterdir())):
        raise PadCaptureCryptoError(
            f"destination {dst} already exists and is not an empty directory"
        )
    dst.mkdir(parents=True, exist_ok=True)


def encrypt_tree(src: str | Path, dst: str | Path, key: bytes) -> int:
    """Encrypt every file under ``src`` into ``dst`` with a ``.aesgcm`` suffix.

    Relative directory structure is preserved. The AAD for each file is its
    POSIX path relative to ``src``. The plaintext tree is never modified and
    the destination must be new or empty.
    """

    _check_key(key)
    src = Path(src)
    dst = Path(dst)
    if not src.is_dir():
        raise PadCaptureCryptoError(f"source directory {src} does not exist")
    _prepare_destination(dst)

    count = 0
    for path in _relative_files(src):
        relative = path.relative_to(src)
        target = dst / relative.parent / (relative.name + ENCRYPTED_SUFFIX)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(
            encrypt_bytes(key, path.read_bytes(), aad=relative.as_posix().encode())
        )
        count += 1
    return count


def decrypt_tree(src: str | Path, dst: str | Path, key: bytes) -> int:
    """Decrypt a tree produced by :func:`encrypt_tree` into ``dst``."""

    _check_key(key)
    src = Path(src)
    dst = Path(dst)
    if not src.is_dir():
        raise PadCaptureCryptoError(f"source directory {src} does not exist")
    _prepare_destination(dst)

    count = 0
    for path in _relative_files(src):
        if not path.name.endswith(ENCRYPTED_SUFFIX):
            continue
        relative = path.relative_to(src)
        plain_relative = relative.with_name(relative.name[: -len(ENCRYPTED_SUFFIX)])
        target = dst / plain_relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(
            decrypt_bytes(
                key,
                path.read_bytes(),
                aad=plain_relative.as_posix().encode(),
            )
        )
        count += 1
    return count
