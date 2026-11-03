#!/usr/bin/env python3
"""Download the ONNX models and verify their SHA-256 digests.

The canonical digests live in ``scripts/model_checksums.json``. Models are
written to ``models/`` (gitignored) together with a ``checksums.txt`` for
offline verification. Nothing here is imported by the service, so the script
only needs the Python standard library.

All four entries (YuNet, SFace, MiniFASNetV2, MiniFASNetV1SE) are required.
Every manifest entry must have a non-empty ``sha256``; a missing digest is a
configuration error and aborts the run. Downloads are streamed to a ``.part``
file, hashed and size-checked there, and only renamed into place after both
checks pass. When an entry declares ``bytes`` the downloader refuses a response
whose declared or observed size exceeds that value.

``optional: true`` remains supported for future non-critical artifacts: an
optional entry that cannot be fetched (network error, timeout, 404) is skipped
with a warning instead of failing the run. Checksum or size mismatches are
integrity failures, not transport failures, so they are always fatal even for
optional entries. None of the current entries use ``optional``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "scripts" / "model_checksums.json"
USER_AGENT = "facecheck-model-downloader/0.1 (+https://github.com/imam-db/facecheck)"
CHUNK_SIZE = 1024 * 1024


class DownloadError(RuntimeError):
    """A model could not be fetched from any of its manifest URLs."""


class IntegrityError(RuntimeError):
    """A model failed checksum or size verification; always fatal."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: Path = MANIFEST_PATH) -> dict[str, dict[str, object]]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def save_manifest(manifest: dict[str, dict[str, object]], path: Path = MANIFEST_PATH) -> None:
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def required_sha256(entry: dict[str, object], filename: str) -> str:
    expected = str(entry.get("sha256") or "").strip().lower()
    if not expected:
        raise SystemExit(f"{filename}: manifest is missing sha256; refusing to continue")
    return expected


def expected_size(entry: dict[str, object], filename: str) -> int | None:
    value = entry.get("bytes")
    if value is None:
        return None
    try:
        size = int(value)
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"{filename}: invalid bytes value {value!r}") from exc
    if size <= 0:
        raise SystemExit(f"{filename}: invalid bytes value {value!r}")
    return size


def check_file(
    path: Path, filename: str, expected_sha: str, expected_bytes: int | None
) -> str:
    actual = sha256_file(path)
    if actual != expected_sha:
        raise IntegrityError(
            f"{filename}: checksum mismatch\n"
            f"  expected {expected_sha}\n  actual   {actual}\n"
            "Re-run with --force to download again."
        )
    if expected_bytes is not None:
        actual_bytes = path.stat().st_size
        if actual_bytes != expected_bytes:
            raise IntegrityError(
                f"{filename}: size mismatch\n"
                f"  expected {expected_bytes}\n  actual   {actual_bytes}\n"
                "Re-run with --force to download again."
            )
    return actual


def download(
    url: str,
    destination: Path,
    *,
    expected_sha: str,
    expected_bytes: int | None,
) -> str:
    """Stream to ``.part``, verify hash and size, then rename. Returns the digest."""

    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    partial = destination.with_suffix(destination.suffix + ".part")
    digest = hashlib.sha256()
    size = 0
    try:
        with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as out:
            content_length = response.headers.get("Content-Length")
            if expected_bytes is not None and content_length is not None:
                try:
                    declared = int(content_length)
                except ValueError as exc:
                    raise DownloadError(
                        f"{destination.name}: invalid Content-Length {content_length!r}"
                    ) from exc
                if declared > expected_bytes:
                    raise IntegrityError(
                        f"{destination.name}: refusing oversized download "
                        f"({declared} bytes > expected {expected_bytes})"
                    )
            while True:
                chunk = response.read(CHUNK_SIZE)
                if not chunk:
                    break
                size += len(chunk)
                if expected_bytes is not None and size > expected_bytes:
                    raise IntegrityError(
                        f"{destination.name}: refusing oversized download "
                        f"(> expected {expected_bytes} bytes)"
                    )
                digest.update(chunk)
                out.write(chunk)

        actual = digest.hexdigest()
        if not actual:
            raise IntegrityError(f"{destination.name}: empty SHA-256 after download")
        if actual != expected_sha:
            raise IntegrityError(
                f"{destination.name}: checksum mismatch before rename\n"
                f"  expected {expected_sha}\n  actual   {actual}"
            )
        if expected_bytes is not None and size != expected_bytes:
            raise IntegrityError(
                f"{destination.name}: size mismatch before rename\n"
                f"  expected {expected_bytes}\n  actual   {size}"
            )
        partial.replace(destination)
        return actual
    except BaseException:
        partial.unlink(missing_ok=True)
        raise


def fetch_model(
    filename: str,
    entry: dict[str, object],
    models_dir: Path,
    *,
    force: bool,
) -> tuple[Path, str, bool]:
    """Return (path, sha256, downloaded)."""

    destination = models_dir / filename
    expected_sha = required_sha256(entry, filename)
    expected_bytes = expected_size(entry, filename)

    if destination.is_file() and not force:
        actual = check_file(destination, filename, expected_sha, expected_bytes)
        return destination, actual, False

    urls = [str(url) for url in entry.get("urls", [])]
    if not urls:
        raise DownloadError(f"{filename}: no download URL in manifest")

    last_error: Exception | None = None
    for url in urls:
        print(f"downloading {filename}\n  from {url}")
        try:
            actual = download(
                url,
                destination,
                expected_sha=expected_sha,
                expected_bytes=expected_bytes,
            )
            break
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = exc
            print(f"  failed: {exc}", file=sys.stderr)
    else:
        raise DownloadError(f"{filename}: all download URLs failed ({last_error})")

    return destination, actual, True


def write_checksums(models_dir: Path, checksums: dict[str, str]) -> None:
    lines = ["# SHA-256 checksums for facecheck model files", *(
        f"{digest}  {filename}" for filename, digest in sorted(checksums.items())
    )]
    (models_dir / "checksums.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=REPO_ROOT / "models",
        help="destination directory (default: ./models)",
    )
    parser.add_argument("--force", action="store_true", help="download even if files exist")
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="check existing files against the manifest without downloading",
    )
    parser.add_argument(
        "--update-checksums",
        action="store_true",
        help="write newly computed digests back to scripts/model_checksums.json",
    )
    args = parser.parse_args(argv)

    models_dir: Path = args.models_dir
    models_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest()

    for filename, entry in manifest.items():
        required_sha256(entry, filename)
        expected_size(entry, filename)

    checksums: dict[str, str] = {}
    updated = False
    skipped: list[str] = []
    for filename, entry in manifest.items():
        optional = bool(entry.get("optional"))
        destination = models_dir / filename
        expected_sha = required_sha256(entry, filename)
        expected_bytes = expected_size(entry, filename)
        try:
            if args.verify_only:
                if not destination.is_file():
                    if optional:
                        print(f"skip optional {filename} (not present)", file=sys.stderr)
                        skipped.append(filename)
                        continue
                    raise SystemExit(f"{filename}: missing (expected in {models_dir})")
                actual = check_file(destination, filename, expected_sha, expected_bytes)
                print(f"ok {filename} {actual}")
                checksums[filename] = actual
                continue

            try:
                destination, actual, downloaded = fetch_model(
                    filename, entry, models_dir, force=args.force
                )
            except DownloadError as exc:
                if optional:
                    print(f"skip optional {filename}: {exc}", file=sys.stderr)
                    skipped.append(filename)
                    continue
                raise SystemExit(str(exc)) from exc
        except IntegrityError as exc:
            # Checksum or size failures are never skipped, even for optional entries.
            raise SystemExit(str(exc)) from exc

        entry["sha256"] = actual
        updated = updated or downloaded
        checksums[filename] = actual
        status = "downloaded" if downloaded else "cached"
        print(f"{status} {destination} ({actual})")

    write_checksums(models_dir, checksums)
    if args.update_checksums and updated:
        save_manifest(manifest)
        print(f"updated {MANIFEST_PATH}")
    if skipped:
        print(f"skipped {len(skipped)} optional model(s): {', '.join(skipped)}")
    print(f"models ready in {models_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
