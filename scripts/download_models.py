#!/usr/bin/env python3
"""Download the ONNX models and verify their SHA-256 digests.

The canonical digests live in ``scripts/model_checksums.json``. Models are
written to ``models/`` (gitignored) together with a ``checksums.txt`` for
offline verification. Nothing here is imported by the service, so the script
only needs the Python standard library.

The YuNet and SFace entries are required. MiniFASNet PAD entries are marked
``"optional": true`` because their GitHub release assets may not be published
yet; an unavailable optional asset is skipped with a warning instead of failing
the whole download. Existing local files are always checksum-verified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
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


def download(url: str, destination: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    partial = destination.with_suffix(destination.suffix + ".part")
    with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as out:
        shutil.copyfileobj(response, out, CHUNK_SIZE)
    partial.replace(destination)


def fetch_model(
    filename: str,
    entry: dict[str, object],
    models_dir: Path,
    *,
    force: bool,
) -> tuple[Path, str, bool]:
    """Return (path, sha256, downloaded)."""

    destination = models_dir / filename
    expected = str(entry.get("sha256") or "")

    if destination.is_file() and not force:
        actual = sha256_file(destination)
        if expected and actual != expected:
            raise SystemExit(
                f"{filename}: checksum mismatch for existing file\n"
                f"  expected {expected}\n  actual   {actual}\n"
                "Re-run with --force to download again."
            )
        return destination, actual, False

    urls = [str(url) for url in entry.get("urls", [])]
    if not urls:
        raise DownloadError(f"{filename}: no download URL in manifest")

    last_error: Exception | None = None
    for url in urls:
        print(f"downloading {filename}\n  from {url}")
        try:
            download(url, destination)
            break
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = exc
            print(f"  failed: {exc}", file=sys.stderr)
    else:
        raise DownloadError(f"{filename}: all download URLs failed ({last_error})")

    actual = sha256_file(destination)
    if expected and actual != expected:
        destination.unlink(missing_ok=True)
        raise SystemExit(
            f"{filename}: checksum mismatch after download\n"
            f"  expected {expected}\n  actual   {actual}"
        )
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

    checksums: dict[str, str] = {}
    updated = False
    skipped: list[str] = []
    for filename, entry in manifest.items():
        optional = bool(entry.get("optional"))
        destination = models_dir / filename
        if args.verify_only:
            if not destination.is_file():
                if optional:
                    print(f"skip optional {filename} (not present)", file=sys.stderr)
                    skipped.append(filename)
                    continue
                raise SystemExit(f"{filename}: missing (expected in {models_dir})")
            actual = sha256_file(destination)
            expected = str(entry.get("sha256") or "")
            if expected and actual != expected:
                raise SystemExit(
                    f"{filename}: checksum mismatch\n  expected {expected}\n  actual   {actual}"
                )
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
                if entry.get("pending_release"):
                    print(f"  note: {entry['pending_release']}", file=sys.stderr)
                skipped.append(filename)
                continue
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
