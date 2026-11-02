#!/usr/bin/env python3
"""Fetch the LFW test pairs and images used by ``eval/run_lfw.py``.

The canonical LFW download (figshare / UMass) was returning 403/500 while this
project was built, so the script uses two mirrors and verifies every download:

- pairs: the official ``pairsDevTest.txt`` mirrored on GitHub; its SHA-256
  matches the checksum published in scikit-learn's LFW loader.
- images: the archive.org ``lfw-dataset`` mirror, which bundles LFW
  deep-funneled images. Deep-funneled is an aligned variant of LFW; images are
  used only for evaluation, never committed.

Output layout::

    data/lfw_home/pairsDevTest.txt
    data/lfw_home/lfw_deepfunneled/<name>/<name>_0001.jpg

Usage::

    python scripts/fetch_lfw.py
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
USER_AGENT = "facecheck-lfw-fetcher/0.1 (+https://github.com/imam-db/facecheck)"
EXPECTED_IMAGES = 13233


@dataclass(frozen=True)
class RemoteFile:
    filename: str
    url: str
    sha256: str


PAIRS = RemoteFile(
    filename="pairsDevTest.txt",
    url="https://raw.githubusercontent.com/AruniRC/resnet-face-pytorch/master/lfw/data/pairsDevTest.txt",
    sha256="7cb06600ea8b2814ac26e946201cdb304296262aad67d046a16a7ec85d0ff87c",
)

IMAGES_ARCHIVE = RemoteFile(
    filename="lfw-dataset.zip",
    url="https://archive.org/download/lfw-dataset/lfw-dataset.zip",
    sha256="62e73d1679ada6ab098df099fca1590f0d4393fa67462a0113377ba8abf4b1cb",
)

NESTED_ARCHIVE = "lfw-deepfunneled.zip"
NESTED_SHA256 = "72fca59f3cf5e20e5dff4292e2cb2d24808c9d49e2eb0259c9e00b2cbbe864aa"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(remote: RemoteFile, destination: Path, *, force: bool) -> Path:
    if destination.is_file() and not force:
        if sha256_file(destination) == remote.sha256:
            print(f"cached {destination}")
            return destination
        destination.unlink()

    partial = destination.with_suffix(destination.suffix + ".part")
    print(f"downloading {remote.url}")
    request = urllib.request.Request(remote.url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=300) as response, partial.open("wb") as out:
        shutil.copyfileobj(response, out, 1024 * 1024)
    actual = sha256_file(partial)
    if actual != remote.sha256:
        partial.unlink(missing_ok=True)
        raise SystemExit(
            f"{remote.filename}: checksum mismatch\n"
            f"  expected {remote.sha256}\n  actual   {actual}"
        )
    partial.replace(destination)
    print(f"verified {destination} ({actual})")
    return destination


def _safe_member(name: str) -> str | None:
    if name.startswith("__MACOSX/") or not name.startswith("lfw-deepfunneled/"):
        return None
    relative = name[len("lfw-deepfunneled/") :]
    if not relative or ".." in Path(relative).parts:
        return None
    return relative


def extract_images(archive: Path, lfw_home: Path) -> Path:
    target = lfw_home / "lfw_deepfunneled"
    staging = lfw_home / ".lfw_deepfunneled.tmp"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    with zipfile.ZipFile(archive) as outer:
        nested_bytes = outer.read(NESTED_ARCHIVE)
    nested_path = archive.parent / NESTED_ARCHIVE
    nested_path.write_bytes(nested_bytes)
    actual = sha256_file(nested_path)
    if actual != NESTED_SHA256:
        raise SystemExit(
            f"{NESTED_ARCHIVE}: checksum mismatch\n"
            f"  expected {NESTED_SHA256}\n  actual   {actual}"
        )

    written = 0
    with zipfile.ZipFile(nested_path) as nested:
        for info in nested.infolist():
            relative = _safe_member(info.filename)
            if relative is None:
                continue
            destination = staging / relative
            if info.is_dir():
                destination.mkdir(parents=True, exist_ok=True)
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            with nested.open(info) as source, destination.open("wb") as out:
                shutil.copyfileobj(source, out)
            written += 1
    nested_path.unlink(missing_ok=True)

    if written != EXPECTED_IMAGES:
        shutil.rmtree(staging)
        raise SystemExit(f"expected {EXPECTED_IMAGES} images, extracted {written}")

    if target.exists():
        shutil.rmtree(target)
    staging.rename(target)
    print(f"extracted {written} images to {target}")
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data")
    parser.add_argument("--force", action="store_true", help="re-download and re-extract")
    args = parser.parse_args(argv)

    lfw_home = args.data_dir / "lfw_home"
    lfw_home.mkdir(parents=True, exist_ok=True)

    pairs_path = download(PAIRS, lfw_home / PAIRS.filename, force=args.force)

    images_dir = lfw_home / "lfw_deepfunneled"
    image_count = len(list(images_dir.glob("*/*.jpg"))) if images_dir.is_dir() else 0
    if image_count == EXPECTED_IMAGES and not args.force:
        print(f"cached {images_dir} ({image_count} images)")
    else:
        archive = download(
            IMAGES_ARCHIVE, args.data_dir / IMAGES_ARCHIVE.filename, force=args.force
        )
        extract_images(archive, lfw_home)
        archive.unlink(missing_ok=True)

    pair_lines = [line for line in pairs_path.read_text().splitlines() if line.strip()]
    declared = int(pair_lines[0])
    if len(pair_lines) != 2 * declared + 1:
        raise SystemExit(f"{pairs_path}: expected {2 * declared + 1} lines, got {len(pair_lines)}")
    print(f"ready: {declared} matched + {declared} mismatched test pairs in {lfw_home}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
