"""Command line interface for facecheck."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from facecheck.config import Settings
from facecheck.embeddings import ModelNotFoundError, build_embedder, decode_image_bgr
from facecheck.quality import QualityError
from facecheck.similarity import cosine_similarity, decide


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="facecheck",
        description="Minimal 1:1 face verification for SFA attendance prototypes.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    compare = subparsers.add_parser("compare", help="compare two face photos")
    compare.add_argument("image_a", type=Path, help="path to the first image")
    compare.add_argument("image_b", type=Path, help="path to the second image")
    compare.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="cosine threshold (default: FACECHECK_MATCH_THRESHOLD or 0.363)",
    )
    compare.add_argument(
        "--models-dir",
        type=Path,
        default=None,
        help="directory containing the YuNet and SFace ONNX files",
    )
    return parser


def _load_image(path: Path):
    try:
        return decode_image_bgr(path.read_bytes())
    except OSError as exc:
        raise SystemExit(f"error: cannot read {path}: {exc}") from exc


def _compare(args: argparse.Namespace) -> int:
    settings = Settings()
    if args.models_dir is not None:
        settings = settings.model_copy(update={"models_dir": args.models_dir})
    if args.threshold is not None:
        settings = settings.model_copy(update={"match_threshold": args.threshold})

    try:
        embedder = build_embedder(settings)
    except ModelNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    try:
        first = embedder.detect_and_embed(_load_image(args.image_a))
        second = embedder.detect_and_embed(_load_image(args.image_b))
    except QualityError as exc:
        print(f"error: {exc} ({exc.reason})", file=sys.stderr)
        return 2

    score = cosine_similarity(first.embedding, second.embedding)
    decision = decide(score, settings.match_threshold)
    print(
        f"score={score:.4f} decision={decision} "
        f"threshold={settings.match_threshold:.4f} model={first.model_version}"
    )
    return 0 if decision == "match" else 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "compare":
        return _compare(args)
    parser.error(f"unknown command {args.command!r}")
    return 2
