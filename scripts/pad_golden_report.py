#!/usr/bin/env python3
"""Print runtime MiniFASNet scores for the out-of-repo PAD golden set.

Usage:
    FACECHECK_PAD_GOLDEN_DIR=/path/to/golden python scripts/pad_golden_report.py
    python scripts/pad_golden_report.py /path/to/golden

The golden directory holds ``manifest.json`` (schema:
``docs/pad_golden_manifest.md``) and images that are never committed. For each
case this loads the image, runs the runtime ``MiniFASNetLivenessDetector``, and
prints one line with the case id, label, manifest ``expected_real_prob``, the
runtime score, and the absolute delta.

Exits nonzero when the environment variable or argument is missing, the
directory or manifest is missing, the ONNX models are not present, or an image
cannot be read.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import cv2

from facecheck.config import Settings
from facecheck.liveness import (
    LivenessModelNotFoundError,
    MiniFASNetLivenessDetector,
)

GOLDEN_ENV = "FACECHECK_PAD_GOLDEN_DIR"


def _load_cases(manifest_path: Path) -> list[dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return manifest.get("cases") or []


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "golden_dir",
        nargs="?",
        default=os.environ.get(GOLDEN_ENV),
        help=f"golden directory (default: ${GOLDEN_ENV})",
    )
    args = parser.parse_args(argv)

    if not args.golden_dir:
        print(f"{GOLDEN_ENV} is not set; pass a directory or export it", file=sys.stderr)
        return 2

    golden_dir = Path(args.golden_dir)
    manifest_path = golden_dir / "manifest.json"
    if not manifest_path.is_file():
        print(f"missing golden manifest {manifest_path}", file=sys.stderr)
        return 2

    cases = _load_cases(manifest_path)
    if not cases:
        print(f"{manifest_path} has no cases", file=sys.stderr)
        return 2

    try:
        detector = MiniFASNetLivenessDetector.from_settings(Settings(_env_file=None))
    except LivenessModelNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    print(f"{'case':<16} {'label':<6} {'expected':>10} {'runtime':>10} {'delta':>10}")
    for case in cases:
        image_path = golden_dir / case["image"]
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            print(f"could not read {image_path}", file=sys.stderr)
            return 2

        expected = float(case["expected_real_prob"])
        bbox = [int(value) for value in case["bbox_xywh"]]
        result = detector.score(image, bbox)
        delta = abs(result.score - expected)
        print(
            f"{case['id']:<16} {case['label']:<6} "
            f"{expected:>10.6f} {result.score:>10.6f} {delta:>10.6f}"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
