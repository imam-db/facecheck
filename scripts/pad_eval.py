#!/usr/bin/env python3
"""Score a consented PAD capture directory and write aggregate reports.

See ``eval/pad_eval.py`` for the implementation and
``docs/pad-capture-protocol.md`` for the manifest schema and capture rules.

Usage::

    FACECHECK_PAD_CAPTURE_DIR=/path/to/captures \
    python scripts/pad_eval.py --manifest /path/to/captures/manifest.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.pad_eval import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
