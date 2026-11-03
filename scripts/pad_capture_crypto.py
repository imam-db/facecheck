#!/usr/bin/env python3
"""Encrypt or decrypt an out-of-repo PAD capture directory with AES-GCM.

The key comes from ``FACECHECK_PAD_CAPTURE_KEY`` and is never printed. The
encrypted tree keeps relative paths and appends ``.aesgcm`` to every file.

Usage::

    export FACECHECK_PAD_CAPTURE_KEY="$(
      python -c 'import base64,os; print(base64.urlsafe_b64encode(os.urandom(32)).decode())'
    )"
    python scripts/pad_capture_crypto.py encrypt \
      --src /path/to/pad-capture --dst /path/to/pad-capture.encrypted
    python scripts/pad_capture_crypto.py decrypt \
      --src /path/to/pad-capture.encrypted --dst /tmp/pad-capture-plain

Exit codes: 0 on success, 2 on a missing key, a missing directory, or an
authentication failure. Never commit either tree to the repository.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.pad_crypto import (  # noqa: E402
    PAD_CAPTURE_KEY_ENV,
    PadCaptureCryptoError,
    PadCaptureKeyError,
    decrypt_tree,
    encrypt_tree,
    parse_capture_key,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    subparsers = parser.add_subparsers(dest="command", required=True)

    encrypt_parser = subparsers.add_parser("encrypt", help="encrypt a capture tree")
    encrypt_parser.add_argument("--src", type=Path, required=True)
    encrypt_parser.add_argument("--dst", type=Path, required=True)

    decrypt_parser = subparsers.add_parser("decrypt", help="decrypt a capture tree")
    decrypt_parser.add_argument("--src", type=Path, required=True)
    decrypt_parser.add_argument("--dst", type=Path, required=True)

    args = parser.parse_args(argv)

    try:
        key = parse_capture_key(os.environ.get(PAD_CAPTURE_KEY_ENV))
        if args.command == "encrypt":
            count = encrypt_tree(args.src, args.dst, key)
            print(f"encrypted {count} file(s) into {args.dst}")
        else:
            count = decrypt_tree(args.src, args.dst, key)
            print(f"decrypted {count} file(s) into {args.dst}")
    except (PadCaptureKeyError, PadCaptureCryptoError) as exc:
        print(str(exc), file=sys.stderr)
        return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())
