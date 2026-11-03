#!/usr/bin/env python3
"""Export Minivision MiniFASNet anti-spoofing checkpoints to fixed-shape ONNX.

The two upstream checkpoints shipped with Silent-Face-Anti-Spoofing are:

- ``2.7_80x80_MiniFASNetV2.pth``   -> ``MiniFASNetV2``,   scale 2.7
- ``4_0_0_80x80_MiniFASNetV1SE.pth`` -> ``MiniFASNetV1SE``, scale 4.0

Both consume an 80x80 face crop and produce **three logits**. In the upstream
label convention index ``1`` is the real / bona-fide class; indices ``0`` and
``2`` are presentation-attack classes.

CRITICAL PREPROCESSING NOTE
---------------------------
The upstream ``AntiSpoofPredict.predict`` pipeline calls ``ToTensor`` and then
the model directly. In upstream ``src/data_io/functional.py`` the classic
``div(255)`` was commented out (``return img.float()``), so the network was
trained and evaluated on **raw BGR floats in [0, 255]**, not [0, 1] and not
ImageNet-normalized. Do **not** divide by 255 and do **not** apply mean/std
normalization. Feed a float32 NCHW blob built from a BGR 80x80 crop with values
in [0, 255]. Getting this wrong silently produces garbage confidence scores.

The architecture is vendored in ``scripts/_minifasnet_arch.py`` so exporting
does not require importing code from the Silent-Face clone; only the ``.pth``
files are read from its ``resources/anti_spoof_models`` directory.

PyTorch, ONNX, and ONNX Runtime are export-only dependencies:

    python -m pip install -e ".[export]"
    python scripts/export_minifasnet_onnx.py \
        --silent-face-root /path/to/Silent-Face-Anti-Spoofing

The generated ``.onnx`` files are written under ``models/`` (gitignored) and
are published as GitHub release assets, not committed. This script also runs
PyTorch vs ONNX Runtime parity on seeded synthetic inputs, asserts
``max_abs_diff < 1e-4``, and writes the replayable reference fixture used by
``tests/test_minifasnet_parity.py``.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import sys
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
INPUT_SHAPE = (1, 3, 80, 80)
INPUT_HEIGHT = 80
INPUT_WIDTH = 80
INPUT_NAME = "input"
OUTPUT_NAME = "logits"
NUM_CLASSES = 3
REAL_CLASS_INDEX = 1
PARITY_TOLERANCE = 1e-4
PARITY_SEED = 20261103
PARITY_CASES = 3
CHUNK_SIZE = 1024 * 1024


@dataclass(frozen=True)
class ModelSpec:
    checkpoint: str
    onnx_name: str
    architecture: str
    scale: float


MODEL_SPECS = (
    ModelSpec(
        checkpoint="2.7_80x80_MiniFASNetV2.pth",
        onnx_name="2.7_80x80_MiniFASNetV2.onnx",
        architecture="MiniFASNetV2",
        scale=2.7,
    ),
    ModelSpec(
        checkpoint="4_0_0_80x80_MiniFASNetV1SE.pth",
        onnx_name="4_0_0_80x80_MiniFASNetV1SE.onnx",
        architecture="MiniFASNetV1SE",
        scale=4.0,
    ),
)


def get_kernel(height: int, width: int) -> tuple[int, int]:
    """Upstream ``utility.get_kernel``: 80x80 -> (5, 5)."""

    return ((height + 15) // 16, (width + 15) // 16)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def strip_module_prefix(state_dict: dict[str, Any]) -> dict[str, Any]:
    """Drop the ``module.`` prefix added by ``torch.nn.DataParallel``."""

    if any(key.startswith("module.") for key in state_dict):
        return OrderedDict(
            (key[len("module.") :] if key.startswith("module.") else key, value)
            for key, value in state_dict.items()
        )
    return state_dict


def _require_torch():
    try:
        import torch
    except ModuleNotFoundError as exc:  # pragma: no cover - environment guard
        raise SystemExit(
            "torch is required to export MiniFASNet; install it with "
            "`python -m pip install -e '.[export]'`"
        ) from exc
    return torch


def _load_architectures() -> dict[str, Any]:
    try:
        from _minifasnet_arch import ARCHITECTURES
    except ModuleNotFoundError:  # imported as scripts.export_minifasnet_onnx
        from scripts._minifasnet_arch import ARCHITECTURES
    return ARCHITECTURES


def _load_onnx():
    try:
        import onnx
    except ModuleNotFoundError as exc:  # pragma: no cover - environment guard
        raise SystemExit(
            "onnx is required to export MiniFASNet; install it with "
            "`python -m pip install -e '.[export]'`"
        ) from exc
    return onnx


def _load_onnxruntime():
    try:
        import onnxruntime as ort
    except ModuleNotFoundError as exc:  # pragma: no cover - environment guard
        raise SystemExit(
            "onnxruntime is required for export parity checks; install it with "
            "`python -m pip install -e '.[export]'`"
        ) from exc
    return ort


def build_model(torch, factory: Any, weights_path: Path, conv6_kernel: tuple[int, int]):
    model = factory(conv6_kernel=conv6_kernel)
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(strip_module_prefix(state_dict))
    model.eval()
    return model


def export_onnx(torch, model, onnx_path: Path, opset: int) -> None:
    dummy_input = torch.zeros(INPUT_SHAPE, dtype=torch.float32)
    kwargs = {
        "export_params": True,
        "opset_version": opset,
        "do_constant_folding": True,
        "input_names": [INPUT_NAME],
        "output_names": [OUTPUT_NAME],
        "dynamic_axes": None,
    }
    try:
        torch.onnx.export(model, dummy_input, str(onnx_path), dynamo=False, **kwargs)
    except TypeError:  # older torch without the ``dynamo`` switch
        torch.onnx.export(model, dummy_input, str(onnx_path), **kwargs)


def make_inputs(seed: int, count: int = PARITY_CASES) -> list[np.ndarray]:
    """Deterministic pixel-like float32 BGR inputs in [0, 255]."""

    rng = np.random.default_rng(seed)
    return [
        rng.integers(0, 256, size=INPUT_SHAPE, dtype=np.int32).astype(np.float32)
        for _ in range(count)
    ]


def run_parity(torch, model, session, inputs: list[np.ndarray]) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for index, input_array in enumerate(inputs):
        with torch.no_grad():
            pytorch_logits = model(torch.from_numpy(input_array)).detach().numpy()
        onnx_logits = session.run([OUTPUT_NAME], {INPUT_NAME: input_array})[0]
        max_abs_diff = float(np.max(np.abs(pytorch_logits - onnx_logits)))
        cases.append(
            {
                "index": index,
                "pytorch_logits": pytorch_logits.reshape(-1).tolist(),
                "onnx_logits": onnx_logits.reshape(-1).tolist(),
                "max_abs_diff": max_abs_diff,
            }
        )
    return cases


def build_reference(
    seed: int,
    opset: int,
    inputs: list[np.ndarray],
    model_results: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    return {
        "description": (
            "PyTorch vs ONNX Runtime parity reference for MiniFASNet PAD models. "
            "Generated by scripts/export_minifasnet_onnx.py."
        ),
        "input": {
            "dtype": "float32",
            "shape": list(INPUT_SHAPE),
            "layout": "NCHW",
            "color_order": "BGR",
            "value_range": [0, 255],
            "encoding": "base64_le",
            "note": (
                "Each case stores its input as input_b64, base64 of the raw "
                "little-endian float32 NCHW bytes."
            ),
            "preprocessing": (
                "Raw BGR float in [0, 255]. Upstream commented out div(255); "
                "do not divide by 255 and do not apply mean/std normalization."
            ),
        },
        "output": {
            "name": OUTPUT_NAME,
            "shape": [1, NUM_CLASSES],
            "classes": NUM_CLASSES,
            "real_class_index": REAL_CLASS_INDEX,
            "note": "3-class logits; label index 1 = real/bona fide.",
        },
        "seed": seed,
        "opset": opset,
        "cases": [
            {
                "index": index,
                "input_dtype": "float32",
                "input_shape": list(INPUT_SHAPE),
                "input_encoding": "base64_le",
                "input_b64": base64.b64encode(
                    np.asarray(input_array, dtype="<f4").tobytes()
                ).decode("ascii"),
            }
            for index, input_array in enumerate(inputs)
        ],
        "models": model_results,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    default_root = os.environ.get("SILENT_FACE_ROOT") or os.environ.get(
        "FACECHECK_SILENT_FACE_ROOT"
    )
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--silent-face-root",
        type=Path,
        default=Path(default_root) if default_root else None,
        help=(
            "path to the Silent-Face-Anti-Spoofing clone "
            "(default: $SILENT_FACE_ROOT or $FACECHECK_SILENT_FACE_ROOT)"
        ),
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=REPO_ROOT / "models",
        help="destination directory for the exported ONNX files (default: ./models)",
    )
    parser.add_argument(
        "--parity-json",
        type=Path,
        default=REPO_ROOT / "tests" / "fixtures" / "minifasnet_parity.json",
        help="where to write the replayable parity reference (default: tests/fixtures)",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=12,
        help="ONNX opset version (default: 12)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=PARITY_SEED,
        help=f"seed for the synthetic parity inputs (default: {PARITY_SEED})",
    )
    args = parser.parse_args(argv)
    if args.silent_face_root is None:
        parser.error(
            "--silent-face-root is required (or set SILENT_FACE_ROOT / "
            "FACECHECK_SILENT_FACE_ROOT)"
        )
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    torch = _require_torch()
    onnx = _load_onnx()
    ort = _load_onnxruntime()
    architectures = _load_architectures()

    weights_dir = args.silent_face_root / "resources" / "anti_spoof_models"
    missing = [
        spec.checkpoint
        for spec in MODEL_SPECS
        if not (weights_dir / spec.checkpoint).is_file()
    ]
    if missing:
        raise SystemExit(
            f"missing checkpoint(s) under {weights_dir}: {', '.join(missing)}"
        )

    conv6_kernel = get_kernel(INPUT_HEIGHT, INPUT_WIDTH)
    models_dir: Path = args.models_dir
    models_dir.mkdir(parents=True, exist_ok=True)

    inputs = make_inputs(args.seed)
    model_results: dict[str, dict[str, Any]] = {}
    failures: list[str] = []

    for spec in MODEL_SPECS:
        weights_path = weights_dir / spec.checkpoint
        factory = architectures[spec.architecture]
        model = build_model(torch, factory, weights_path, conv6_kernel)

        onnx_path = models_dir / spec.onnx_name
        export_onnx(torch, model, onnx_path, args.opset)
        onnx.checker.check_model(str(onnx_path))

        session = ort.InferenceSession(
            str(onnx_path), providers=["CPUExecutionProvider"]
        )
        cases = run_parity(torch, model, session, inputs)
        max_abs_diff = max(case["max_abs_diff"] for case in cases)
        checksum = sha256_file(onnx_path)

        model_results[spec.onnx_name] = {
            "checkpoint": spec.checkpoint,
            "architecture": spec.architecture,
            "scale": spec.scale,
            "conv6_kernel": list(conv6_kernel),
            "opset": args.opset,
            "max_abs_diff": max_abs_diff,
            "cases": cases,
        }

        print(
            f"exported {onnx_path} "
            f"({onnx_path.stat().st_size} bytes, sha256={checksum})"
        )
        print(
            f"  {spec.architecture} scale={spec.scale} "
            f"max_abs_diff={max_abs_diff:.3e}"
        )

        if max_abs_diff >= PARITY_TOLERANCE:
            failures.append(
                f"{spec.onnx_name}: max_abs_diff {max_abs_diff:.3e} "
                f">= {PARITY_TOLERANCE:.0e}"
            )

    reference = build_reference(args.seed, args.opset, inputs, model_results)
    parity_path: Path = args.parity_json
    parity_path.parent.mkdir(parents=True, exist_ok=True)
    parity_path.write_text(
        json.dumps(reference, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"wrote parity reference {parity_path}")

    if failures:
        raise SystemExit("parity check failed:\n  " + "\n  ".join(failures))
    return 0


if __name__ == "__main__":
    sys.exit(main())
