#!/usr/bin/env python3
"""Export Minivision MiniFASNet anti-spoofing checkpoints to fixed-shape ONNX.

The two upstream checkpoints shipped with Silent-Face-Anti-Spoofing are:

- ``2.7_80x80_MiniFASNetV2.pth``     -> ``MiniFASNetV2``,   scale 2.7
- ``4_0_0_80x80_MiniFASNetV1SE.pth`` -> ``MiniFASNetV1SE``, scale 4.0

Both consume an 80x80 face crop and produce **three logits**. In the upstream
label convention index ``1`` is the real / bona-fide class; indices ``0`` and
``2`` are presentation-attack classes.

The upstream clone is pinned to commit
``b6d5f04ad78778917853b25c778acef6d5626d15``. The ``.pth`` files are verified
against pinned SHA-256 digests before they are loaded. The ONNX file SHA-256
recorded in ``scripts/model_checksums.json`` (and published as a GitHub release
asset) is authoritative: re-exports with a different torch version may produce
different bytes.

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
``tests/test_minifasnet_parity.py``. The fixture is only written when every
model passes parity.

Optional semantic goldens
-------------------------
``--golden-dir`` (or ``$FACECHECK_PAD_GOLDEN_DIR``) points at a directory of
consented real/print/screen images plus ``manifest.json`` (schema in
``docs/pad_golden_manifest.md``). For each case this script reproduces the
upstream crop at scales 2.7 and 4.0, scores raw BGR [0, 255] inputs, softmaxes
each model, and averages index ``1``. Pass ``--update-golden`` to write the
computed values back to that local manifest; the images stay outside the repo.
When ``--silent-face-root`` is provided as well, every golden crop is also
compared with the upstream ``CropImage`` using ``np.array_equal`` so a drift in
the verbatim port fails the run instead of silently changing the semantic
reference.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import inspect
import json
import os
import sys
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:
    from _crop_image import CropImage
except ModuleNotFoundError:  # imported as scripts.export_minifasnet_onnx
    from scripts._crop_image import CropImage

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
UPSTREAM_COMMIT = "b6d5f04ad78778917853b25c778acef6d5626d15"

PTH_SHA256 = {
    "2.7_80x80_MiniFASNetV2.pth": (
        "a5eb02e1843f19b5386b953cc4c9f011c3f985d0ee2bb9819eea9a142099bec0"
    ),
    "4_0_0_80x80_MiniFASNetV1SE.pth": (
        "84ee1d37d96894d5e82de5a57df044ef80a58be2b218b5ed7cdfd875ec2f5990"
    ),
}


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


def verify_checkpoint(path: Path) -> str:
    actual = sha256_file(path)
    expected = PTH_SHA256[path.name]
    if actual != expected:
        raise SystemExit(
            f"{path.name}: checkpoint SHA-256 mismatch\n"
            f"  expected {expected}\n  actual   {actual}\n"
            f"Pin the clone to Silent-Face-Anti-Spoofing {UPSTREAM_COMMIT}."
        )
    return actual


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
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        torch.onnx.export(model, dummy_input, str(onnx_path), dynamo=False, **kwargs)
    else:  # older torch without the ``dynamo`` switch
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
    toolchain: dict[str, str],
) -> dict[str, Any]:
    return {
        "description": (
            "PyTorch vs ONNX Runtime parity reference for MiniFASNet PAD models. "
            "Generated by scripts/export_minifasnet_onnx.py."
        ),
        "upstream_commit": UPSTREAM_COMMIT,
        "toolchain": toolchain,
        "release_asset_sha256_authoritative": True,
        "provenance_note": (
            "The GitHub release asset SHA-256 in scripts/model_checksums.json is "
            "authoritative. Re-exports with a different torch version may differ in bytes."
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


def make_onnx_runner(onnx_path: Path):
    """Return a callable (blob) -> logits, preferring ONNX Runtime."""

    try:
        import onnxruntime as ort
    except ModuleNotFoundError:
        net = cv2.dnn.readNetFromONNX(str(onnx_path))

        def run_cv2(blob: np.ndarray) -> np.ndarray:
            net.setInput(blob)
            return net.forward()

        return run_cv2

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    def run_ort(blob: np.ndarray) -> np.ndarray:
        return session.run([OUTPUT_NAME], {INPUT_NAME: blob})[0]

    return run_ort


def softmax(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64).reshape(-1)
    shifted = values - values.max()
    exp = np.exp(shifted)
    return exp / exp.sum()


def make_blob(crop: np.ndarray, *, div255: bool = False, rgb_swap: bool = False) -> np.ndarray:
    image = crop[:, :, ::-1] if rgb_swap else crop
    array = np.asarray(image, dtype=np.float32)
    if div255:
        array = array / 255.0
    return np.ascontiguousarray(array.transpose(2, 0, 1)[None])


def score_case(
    image: np.ndarray,
    bbox_xywh: list[int],
    runners: dict[str, Any],
    *,
    div255: bool = False,
    rgb_swap: bool = False,
) -> dict[str, float]:
    probs: dict[str, float] = {}
    cropper = CropImage()
    for spec in MODEL_SPECS:
        crop = cropper.crop(image, bbox_xywh, spec.scale, INPUT_WIDTH, INPUT_HEIGHT)
        blob = make_blob(crop, div255=div255, rgb_swap=rgb_swap)
        probs[spec.onnx_name] = float(softmax(runners[spec.onnx_name](blob))[REAL_CLASS_INDEX])
    return probs


def load_upstream_crop_image(silent_face_root: Path):
    """Import upstream ``CropImage`` from a pinned Silent-Face clone."""

    root = str(silent_face_root.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    try:
        from src.generate_patches import CropImage as UpstreamCropImage
    except ModuleNotFoundError as exc:
        raise SystemExit(
            f"could not import upstream CropImage from {silent_face_root}: {exc}"
        ) from exc
    return UpstreamCropImage


def verify_upstream_crop(
    image: np.ndarray,
    bbox_xywh: list[int],
    case_id: str,
    upstream_cropper: Any,
) -> None:
    """Assert the vendored crop port matches upstream bit-for-bit at both scales."""

    cropper = CropImage()
    for spec in MODEL_SPECS:
        port_crop = cropper.crop(image, bbox_xywh, spec.scale, INPUT_WIDTH, INPUT_HEIGHT)
        upstream_crop = upstream_cropper.crop(
            image, bbox_xywh, spec.scale, INPUT_WIDTH, INPUT_HEIGHT
        )
        if np.array_equal(port_crop, upstream_crop):
            continue
        differing = (
            int(np.count_nonzero(port_crop != upstream_crop))
            if port_crop.shape == upstream_crop.shape
            else "shape-dependent"
        )
        raise SystemExit(
            f"{case_id}: crop mismatch vs upstream at scale {spec.scale} "
            f"(port {port_crop.shape} vs upstream {upstream_crop.shape}, "
            f"differing values: {differing})"
        )


def evaluate_golden(
    golden_dir: Path,
    models_dir: Path,
    *,
    update: bool,
    silent_face_root: Path | None = None,
) -> dict[str, Any]:
    manifest_path = golden_dir / "manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"missing golden manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cases = manifest.get("cases") or []
    if not cases:
        raise SystemExit(f"{manifest_path}: no cases")

    runners = {
        spec.onnx_name: make_onnx_runner(models_dir / spec.onnx_name) for spec in MODEL_SPECS
    }
    upstream_cropper = (
        load_upstream_crop_image(silent_face_root)() if silent_face_root is not None else None
    )

    for case in cases:
        image_path = golden_dir / case["image"]
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise SystemExit(f"could not read golden image: {image_path}")
        bbox = [int(value) for value in case["bbox_xywh"]]
        case_id = case.get("id", case["image"])

        if upstream_cropper is not None:
            verify_upstream_crop(image, bbox, case_id, upstream_cropper)

        per_model = score_case(image, bbox, runners)
        expected = float(np.mean(list(per_model.values())))
        div255 = score_case(image, bbox, runners, div255=True)
        rgb_swap = score_case(image, bbox, runners, rgb_swap=True)
        div255_avg = float(np.mean(list(div255.values())))
        rgb_swap_avg = float(np.mean(list(rgb_swap.values())))

        print(
            f"golden {case.get('id', case['image'])} "
            f"label={case.get('label', '?')} expected_real_prob={expected:.6f}"
        )
        for name, prob in per_model.items():
            print(f"  {name}: {prob:.6f}")
        print(f"  div255_real_prob={div255_avg:.6f} rgb_swap_real_prob={rgb_swap_avg:.6f}")

        if update:
            case["expected_real_prob"] = round(expected, 6)
            case["expected_per_model"] = {
                name: round(prob, 6) for name, prob in per_model.items()
            }
            notes = dict(case.get("notes") or {})
            notes["div255_real_prob"] = round(div255_avg, 6)
            notes["rgb_swap_real_prob"] = round(rgb_swap_avg, 6)
            case["notes"] = notes

    if update:
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(f"updated golden manifest {manifest_path}")

    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    default_root = os.environ.get("SILENT_FACE_ROOT") or os.environ.get(
        "FACECHECK_SILENT_FACE_ROOT"
    )
    default_golden = os.environ.get("FACECHECK_PAD_GOLDEN_DIR")
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
    parser.add_argument(
        "--golden-dir",
        type=Path,
        default=Path(default_golden) if default_golden else None,
        help=(
            "directory with a PAD golden manifest.json and images "
            "(default: $FACECHECK_PAD_GOLDEN_DIR)"
        ),
    )
    parser.add_argument(
        "--update-golden",
        action="store_true",
        help="write computed golden scores back to <golden-dir>/manifest.json",
    )
    args = parser.parse_args(argv)
    if args.silent_face_root is None:
        parser.error(
            "--silent-face-root is required (or set SILENT_FACE_ROOT / "
            "FACECHECK_SILENT_FACE_ROOT)"
        )
    if args.update_golden and args.golden_dir is None:
        parser.error("--update-golden requires --golden-dir or FACECHECK_PAD_GOLDEN_DIR")
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
        pth_sha256 = verify_checkpoint(weights_path)
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
            "pth_sha256": pth_sha256,
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

    if failures:
        raise SystemExit(
            "parity check failed; parity fixture not written:\n  "
            + "\n  ".join(failures)
        )

    reference = build_reference(
        args.seed,
        args.opset,
        inputs,
        model_results,
        {
            "torch": torch.__version__,
            "onnx": onnx.__version__,
            "onnxruntime": ort.__version__,
        },
    )
    parity_path: Path = args.parity_json
    parity_path.parent.mkdir(parents=True, exist_ok=True)
    parity_path.write_text(
        json.dumps(reference, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"wrote parity reference {parity_path}")

    if args.golden_dir is not None:
        evaluate_golden(
            args.golden_dir,
            models_dir,
            update=args.update_golden,
            silent_face_root=args.silent_face_root,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
