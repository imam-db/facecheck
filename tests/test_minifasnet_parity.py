"""ONNX parity tests for the exported MiniFASNet PAD models.

The ``.onnx`` files are not committed (see the README). When they are missing
under the configured models directory these tests skip. They replay the exact
float32 inputs stored in ``tests/fixtures/minifasnet_parity.json`` and compare
ONNX output against the PyTorch logits recorded by
``scripts/export_minifasnet_onnx.py``, so torch is not needed here.

Both ``cv2.dnn`` and ``onnxruntime`` are exercised when available. ``cv2`` is a
hard runtime dependency, so ``cv2.dnn`` always runs; ``onnxruntime`` is only an
export/development dependency, so its parametrization skips cleanly when the
package is not installed.

The models emit **3 logits** per 80x80 crop. In the upstream Minivision
convention label index ``1`` is the real / bona-fide class. Inputs are raw BGR
floats in [0, 255]; do not divide by 255.
"""

from __future__ import annotations

import base64
import json
from collections.abc import Callable
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from facecheck.config import Settings

pytestmark = pytest.mark.minifasnet

FIXTURE_PATH = Path(__file__).with_name("fixtures") / "minifasnet_parity.json"
ORT_TOLERANCE = 1e-4
CV2_TOLERANCE = 1e-3

if not FIXTURE_PATH.is_file():
    pytest.skip(
        f"missing parity fixture {FIXTURE_PATH}; run scripts/export_minifasnet_onnx.py",
        allow_module_level=True,
    )

REFERENCE: dict[str, Any] = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
MODEL_NAMES = sorted(REFERENCE["models"])


def _models_dir() -> Path:
    return Settings(_env_file=None).models_dir


def _has_onnxruntime() -> bool:
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        return False
    return True


BACKENDS = [
    "cv2.dnn",
    pytest.param(
        "onnxruntime",
        marks=pytest.mark.skipif(
            not _has_onnxruntime(), reason="onnxruntime is not installed"
        ),
    ),
]


def _missing_models() -> list[str]:
    models_dir = _models_dir()
    return [name for name in MODEL_NAMES if not (models_dir / name).is_file()]


requires_models = pytest.mark.skipif(
    bool(_missing_models()),
    reason=f"MiniFASNet ONNX models not exported: {', '.join(_missing_models())}",
)


@cache
def _onnxruntime_session(path: str):
    import onnxruntime as ort

    return ort.InferenceSession(path, providers=["CPUExecutionProvider"])


@cache
def _cv2_net(path: str):
    import cv2

    return cv2.dnn.readNetFromONNX(path)


def _run_onnxruntime(path: Path, array: np.ndarray) -> np.ndarray:
    session = _onnxruntime_session(str(path))
    return session.run(["logits"], {"input": array})[0]


def _run_cv2(path: Path, array: np.ndarray) -> np.ndarray:
    net = _cv2_net(str(path))
    net.setInput(array)
    return net.forward()


RUNNERS: dict[str, Callable[[Path, np.ndarray], np.ndarray]] = {
    "onnxruntime": _run_onnxruntime,
    "cv2.dnn": _run_cv2,
}


def _decode_input(case: dict[str, Any]) -> np.ndarray:
    """Decode a fixture input, supporting ``base64_le`` and legacy lists."""

    if "input_b64" not in case:
        return np.asarray(case["input"], dtype=np.float32)

    encoding = case.get("input_encoding", "base64_le")
    if encoding != "base64_le":
        raise ValueError(f"unsupported fixture input encoding: {encoding!r}")

    dtype = np.dtype(case.get("input_dtype", "float32")).newbyteorder("<")
    shape = tuple(case.get("input_shape", REFERENCE["input"]["shape"]))
    raw = base64.b64decode(case["input_b64"])
    return np.frombuffer(raw, dtype=dtype).reshape(shape).astype(np.float32, copy=False)


@requires_models
@pytest.mark.parametrize("model_name", MODEL_NAMES)
@pytest.mark.parametrize("backend", BACKENDS)
def test_minifasnet_matches_pytorch_reference(model_name: str, backend: str) -> None:
    entry: dict[str, Any] = REFERENCE["models"][model_name]
    assert float(entry["max_abs_diff"]) < ORT_TOLERANCE

    tolerance = ORT_TOLERANCE if backend == "onnxruntime" else CV2_TOLERANCE
    tolerance = max(tolerance, float(entry["max_abs_diff"]))

    model_path = _models_dir() / model_name
    recorded_cases = {case["index"]: case for case in entry["cases"]}

    for case in REFERENCE["cases"]:
        input_array = _decode_input(case)
        assert input_array.shape == tuple(REFERENCE["input"]["shape"])

        expected = np.asarray(
            recorded_cases[case["index"]]["pytorch_logits"], dtype=np.float32
        )
        actual = np.asarray(RUNNERS[backend](model_path, input_array), dtype=np.float32)

        # 3-class logits; index 1 is real/bona fide in the upstream models.
        assert actual.shape == (1, 3)
        max_abs_diff = float(np.max(np.abs(actual.reshape(-1) - expected)))
        assert max_abs_diff < tolerance, (
            f"{model_name} case {case['index']} via {backend}: "
            f"max_abs_diff {max_abs_diff:.3e} >= {tolerance:.0e}"
        )
