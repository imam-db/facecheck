# ADR 0002: Presentation-attack detection (PAD) prep

- Status: Proposed
- Date: 2026-11-03
- Deciders: facecheck maintainers
- Related: `scripts/export_minifasnet_onnx.py`, `scripts/model_checksums.json`,
  `tests/test_minifasnet_parity.py`

## Context

v0 performs face matching and deliberately has no liveness check, so a printed
photo, a screen replay, or a video call can spoof a probe. The roadmap calls
for passive anti-spoofing before any higher-value attendance use. This ADR
covers only the offline model preparation for that work; no API, service, or
storage behavior changes in this step.

## Decision

Use the MiniFASNet models from minivision-ai/Silent-Face-Anti-Spoofing
(Apache-2.0), exported to fixed-shape ONNX:

- `2.7_80x80_MiniFASNetV2.onnx` from `2.7_80x80_MiniFASNetV2.pth` (scale 2.7)
- `4_0_0_80x80_MiniFASNetV1SE.onnx` from `4_0_0_80x80_MiniFASNetV1SE.pth`
  (scale 4.0)

Details:

- Input is a fixed `1x3x80x80` float32 NCHW crop named `input`; output is
  `logits` of shape `1x3`, exported at opset 12.
- **Raw pixel gotcha.** The upstream pipeline commented out `div(255)`, so the
  models expect raw BGR floats in `[0, 255]`. There is no mean/std
  normalization. Feeding `[0, 1]` values silently degrades outputs.
- Index `1` of the 3 logits is the real / bona-fide class; indices `0` and `2`
  are attack classes.
- The architecture is vendored in `scripts/_minifasnet_arch.py`; only the
  `.pth` files are read from the Silent-Face clone.
- Exported `.onnx` files are published as GitHub release assets under the tag
  `models-minifasnet-v1` and are not committed. `scripts/model_checksums.json`
  pins their SHA-256.
- PyTorch, ONNX, and ONNX Runtime live in the `export` optional-dependency
  group only. They are not runtime dependencies.

## Consequences

Positive:

- PAD model artifacts are reproducible offline and verified against PyTorch at
  export time (`max_abs_diff < 1e-4`) and replayable in tests without torch.
- No change to the v0 FastAPI surface or storage schema.
- Permissive code and model-file license (Apache-2.0).

Negative / open risks:

- The release `models-minifasnet-v1` does not exist yet; Imam must create it
  and upload both ONNX files before `download_models.py` can fetch them.
- Training-data provenance for the upstream weights is undocumented. As with
  SFace, Apache-2.0 on the weights is not evidence of commercial clearance.
- No spoof-vs-genuine evaluation exists yet. Accuracy on the actual device
  fleet, printed photos, and screen replays is unmeasured.
- This ADR only prepares the artifact; threshold selection and integration into
  the verification flow are separate, future decisions.
