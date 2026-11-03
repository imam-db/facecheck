# ADR 0002: Presentation-attack detection (PAD) prep

- Status: Proposed
- Date: 2026-11-03
- Deciders: facecheck maintainers
- Related: `scripts/export_minifasnet_onnx.py`, `scripts/model_checksums.json`,
  `src/facecheck/liveness/`, `tests/test_minifasnet_parity.py`,
  `tests/test_liveness_crops.py`, `tests/test_liveness_boxes.py`

## Context

v0 performs face matching and deliberately has no liveness check, so a printed
photo, a screen replay, or a video call can spoof a probe. The roadmap calls
for passive anti-spoofing before any higher-value attendance use. This ADR
covers only the offline model preparation for that work; no API, service, or
storage behavior changes in this step.

## Decision

Use the MiniFASNet models from minivision-ai/Silent-Face-Anti-Spoofing at
commit `b6d5f04ad78778917853b25c778acef6d5626d15` (Apache-2.0), exported to
fixed-shape ONNX:

- `2.7_80x80_MiniFASNetV2.onnx` from `2.7_80x80_MiniFASNetV2.pth` (scale 2.7)
- `4_0_0_80x80_MiniFASNetV1SE.onnx` from `4_0_0_80x80_MiniFASNetV1SE.pth`
  (scale 4.0)

Details:

- Input is a fixed `1x3x80x80` float32 NCHW crop named `input`; output is
  `logits` of shape `1x3`, exported at opset 12.
- **Raw pixel gotcha.** The upstream pipeline commented out `div(255)`, so the
  models expect raw BGR floats in `[0, 255]`. There is no mean/std
  normalization. Dividing by 255 drives every face toward the attack classes,
  so the real probability collapses (below `0.1` on the golden cases) and every
  genuine face is rejected.
- **Channel-order gotcha.** Feeding RGB where the model expects BGR is
  fail-open: on the golden spoof cases the swapped crop scores more than `0.5`
  higher in real probability than the correct crop, so attacks are accepted.
  The runtime must keep the decode order intact.
- Index `1` of the 3 logits is the real / bona-fide class; indices `0` and `2`
  are attack classes. Softmax each model, then average index `1` across the two
  models.
- The architecture is vendored in `scripts/_minifasnet_arch.py`; only the
  `.pth` files are read from the Silent-Face clone. The export verifies the
  pinned `.pth` SHA-256 before loading.
- Exported `.onnx` files are published as GitHub release assets under the tag
  `models-minifasnet-v1` and are not committed. `scripts/model_checksums.json`
  pins their SHA-256 and byte sizes. The release asset digest is authoritative:
  re-exports with a different torch version may differ in bytes.
- Both MiniFASNet entries are required. `scripts/download_models.py` verifies
  them like YuNet and SFace, and CI fails if the download fails.
- PyTorch, ONNX, and ONNX Runtime live in the `export` optional-dependency
  group only. They are not runtime dependencies.

### Runtime module landed

The runtime building blocks now live under `src/facecheck/liveness/`: the
`LivenessDetector` protocol and `LivenessResult`, the `CropImage` runtime port
(kept bit-identical to `scripts/_crop_image.py`), the detection-to-full-res box
mapper, and `MiniFASNetLivenessDetector`. The detector scores a full-resolution
image plus a box and reports the averaged real probability; it applies no
threshold and is not wired into enroll or verify. The full PAD integration
remains Proposed (see "Carry into PR 2").

## Consequences

Positive:

- PAD model artifacts are reproducible offline and verified against PyTorch at
  export time (`max_abs_diff < 1e-4`) and replayable in tests without torch.
- Semantic goldens captured from consented images outside the repository catch
  preprocessing regressions that graph parity cannot.
- No change to the v0 FastAPI surface or storage schema.
- Permissive code and model-file license (Apache-2.0).

Negative / open risks:

- Training-data provenance for the upstream weights is undocumented. As with
  SFace, Apache-2.0 on the weights is not evidence of commercial clearance.
- No spoof-vs-genuine evaluation exists yet. Accuracy on the actual device
  fleet, printed photos, and screen replays is unmeasured.
- This ADR only prepares the artifact; threshold selection and integration into
  the verification flow are separate, future decisions.

## Carry into PR 2

The integration PR must not treat the export as a drop-in detector. The
following notes are carry-over requirements:

- Port upstream `CropImage` verbatim: clamp the scale to fit, shift the box
  inside the frame (never pad), slice inclusively (`right_bottom + 1`), and use
  `cv2.resize` with `INTER_LINEAR`.
- Crop from the full-resolution decode, mapping the YuNet box out of the
  960-pixel detection coordinates.
- The upstream PAD box convention differs from RetinaFace's; calibrate our own
  threshold instead of reusing upstream numbers.
- Softmax each model independently, then average the index-`1` probabilities.
- Run PAD only after the face gate passes; on noise inputs the scores are about
  `0.99` real, so a PAD score without a detected face is meaningless.
- Fail closed at startup when liveness mode is not `off` and the PAD models are
  missing.
- Log the crop aspect ratio; the upstream models were trained on roughly 3:4
  boxes.

### Watch first in PR 2

1. Run the semantic golden test against the runtime `LivenessDetector`, not the
   `scripts/` helpers. Keep the scripts crop as the reference and assert
   `np.array_equal` between it and the runtime crop on random boxes, so the
   runtime cannot drift from the values the manifest pins.
2. Move the crop implementation into `src/facecheck/liveness/` with the
   Apache-2.0 modification notice. The runtime cannot import from `scripts/`,
   which is not part of the installed package.
3. Coordinate frames: `detect_and_embed` must return the original frame and the
   detection scale; map the YuNet box back to full resolution before cropping.
   Pin the integer `[x, y, w, h]` rounding with a unit test so the box that
   reaches the PAD crop is deterministic.
4. Guard the `cv2.dnn.Net` with a thread lock, or use per-thread nets, as the
   embedder does. `cv2.dnn.Net` is not safe for concurrent `setInput`/`forward`.
5. Fail closed at startup when the liveness mode is not `off`; run PAD only
   after the face gate passes; never emit a match score on `liveness_failed`.
6. Keep golden coverage in CI via a PR-template checkbox (or an encrypted golden
   set later). The images cannot be committed, so the local run must be an
   explicit review step.
7. Default the mode to `off` until the evaluation picks a threshold. Expose the
   PAD `model_version` (both SHA-256 prefixes) in `/v1/meta` and in responses so
   a score can be traced to the artifacts that produced it.
