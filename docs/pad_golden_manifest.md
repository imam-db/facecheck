# PAD golden manifest

`tests/test_minifasnet_golden.py` checks the MiniFASNet ONNX models against a
semantic reference built from real photographs. The images are **never**
committed to this repository: they live in an out-of-repo directory supplied
through `FACECHECK_PAD_GOLDEN_DIR` (for example
`/workspace/facecheck-pad-golden`). Maintainers provide consented
real/print/screen photos; use only images you have permission to process.

When the variable is unset, or the directory or `manifest.json` is missing, the
golden tests skip cleanly. They also skip when the MiniFASNet ONNX files are
not present under the configured models directory. CI does not set the variable,
so CI runs only the synthetic parity fixture; the semantic check is a local
verification step.

## Schema

`manifest.json` is a JSON object:

| Field | Type | Required | Meaning |
| --- | --- | --- | --- |
| `schema_version` | integer | no | Currently `1`. |
| `description` | string | no | Human-readable explanation. |
| `cases` | array | yes | One object per image. |

Each case:

| Field | Type | Required | Meaning |
| --- | --- | --- | --- |
| `id` | string | yes | Stable case identifier (used as the pytest id). |
| `image` | string | yes | Image filename relative to the golden directory. |
| `label` | `"real"` or `"spoof"` | yes | Ground truth for the contrast assertions. |
| `bbox_xywh` | `[x, y, w, h]` | yes | Face box in pixels, in the full-resolution image. |
| `expected_real_prob` | number | yes | Average softmax(index 1) across both models on the correct raw-BGR crop. |
| `expected_per_model` | object | no | Per-model real probabilities keyed by ONNX filename. |
| `notes` | object | no | Informational scores such as `div255_real_prob` and `rgb_swap_real_prob`. |

## How values are produced

For each case:

1. Load the image as BGR.
2. Crop it with the verbatim upstream `CropImage` port
   (`scripts/_crop_image.py`) at scale 2.7 for `2.7_80x80_MiniFASNetV2.onnx`
   and scale 4.0 for `4_0_0_80x80_MiniFASNetV1SE.onnx`, resizing to 80x80.
3. Build a float32 NCHW blob with **raw BGR values in [0, 255]**.
4. Run the ONNX model, softmax the three logits, and read index 1
   (real/bona-fide).
5. Average the two model probabilities.

The upstream crop clamps the scale to fit the frame, shifts the box inside the
frame instead of padding, slices inclusively (`right_bottom + 1`), and uses
`cv2.resize` with its default `INTER_LINEAR` interpolation.

Generate or refresh the manifest with:

```bash
export FACECHECK_PAD_GOLDEN_DIR=/path/to/golden
.venv/bin/python scripts/export_minifasnet_onnx.py \
  --silent-face-root /path/to/Silent-Face-Anti-Spoofing \
  --golden-dir "$FACECHECK_PAD_GOLDEN_DIR" \
  --update-golden
```

`--update-golden` writes `expected_real_prob`, `expected_per_model`, and the
`div255_real_prob` / `rgb_swap_real_prob` notes back to the local manifest.
Omitting it prints the same numbers without modifying the file.

## What the tests assert

- The `cv2.dnn` runtime path reproduces `expected_real_prob` within `1e-3`.
- The `/255` variant puts every case below `0.1` real probability, and for
  `real` cases it differs from the expected value by more than `0.5`. Dividing
  by 255 marks every face as an attack.
- For `spoof` cases the RGB-swapped variant is at least `0.5` higher in real
  probability than the correct crop. The swap is fail-open: it accepts attacks.
  Real cases may still look real after the swap, so no assertion is made there.

## Privacy

- Keep golden images outside the repository and out of any release artifact.
- Use consented subjects only; do not add names or other personal data to the
  manifest.
- The manifest itself only stores numeric probabilities and bounding boxes.
