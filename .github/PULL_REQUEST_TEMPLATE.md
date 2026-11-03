# Pull request

## Checklist

- [ ] Tests pass locally (`pytest -q`).
- [ ] Lint passes (`ruff check .`).
- [ ] Model-backed tests were run with the ONNX files present.
- [ ] PAD golden run pasted below (`FACECHECK_PAD_GOLDEN_DIR=... pytest -q -m minifasnet`
      or the golden file), including the three `expected_real_prob` lines from the
      runtime detector.

## PAD golden run

```text
paste the three expected_real_prob lines from the runtime MiniFASNetLivenessDetector here
```
