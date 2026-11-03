# Pull request

## Checklist

- [ ] Tests pass locally (`pytest -q`).
- [ ] Lint passes (`ruff check .`).
- [ ] Model-backed tests were run with the ONNX files present.
- [ ] PAD golden run pasted below
      (`FACECHECK_PAD_GOLDEN_DIR=... python scripts/pad_golden_report.py`),
      with one line per case showing the manifest `expected_real_prob` and the
      runtime score.

## PAD golden run

```text
paste the pad_golden_report.py output here (one line per case)
```
