# PAD capture protocol

This document defines how to collect, store, and evaluate consented
presentation-attack detection (PAD) images for facecheck. It covers the capture
rules, the manifest schema, the encryption layout, and the metrics emitted by
`scripts/pad_eval.py`.

The harness produces evidence for a later threshold decision. It does **not**
wire liveness into the enroll or verify API, and its numbers are not a
product-ready PAD claim until the capture set is large enough to support one.

## Non-negotiables

1. **Consent comes first.** A presentation is included only when the subject
   consented to PAD evaluation and the manifest row has `consent=true`. The
   harness skips every row whose consent value is not true.
2. **Captures live outside the repository.** Use a dedicated directory such as
   `/workspace/facecheck-pad-capture`, never a path under the git working tree.
   Do not commit images, encrypted images, manifests with real paths, or report
   files that contain per-person data.
3. **Encrypt at rest.** Keep the capture tree on an encrypted volume and/or
   store an encrypted copy produced by `scripts/pad_capture_crypto.py`. The key
   comes from the environment and is never printed, logged, or committed.
4. **Use pseudonymous identifiers.** `subject_id` values such as `subj_01` and
   `session_id` values such as `s01` are sufficient. Never put real names,
   employee numbers, phone numbers, or other direct identifiers in a manifest.
5. **Strip location metadata.** Remove GPS and other unnecessary EXIF fields
   from stored originals. Keep the image pixels; do not keep where they were
   taken.

## Environment

The evaluation harness is configured with two evaluation-only environment
variables. They are deliberately not `Settings` fields because the API does not
read them.

| Variable | Meaning |
| --- | --- |
| `FACECHECK_PAD_CAPTURE_DIR` | Capture root, parallel to `FACECHECK_PAD_GOLDEN_DIR` for the semantic goldens. |
| `FACECHECK_PAD_CAPTURE_KEY` | 32-byte AES-GCM key for the capture tree, as urlsafe base64, standard base64, or 64 hex characters. |

Generate a key without printing it into shell history if the environment
supports it:

```bash
export FACECHECK_PAD_CAPTURE_KEY="$(
  python -c 'import base64,os; print(base64.urlsafe_b64encode(os.urandom(32)).decode())'
)"
```

### Encryption layout

`scripts/pad_capture_crypto.py` encrypts every file under a source tree into a
sibling tree with a `.aesgcm` suffix:

```text
pad-capture/                       pad-capture.encrypted/
  manifest.csv                        manifest.csv.aesgcm
  presentations/subj_01/a.jpg         presentations/subj_01/a.jpg.aesgcm
  presentations/subj_02/b.jpg         presentations/subj_02/b.jpg.aesgcm
```

Each envelope is `b"FCPAD1" + 12-byte nonce + AES-GCM ciphertext`. The
associated data is the file's relative POSIX path, so moving or renaming an
envelope makes decryption fail instead of silently swapping captures. The
destination must be new or empty, which prevents accidentally mixing two
capture sets.

```bash
python scripts/pad_capture_crypto.py encrypt \
  --src /path/to/pad-capture \
  --dst /path/to/pad-capture.encrypted

python scripts/pad_capture_crypto.py decrypt \
  --src /path/to/pad-capture.encrypted \
  --dst /tmp/pad-capture-plain
```

Decrypt to a short-lived local path for evaluation and remove it afterwards.
The script refuses to run when the key is missing and never prints the key.

## Presentation classes

| `class` | Meaning | Typical `attack_instrument` |
| --- | --- | --- |
| `bona_fide` | A live person presenting their own face. Empty instrument. | empty |
| `print` | A printed photo of the consented subject held in front of the camera. | `glossy_print`, `matte_print` |
| `phone_screen` | The photo or video shown on a phone display. | `phone_oled`, `phone_lcd` |
| `laptop_screen` | The photo or video shown on a laptop display. | `laptop_lcd` |

`video_replay` is a future class. Do not invent it in manifests until the
harness and the report tables include it; a replay is currently captured as
`phone_screen` or `laptop_screen` with a note in `condition`.

## Presentations, not frames

The metric unit is a **presentation**, which is one attempt to get a PAD
decision. This is the ISO/IEC 30107-3 convention and it matters:

- A 10-second video of one phone-screen attack is **one** presentation in the
  manifest. Pick a representative frame, or list explicitly independent
  presentations if the protocol is frame-level.
- Do not list every frame of one video as a separate attack. That inflates the
  denominator, narrows confidence intervals artificially, and makes APCER look
  more certain than it is.
- If frames really are independent presentations (for example, separate
  attempts made minutes apart), say so in `condition` and list them
  individually.
- One subject may contribute several presentations. The report counts unique
  subjects and sessions, but BPCER is still a presentation-level rate.

## Calibration versus test splits

Split by `session_id` (or capture day), not by randomly chosen frames:

- Frames from the same session share lighting, device, subject, and attack
  instrument. A random frame split leaks those factors across calibration and
  test.
- Put entire sessions in calibration or in test. Prefer a manifest per split,
  or a stable session prefix if one CSV holds both.
- For claims about **new people**, also hold out subjects, not only sessions. A
  threshold tuned on `subj_01` and tested on another session of `subj_01` does
  not establish BPCER for the population.
- The evaluation harness reports aggregates for whatever manifest it is given.
  It does not invent a split. Make the split explicit before collecting data,
  and run the calibration manifest and the test manifest separately.

## Capture variety

A pilot can be narrow, but record what was actually captured so the limits of
the result are visible.

- **Lighting:** daylight near a window, ordinary indoor light, backlit against
  a window, and dim indoor light. A screen attack in the dark is a realistic
  case.
- **Devices:** one recent phone plus a laptop webcam is acceptable for a pilot.
  Record `capture_device` for each presentation. The cheap-Android fleet is a
  known gap: low-end front cameras, heavy noise reduction, and compressed
  uploads behave differently from a recent flagship.
- **Pose and distance:** straight-on plus modest yaw/pitch, at the distance the
  attendance flow expects. Do not use extreme close-ups that the production
  quality gate would reject.
- **Appearance:** include the range of intended users, including hijab styles,
  glasses, facial hair, and ordinary indoor skin tones. Do not collect images
  of people who have not consented, including bystanders.
- **Attack geometry:** hold the print or screen roughly where a phone would be,
  at several angles and distances, including a slight tilt that creates
  reflections.
- **No cosmetic retouching.** Do not apply beauty filters, denoising, or
  chat-app compression to evaluation images.

## Image format

- Capture originals as full-size files. Do not send them through a chat app that
  recompresses them; that removes exactly the screen texture and print artefacts
  PAD depends on.
- Minimum resolution is about 720p (1280x720). Prefer the phone's native
  resolution. Keep the JPEG quality high; re-encoding at low quality is a
  preprocessing step, not a capture protocol.
- Record the original and used resolutions in the harness output rather than
  editing files by hand.
- The evaluation harness loads images with OpenCV and can downscale with
  `--max-side` for latency comparison. It does not apply the API's 25 MP
  rejection (see below).
- Do not crop the face yourself. The harness runs the production detector and
  quality gate so detection failures remain visible.

## Making the attack presentations

Use only a bona-fide photo of the same consented subject. Do not use a
different person's photo, a stock face, or a face generated from a public
figure.

- **Print:** print the consented photo on a colour printer. Capture at least a
  glossy and a matte instrument if available. Hold the sheet flat or slightly
  curved, fill a realistic portion of the frame, and include angles where the
  paper reflects light.
- **Phone screen:** display the consented photo (or a short consented video) on
  a phone, set a realistic brightness, and capture the phone display with the
  evaluation device. Record `phone_oled` or `phone_lcd`.
- **Laptop screen:** display the same material on a laptop and capture the
  screen. Include a slightly off-axis view where the panel is visible.
- **Pairing:** keep the attack row under the same `subject_id` as the source
  photo so the report can later compute per-subject error rates. Use the same
  `session_id` for presentations captured together, or a new session when the
  capture day or setup changes.
- **Do not** composite the attack into a video call, use a deepfake, or use an
  instrument not listed in the manifest. Those are separate future classes.

## Sample sizes

The confidence bounds, not the point estimate, decide whether a result supports
a claim.

- With **zero** errors in `n` presentations, the one-sided 95% upper bound is
  `1 - 0.05 ** (1/n)`, approximately `3/n` (the rule of three). About 100
  presentations per class bound a zero-error rate near **3%**; about 300 bound
  it near **1%**.
- A pilot of roughly **30 bona-fide** presentations plus **20 per attack type**
  is useful for exploring thresholds, spotting broken instruments, and
  estimating how much capture work a full set needs. It is **not** enough to
  claim an APCER or BPCER target.
- The harness reports a BPCER-at-APCER point only when at least 10 attack
  presentations were scored. That is a reporting guard, not a statistical
  sufficiency threshold.
- More subjects matter more than more frames of one subject for BPCER. A
  single-subject set can show that the pipeline runs; it cannot show that bona
  fide users in general are accepted.

## Manifest schema

The manifest is a UTF-8 CSV file. Each data row is one presentation. Paths are
relative to the capture root and must not be absolute or contain `..`.

| Column | Required | Meaning |
| --- | --- | --- |
| `path` | yes | Relative path under the capture root. |
| `subject_id` | yes | Pseudonymous subject id, for example `subj_01`. Never a real name. |
| `session_id` | yes | Capture session or day id used for calibration/test splits. |
| `class` | yes | `bona_fide`, `print`, `phone_screen`, or `laptop_screen`. |
| `attack_instrument` | yes (may be empty) | Empty for `bona_fide`; otherwise a short instrument id such as `glossy_print`, `phone_oled`, or `laptop_lcd`. |
| `capture_device` | yes (may be empty) | Device that took this presentation, for example `phone_front` or `laptop_webcam`. |
| `condition` | yes (may be empty) | Short free-text lighting/pose note. Keep it non-identifying. |
| `consent` | yes | `true` to include the row. `false`, blank, and unknown values are skipped. |
| `notes` | no | Optional free text, ignored by the metrics. |

Consent parsing is deliberately narrow: only `true`, `1`, `yes`, and `y`
(case-insensitive) include a row. Everything else is counted as skipped.

Example with synthetic paths and no real data:

```csv
path,subject_id,session_id,class,attack_instrument,capture_device,condition,consent,notes
presentations/subj_01/s01_bona_01.jpg,subj_01,s01,bona_fide,,phone_front,daylight indoor,true,synthetic example row
presentations/subj_01/s01_print_01.jpg,subj_01,s01,print,glossy_print,phone_front,daylight indoor,true,synthetic example row
presentations/subj_02/s02_phone_01.jpg,subj_02,s02,phone_screen,phone_oled,laptop_webcam,dim indoor,true,synthetic example row
presentations/subj_02/s02_laptop_01.jpg,subj_02,s02,laptop_screen,laptop_lcd,phone_front,backlit,true,synthetic example row
```

The same four rows are committed as a parser fixture at
`tests/fixtures/pad_manifest_example.csv`. The fixture has no images; it exists
to keep the schema exercised in CI.

## Running the evaluation

```bash
export FACECHECK_PAD_CAPTURE_DIR=/path/to/pad-capture
export FACECHECK_PAD_CAPTURE_KEY=...  # only needed for the encryption helper

python -m eval.pad_eval \
  --manifest "$FACECHECK_PAD_CAPTURE_DIR/manifest.csv" \
  --detector-threshold 0.9 \
  --diagnostic-detector-threshold 0.5 \
  --max-side 0 \
  --bpcer-target 0.05
# or: python scripts/pad_eval.py --manifest ...
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--manifest` | required | Manifest CSV path. |
| `--data-dir` | `$FACECHECK_PAD_CAPTURE_DIR` or the manifest directory | Capture root. |
| `--detector-threshold` | `0.9` | Production YuNet detection score threshold. |
| `--diagnostic-detector-threshold` | `0.5` | Lower threshold for the separate diagnostic pass; values at or above the production threshold disable it. |
| `--max-side` | `0` | Downscale so the longest side is at most this value; `0` keeps full resolution. |
| `--bpcer-target` | `0.05` | BPCER budget for selecting an operating threshold. |
| `--apcer-target` | `0.10` | APCER budget for reporting BPCER at an attack-accept point. |
| `--out-json` | `<manifest dir>/pad_report.json` | Aggregate JSON report path. |
| `--out-md` | `<manifest dir>/pad_report.md` | Aggregate Markdown report path. |

Reports default next to the manifest, outside the repository. They contain
aggregate counts, rates, intervals, threshold sweeps, latency, and resolution
summaries. They do not contain image paths, subject or session ids, or
per-presentation scores. If you point an output inside the repository, still
only aggregates are written, but keeping reports outside the tree is preferred.

Exit code `0` means the run completed. Exit code `2` means a missing manifest or
directory, an empty usable manifest, a malformed manifest, or missing ONNX
models.

### What the harness measures

- **BPCER** at the production threshold: bona-fide presentations rejected by
  the PAD.
- **APCER** per attack class and overall: attack presentations accepted by the
  PAD. The worst-class APCER is the maximum over the classes present.
- **IAPMR** is reported as not measured because this PR does not wire the face
  matcher into the harness. The named helper computes matched/total for callers
  that later have match outcomes.
- **Detection failures** are presentations the production face detector could
  not use. They are counted per class and excluded from APCER and BPCER, so a
  detector that rejects hard frames does not look like a good PAD.
- **Diagnostic pass**: rows that failed production detection are re-run with the
  lower diagnostic threshold. Their PAD scores appear in a separate section and
  are never mixed into production rates.
- **Threshold sweep** and operating points: the highest threshold meeting the
  BPCER target, and the lowest threshold meeting the APCER target. The latter is
  reported only with enough scored attacks.
- **Rule of three and Wilson intervals** for zero-error bounds and binomial
  confidence intervals.
- **Latency** for the full load + detect/embed + PAD path, plus the detection
  and PAD stages separately.
- **Resolution** counts, including how many presentations were downscaled.

## The 25 MP upload limit

`facecheck.embeddings.decode_image_bgr` rejects images whose header exceeds
`DEFAULT_MAX_PIXELS` (25,000,000 pixels) before decoding. Many modern phone
JPEGs are larger: a 32 MP photo is about 8000x4000 and exceeds the limit, so the
upload path returns `invalid_image`.

This PR does **not** change the API ingest path. It documents the issue and
makes the evaluation harness load images with OpenCV directly, optionally
downscaling with `--max-side`. A later PR should decide the client/API policy,
for example downscaling on the client before upload or accepting a larger
decode with an explicit pixel budget. Until then, a production client sending
native 32 MP photos will fail the quality gate for a reason unrelated to face
matching or PAD.

## Privacy checklist

- Consent recorded before capture; `consent=true` only for included rows.
- Pseudonymous `subject_id` and `session_id`; no names or direct identifiers.
- Capture tree outside the repository and encrypted at rest.
- Key supplied through `FACECHECK_PAD_CAPTURE_KEY`; never echoed or committed.
- GPS and unnecessary EXIF stripped from stored originals.
- Reports contain aggregates only.
- Delete plaintext working copies and the encrypted tree when the evaluation
  is complete, according to the retention policy.
- Do not use faces of bystanders, public figures, or people who did not consent
  as attack material.
