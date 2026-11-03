# facecheck

Minimal 1:1 face verification service for SFA field attendance check-ins.

Give it a person's enrollment photos and a probe photo, and it answers one
question: **is this the enrolled person?** It does not try to identify who the
person is, and v0 deliberately does not attempt liveness detection, GPS
verification, or manual review.

The service is CPU-only, encrypts face templates at rest, requires recorded
consent before enrollment, and uses the permissively licensed OpenCV Zoo models
(YuNet + SFace) behind a small FastAPI API. On the LFW test split it reaches
**0.9945 ROC AUC** and **8.56 ms mean detect-and-embed latency** per 250 px
image; a 12 MP JPEG takes **73.9 ms** end to end (decode, downscale, detect,
and embed). See [evaluation](#evaluation) for the full protocol and bounds.

## Problem

SFA field attendance (SPG / promoter check-in) needs to confirm that the person
checking in is the person enrolled for that shift. Today this is often a
selfie + supervisor eyeball. facecheck reduces it to a deterministic 1:1
comparison:

1. Record consent and enroll a person once with 3-5 photos.
2. On check-in, send one probe photo.
3. Accept or reject based on a cosine threshold over face embeddings.

## Scope of v0

In scope:

- 1:1 verification (consent, enroll, verify, delete, metadata)
- YuNet detection with exactly-one-face, absolute-size, and relative-size gates
- Detection, alignment, and embedding on a downscaled frame for predictable CPU cost
- SFace 128-dimensional embeddings, L2-normalized, encrypted with AES-GCM at rest
- Multiple templates per person, maximum cosine across templates
- Pairwise enrollment consistency check (mixed faces are rejected)
- API-key authentication and localhost-bound Docker port
- SQLite by default, optional Postgres via docker-compose
- CLI comparison of two images
- LFW evaluation harness with train-fitted thresholds and test-reported metrics
- Unit tests that need no models + integration tests that need the ONNX files

Out of scope (see [roadmap](#roadmap)):

- Liveness / anti-spoofing, deepfake or replay detection
- GPS, mock-location detection, Play Integrity
- 1:N identification or duplicate-person search
- Review queue, retention automation, full consent-record management
- Mobile SDK / client app
- Role-based access control or rate limiting

## Architecture

```mermaid
flowchart LR
    client[SFA client] -->|X-API-Key + multipart| api[FastAPI /v1]
    api --> service[FaceVerificationService]
    service --> embedder[YuNet + SFace embedder]
    embedder -->|128-d float32, L2-normalized| service
    service --> storage[(Storage: persons + encrypted templates)]
    storage -->|AES-GCM decrypt on read| service
    service -->|score, decision, threshold| api
    api -->|JSON| client
```

The embedder sits behind a `FaceEmbedder` protocol
(`detect_and_embed(bgr) -> EmbeddingResult`), so tests use a deterministic fake
and the production implementation is replaceable without touching the API or
storage layers.

### Design notes

- **Templates are not averaged.** Enrolling 3-5 photos stores 3-5 templates;
  verification takes the maximum cosine similarity against any of them.
  Averaging would blur pose and lighting variation that multiple templates are
  meant to capture. Max-over-k is conservative for FRR but **optimistic for
  FAR**: an impostor gets k chances, so the multi-template FAR can be higher
  than the single-template rate measured on LFW.
- **Enrollment is atomic and consistent.** If any submitted image fails the
  quality gate, the request returns HTTP 400 with a per-image reason and no
  templates are written. After all images embed successfully, every pairwise
  cosine among the enrollment embeddings must meet the enrollment threshold
  (`FACECHECK_ENROLL_THRESHOLD`, defaulting to the match threshold). Mixed faces
  return `inconsistent_enrollment` and leave any previous templates untouched.
- **Consent comes first.** A person must have `consent_version` and `consent_at`
  recorded before enrollment; otherwise enrollment returns HTTP 409.
- **Templates are encrypted at the storage choke point.** AES-GCM uses a
  32-byte key from `FACECHECK_TEMPLATE_KEY`, a fresh 12-byte nonce per row, and
  associated data binding the ciphertext to `person_id|template_id`. The key is
  required at startup: there is no plaintext mode.
- **Model versions are enforced and fingerprinted.** Each template records a
  `model_version` that includes an 8-character SHA-256 prefix of both ONNX
  files. Verification compares only templates from the current model and
  returns HTTP 409 if none match. A model change therefore forces re-enrollment
  and threshold review.
- **Strict quality gate.** Zero faces, more than one face, a face under the
  absolute pixel minimum, or a face under `FACECHECK_MIN_FACE_FRAC` of the
  frame's short side is rejected. In a group photo the user must retake, which
  is intentional for a 1:1 attendance check.
- **Pixel guard before decode.** Uploaded bytes are capped, and the image
  container header is checked against `FACECHECK_MAX_PIXELS` before the bitmap
  is decoded, so a small compressed file cannot expand into an unbounded
  allocation.
- **No vector database.** A person has at most five 128-float templates, so
  brute-force cosine similarity is faster than any index and keeps deployment
  trivial.

## Quickstart

Requires Python 3.11+ (tested on 3.13) and no GPU.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[dev]"

# Downloads ~39 MB of ONNX models into models/, verifying pinned SHA-256 digests
python scripts/download_models.py

# Both keys are required at startup: the service fails closed.
export FACECHECK_API_KEY=$(
  python -c "import secrets; print(secrets.token_urlsafe(32))"
)
export FACECHECK_TEMPLATE_KEY=$(
  python -c "import base64,os; print(base64.urlsafe_b64encode(os.urandom(32)).decode())"
)

uvicorn facecheck.api:app --reload --port 8000
```

Run the test suite:

```bash
pytest -v
```

Unit tests run everywhere. Tests marked `model` and `minifasnet` require the
downloaded ONNX files; when the files are missing locally they skip, but CI
downloads and verifies all four required models and fails loudly if that
download fails. Semantic golden tests additionally require
`FACECHECK_PAD_GOLDEN_DIR` (images kept outside the repo) and skip when it is
unset. The key requires-deletion and missing-model tests always run.

### Enroll and verify with curl

```bash
API=http://127.0.0.1:8000
KEY="$FACECHECK_API_KEY"

# 1. Create a person with a consent record
PERSON_ID=$(curl -s -X POST "$API/v1/persons" \
  -H "X-API-Key: $KEY" -H 'Content-Type: application/json' \
  -d '{"external_id": "spg-001",
       "consent_version": "2026-11",
       "consent_at": "2026-11-01T09:00:00Z"}' \
  | python -c 'import json,sys; print(json.load(sys.stdin)["id"])')

# 2. Enroll 3-5 photos in one multipart request
curl -s -X POST "$API/v1/persons/$PERSON_ID/enroll" \
  -H "X-API-Key: $KEY" \
  -F files=@enroll-1.jpg \
  -F files=@enroll-2.jpg \
  -F files=@enroll-3.jpg
# {"enrolled":3,"model_version":"face_detection_yunet_2023mar@8f2383e4+..."}

# 3. Verify a probe photo
curl -s -X POST "$API/v1/persons/$PERSON_ID/verify" \
  -H "X-API-Key: $KEY" \
  -F file=@probe.jpg
# {"score":0.7123,"decision":"match","threshold":0.363,"model_version":"..."}

# 4. Metadata / delete
curl -s -H "X-API-Key: $KEY" "$API/v1/meta"
curl -s -X DELETE "$API/v1/persons/$PERSON_ID" -H "X-API-Key: $KEY" \
  -o /dev/null -w '%{http_code}\n'
```

A quality rejection looks like:

```json
{"code": "invalid_enrollment",
 "message": "one or more images failed the face quality gate",
 "details": {"reasons": [
   {"index": 1, "reason": "multiple_faces",
    "message": "expected exactly one face, detected 2"}]}}
```

Reason codes: `no_face`, `multiple_faces`, `face_too_small`, `invalid_image`,
`inconsistent_enrollment`, `consent_required`.

### CLI

```bash
python -m facecheck compare photos/a.jpg photos/b.jpg
# score=0.7123 decision=match threshold=0.3630 model=face_detection_yunet_2023mar@8f2383e4+...
```

Exit code `0` means match, `1` means no match, `2` means an input/model error.
Use `--threshold` and `--models-dir` to override the defaults.

### API summary

| Method | Path | Auth | Body | Response |
| --- | --- | --- | --- | --- |
| `POST` | `/v1/persons` | API key | JSON `{external_id?, consent_version, consent_at}` | `201 {id, external_id, consent_version, consent_at, created_at}` |
| `POST` | `/v1/persons/{id}/enroll` | API key | multipart `files` (3-5 images) | `{enrolled, model_version}` |
| `POST` | `/v1/persons/{id}/verify` | API key | multipart `file` | `{score, decision, threshold, model_version}` |
| `DELETE` | `/v1/persons/{id}` | API key | - | `204` |
| `GET` | `/v1/meta` | API key | - | `{model_version, threshold, enroll_threshold, embedding_dim, min_face_size, min_face_frac, detect_max_side}` |
| `GET` | `/healthz` | open | - | `{status}` |

Errors use one envelope: `{"code": "...", "message": "...", "details": ...}`.
Request validation failures return HTTP 422 with `code=validation_error`; a
template that cannot be decrypted (wrong or rotated key, or a pre-encryption
row) returns HTTP 409 with `code=template_decrypt_failed` and asks for
re-enrollment. Quality errors put the reason under `details.reason`; enrollment
errors put per-image rejections under `details.reasons`. OpenAPI docs are at
`/docs`.

### Configuration

All settings use the `FACECHECK_` prefix and can come from environment
variables or a local `.env` (see `.env.example`).

| Variable | Default | Meaning |
| --- | --- | --- |
| `FACECHECK_DB_URL` | `sqlite:///./data/facecheck.db` | SQLAlchemy URL; use `postgresql+psycopg://...` for Postgres |
| `FACECHECK_MODELS_DIR` | `models` | Directory containing the ONNX files |
| `FACECHECK_MATCH_THRESHOLD` | `0.363` | Cosine similarity threshold for verify |
| `FACECHECK_ENROLL_THRESHOLD` | match threshold | Minimum pairwise cosine among enrollment templates |
| `FACECHECK_MIN_FACE_SIZE` | `80` | Minimum shortest side of the detected face box, in detection pixels |
| `FACECHECK_MIN_FACE_FRAC` | `0.15` | Minimum face short side as a fraction of the frame's short side |
| `FACECHECK_DETECTOR_SCORE_THRESHOLD` | `0.9` | YuNet detection confidence |
| `FACECHECK_DETECT_MAX_SIDE` | `960` | Downscale the longest side to this before detection; `0` disables |
| `FACECHECK_MAX_PIXELS` | `25000000` | Reject larger images from the header before decoding |
| `FACECHECK_CV_THREADS` | unset | Cap OpenCV intra-op threads (`cv2.setNumThreads`); unset keeps OpenCV's default. Useful under multiple uvicorn workers to avoid CPU oversubscription |
| `FACECHECK_ENROLL_MIN_IMAGES` | `3` | Minimum images per enrollment |
| `FACECHECK_ENROLL_MAX_IMAGES` | `5` | Maximum images per enrollment |
| `FACECHECK_MAX_UPLOAD_BYTES` | `10485760` | Per-image upload limit |
| `FACECHECK_API_KEY` | required | Required as `X-API-Key` or `Authorization: Bearer` on every route except `/healthz` |
| `FACECHECK_TEMPLATE_KEY` | required | AES-256 key as urlsafe base64 or 64 hex characters |
| `FACECHECK_DEV_MODE` | `false` | Allow the public placeholder keys for local use; keys must still be set |

The service **fails closed**. Startup is refused unless both `FACECHECK_API_KEY`
and `FACECHECK_TEMPLATE_KEY` are non-empty. When `FACECHECK_DEV_MODE=false`,
the committed placeholder values (`dev-api-key` and the example template key)
are also refused. `FACECHECK_DEV_MODE=true` relaxes only the placeholder ban,
not the requirement that the keys exist. Never enable it for a shared instance.

## Privacy and security

### Threat model (v0)

| Threat | Control | Residual risk |
| --- | --- | --- |
| Unauthenticated API access | `FACECHECK_API_KEY`, constant-time comparison, loopback-bound compose port | No RBAC or rate limiting; a leaked shared key is powerful |
| Stolen database file | AES-GCM encrypted templates, key kept outside the DB | Person metadata and consent fields are not encrypted; key compromise defeats this |
| Template swap between people | AES-GCM associated data binds ciphertext to `person_id` and `template_id` | Key compromise still permits decryption |
| Recovering deleted templates | Cascading delete + SQLite `PRAGMA secure_delete=ON` | Backups, snapshots, and Postgres storage need their own retention/erasure process |
| Enrollment without consent | `consent_version` + `consent_at` required before templates are written | Consent record itself is minimal; no withdrawal workflow yet |
| Spoofed probe (photo, screen, replay) | None in v0 | Face matching alone cannot stop presentation attacks |
| Network interception | Loopback binding in compose; TLS expected at a reverse proxy | The app does not terminate TLS itself |
| Model tampering | Downloader verifies pinned SHA-256; model hashes appear in `model_version` | Runtime does not re-verify files on every start |

### What is stored

Stored:

- person id, optional external id, consent version, consent timestamp,
  creation timestamp
- for each template: encrypted 128-float embedding, dimension, model version,
  template id, creation timestamp

Not stored:

- raw images
- plaintext embeddings
- display names

Template ciphertext is `b"FCK1" + nonce + AES-GCM ciphertext`. The nonce is
random per template, and the associated data prevents moving a ciphertext to a
different person or template row. Decryption happens only when a person's
templates are loaded for verification.

### Deletion and retention

`DELETE /v1/persons/{id}` removes the person and all templates. On SQLite the
connection enables `PRAGMA secure_delete=ON`, so deleted pages are overwritten
rather than left in free space. This does not erase copies in filesystem
snapshots, database backups, or Postgres WAL/backups; production deployments
need a retention and backup-erasure policy. v0 has no automatic retention
sweep.

Encryption keys are supplied through the environment. Key rotation is not
automated: rotating `FACECHECK_TEMPLATE_KEY` makes existing templates
undecryptable, so a rotation procedure currently means re-enrolling people.

Face templates are biometric data. Encryption reduces exposure from a stolen
database file; it does not make the data non-biometric or remove obligations
under Indonesia's UU PDP or similar regimes.

## Docker

```bash
# compose has no key defaults and refuses to render without both
export FACECHECK_API_KEY="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
export FACECHECK_TEMPLATE_KEY="$(
  python -c 'import base64,os; print(base64.urlsafe_b64encode(os.urandom(32)).decode())'
)"
docker compose up --build
# API on http://127.0.0.1:8000, SQLite volume mounted at /data
```

The compose file binds the port to loopback and requires both keys from the
environment (`${FACECHECK_API_KEY:?…}` and `${FACECHECK_TEMPLATE_KEY:?…}`); it
has no placeholder defaults. The image installs
`opencv-contrib-python-headless`, the `[postgres]` extra (so `psycopg` is
available), downloads and verifies the models at build time, and runs as a
non-root user.

Databases created before the consent columns were added are **not migrated
automatically**. Startup fails with an incompatibility message if `persons`
still has `display_name` or is missing `consent_version` / `consent_at`. Delete
the old SQLite file or Docker volume and recreate it:

```bash
docker compose down -v   # removes the facecheck-data volume
docker compose up --build
```

Re-enrollment is required because templates from the old schema cannot be
trusted.

```bash
# Optional Postgres profile; also point FACECHECK_DB_URL at the server
docker compose --profile postgres up --build
# FACECHECK_DB_URL=postgresql+psycopg://facecheck:facecheck@postgres:5432/facecheck
```

## Evaluation

The benchmark uses the official LFW pairs. The operating threshold is fitted
on **`pairsDevTrain.txt`** (2200 pairs) and every reported metric is measured
on **`pairsDevTest.txt`** (1000 pairs), keeping the test split out of threshold
selection.

```bash
python scripts/fetch_lfw.py        # train + test pairs and deep-funneled images
python -m eval.run_lfw             # writes eval/reports/lfw_report.md + metrics JSON
python -m eval.run_lfw --limit 100 # balanced quick subset of each split
```

**Results** (2026-11-02, 8 vCPU, Python 3.13.5, OpenCV 4.14.0, CPU inference):

| Metric | Value |
| --- | --- |
| Test pairs evaluated | **848 / 1000** (418 same, 430 different) |
| Test pairs skipped by quality gate | 152 (`multiple_faces` 93, `face_too_small` 43, `no_face` 16) |
| Failure to acquire (FTA) | **15.20%** (152 / 1000 pairs) |
| ROC AUC | **0.9945** |
| EER | **1.77%** at threshold 0.2872 |
| Shipped threshold | **0.363** (OpenCV a-priori default, not tuned on test) |
| TPR at 0.363 | 98.09% |
| FAR at 0.363 | 0.00% (0 / 430) |
| FAR 95% upper bound, rule of three | **0.69%** |
| FAR 95% Wilson interval | 0.00% - 0.89% |
| Ordinary FRR at 0.363 | 1.91% (95% CI 0.97% - 3.73%) |
| Generalized FRR at 0.363 | **18.00%** (90 / 500 genuine pairs attempted) |
| Generalized FRR 95% Wilson interval | 14.88% - 21.61% |
| Train-fitted threshold at FAR <= 1% | **0.3003** |
| Test metrics at 0.3003 | TPR 98.33%, FAR 0.93% (4 / 430), generalized FRR 17.80% |
| Detect + align + embed, 250 px | 8.56 ms mean, 8.10 ms median, 10.82 ms p95 |
| Verification, two 250 px images + cosine | 17.15 ms mean, 16.54 ms median, 20.99 ms p95 |
| JPEG decode + detect + align + embed, 1280x720 | 19.73 ms mean, 18.85 ms median, 23.89 ms p95 |
| JPEG decode + detect + align + embed, 4000x3000 (12 MP) | 73.90 ms mean, 71.14 ms median, 91.79 ms p95 |

**Statistical honesty.** Zero false accepts in 430 impostor pairs does not mean
FAR is zero. With 0 accepts in `n` trials the one-sided 95% upper bound is
`1 - 0.05^(1/n)`, approximately `3/n` (the rule of three); for `n=430` that is
**0.69%**. Ordinary FRR is 8 false rejects out of 418 usable genuine pairs,
with a Wilson 95% interval of 0.97%-3.73%. Generalized FRR counts the 82
skipped genuine pairs as failures too: `(82 + 8) / 500 = 18.00%`, with interval
14.88%-21.61%. The gap between 1.91% and 18.00% is the cost of a strict
single-face quality gate, and it is the more useful number for attendance
operations.

**Threshold policy.** The shipped 0.363 is OpenCV's documented SFace cosine
threshold. It was chosen a priori and was not tuned on the test split. Fitting
on the train split for FAR <= 1% gives 0.3003; applied unchanged to the test
split it raises TPR from 98.09% to 98.33% while accepting four impostors
(FAR 0.93%). For attendance, a false accept is the more expensive error, so the
stricter 0.363 remains the default. The recommended value and its bounds are
reported so operators can choose a different operating point with eyes open.

**Max-over-k trade-off.** Production enrolls 3-5 templates and takes the
maximum cosine. That is conservative for FRR (a genuine probe needs only one
good template) but optimistic for FAR: an impostor gets k chances, and
correlated errors can make multi-template FAR grow roughly k-times relative to
the single-template estimate. LFW's one-template-per-person protocol does not
measure that effect.

The full report, including operating points, score distributions, and the
model/git revisions, is committed at
[`eval/reports/lfw_report.md`](eval/reports/lfw_report.md), with the ROC and
DET curves as PNGs next to it.

### Evaluation caveats

- **152 test pairs were excluded** by the production quality gate. This is not
  the full LFW protocol: excluded images include group photos and small faces,
  and skipping them may flatter AUC relative to a detector-free benchmark.
  Generalized FRR and FTA are reported precisely so this exclusion is visible.
- **Images are deep-funneled LFW**, an aligned variant, fetched from an
  archive.org mirror because the canonical figshare/UMass downloads returned
  403/500 during development. The pairs-file checksums match the copies shipped
  by scikit-learn; the image archive checksum is pinned in `scripts/fetch_lfw.py`.
- **Single image per side.** Production enrollment uses 3-5 templates and takes
  the max cosine. This benchmark therefore does not represent enrolled users or
  the multi-template FAR inflation described above.
- **LFW is not the target population.** See limitations.

## Limitations

- **LFW does not represent SFA users.** It is a historical celebrity dataset,
  mostly Western, mostly well-lit, mostly unoccluded. Accuracy on Indonesian
  SPG users, hijab styles, cheap-phone front cameras, compressed WhatsApp
  selfies, and outdoor Bali lighting is unmeasured and will be lower. Treat the
  numbers above as a sanity check, not a deployment SLA.
- **No liveness.** A printed photo, a screen replay, or a video call can spoof a
  face matcher. Do not use this alone for high-value attendance or payroll.
- **Thresholds are dataset-specific.** 0.363 produced zero false accepts on 430
  LFW negative pairs, but the 95% upper bound is 0.69%. Recalibrate on a
  consented, representative internal set before rollout, and re-tune whenever
  `model_version` changes.
- **Quality gate drives FTA.** The strict one-face gate skipped 15.2% of test
  pairs. If field users cannot produce a clean single-face frame, operational
  FTA will dominate the error budget even when matching accuracy is good.
- **Biometric data obligations remain.** Face templates fall under Indonesia's
  UU PDP (and similar regimes). Encryption, consent capture, and deletion
  support are necessary but not sufficient; a lawful basis, retention limits,
  access control, and a breach process are still required.
- **No rate limiting or RBAC** in v0. A shared API key authenticates callers but
  does not distinguish them.
- **Model updates invalidate templates.** Changing the ONNX models changes
  `model_version` and requires re-enrollment; the API surfaces this as HTTP 409.

## Roadmap

1. **Liveness** — passive anti-spoofing model or challenge-response blink/turn,
   evaluated on spoof vs. genuine samples from the actual device fleet.
2. **Device and location integrity** — Play Integrity / App Attest attestation,
   mock-location detection, optional geofence around the store.
3. **Review queue** — borderline scores routed to a supervisor with an audit
   trail instead of hard accept/reject.
4. **UU PDP operations** — full consent records and withdrawal, purpose
   limitation, retention automation, key rotation, audit logs, DPIA.
5. **Target-domain evaluation** — consented internal dataset covering SPG
   uniforms, hijab, indoor/outdoor Bali light, low-end Android cameras.
6. **Threshold recalibration** — per-device or per-cohort thresholds with
   FAR/FRR targets agreed with operations, keyed by `model_version`.
7. **Better embedder** — swap SFace through the `FaceEmbedder` protocol once a
   commercially licensed model passes legal review.

## Project layout

```text
src/facecheck/
  config.py       settings (pydantic-settings, FACECHECK_* env)
  embeddings.py   FaceEmbedder protocol + YuNet/SFace implementation
  quality.py      exactly-one-face, absolute/relative size gate
  similarity.py   cosine similarity, best-of-templates, decision
  liveness/       PAD runtime: protocol, CropImage port, box mapper, MiniFASNet
  storage.py      SQLAlchemy models + AES-GCM template encryption
  service.py      consent, enroll / verify / delete use cases
  api.py          FastAPI app, API-key middleware, error envelope
  cli.py          `python -m facecheck compare`
scripts/
  download_models.py    pinned model download + SHA-256 verification
  fetch_lfw.py          LFW pairs + images for evaluation
  export_minifasnet_onnx.py  offline MiniFASNet -> ONNX export + parity
  pad_golden_report.py  runtime MiniFASNet scores for the golden set
  _minifasnet_arch.py   vendored MiniFASNet architecture (export only)
  _crop_image.py        verbatim upstream CropImage port for PAD crops
eval/
  metrics.py            FAR/FRR helpers, rule of three, Wilson intervals
  run_lfw.py            train-fit / test-report LFW benchmark
  reports/              committed markdown/JSON + ROC/DET PNGs
tests/                  unit tests + model-gated integration tests
  fixtures/minifasnet_parity.json  replayable PyTorch/ONNX parity reference
docs/adr/0001-face-models.md
docs/adr/0002-presentation-attack-detection.md
docs/pad_golden_manifest.md  out-of-repo semantic golden schema
```

## Models and licensing

Source code is MIT (see [LICENSE](LICENSE)).

| Artifact | License | Notes |
| --- | --- | --- |
| YuNet `face_detection_yunet_2023mar.onnx` | MIT | OpenCV Zoo detector |
| SFace `face_recognition_sface_2021dec.onnx` | Apache-2.0 (model files) | OpenCV Zoo recognizer |
| MiniFASNet `2.7_80x80_MiniFASNetV2.onnx` | Apache-2.0 (model files) | Silent-Face anti-spoofing, required PAD artifact (not wired into v0) |
| MiniFASNet `4_0_0_80x80_MiniFASNetV1SE.onnx` | Apache-2.0 (model files) | Silent-Face anti-spoofing, required PAD artifact (not wired into v0) |
| LFW dataset | Research use | Used only for evaluation; never redistributed |

The ONNX weights are **not committed**. `scripts/download_models.py` fetches
YuNet and SFace from OpenCV Zoo (GitHub or Hugging Face mirror) and MiniFASNet
from the `models-minifasnet-v1` GitHub release, verifying pinned SHA-256 digests
and byte sizes in all cases. All four entries are required: a failed download
aborts the script, and CI fails rather than silently skipping model-backed
tests. See `scripts/model_checksums.json` for URLs, hashes, and the upstream
commit and `.pth` digests recorded on each MiniFASNet entry. The short digest
prefixes appear in every reported `model_version`; the full hashes are recorded
in the evaluation JSON.

Important: SFace's Apache-2.0 model-file license does **not** establish that its
training data is cleared for commercial use. The SFace paper trains on large
web-collected face corpora. A commercial deployment needs legal review of the
model card and training-data provenance before shipping. This is why
InsightFace's ArcFace weights, whose distribution terms and training sets carry
similar or stricter restrictions, were rejected for v0. The full rationale is
in [`docs/adr/0001-face-models.md`](docs/adr/0001-face-models.md). The same
caveat applies to the MiniFASNet weights below.

### MiniFASNet ONNX export (PAD prep)

v0 does **not** run liveness detection, but both anti-spoofing artifacts are
part of the required model set: `scripts/download_models.py` fetches and
verifies them like YuNet and SFace, and CI fails if they are missing. The
weights come from
[minivision-ai/Silent-Face-Anti-Spoofing](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing)
pinned to commit `b6d5f04ad78778917853b25c778acef6d5626d15` (Apache-2.0,
Copyright 2020 Minivision) and are exported to fixed-shape ONNX:

| ONNX file | Source checkpoint | Input | Output |
| --- | --- | --- | --- |
| `2.7_80x80_MiniFASNetV2.onnx` | `2.7_80x80_MiniFASNetV2.pth` | `1x3x80x80` float32 | `1x3` logits |
| `4_0_0_80x80_MiniFASNetV1SE.onnx` | `4_0_0_80x80_MiniFASNetV1SE.pth` | `1x3x80x80` float32 | `1x3` logits |

Index `1` of the 3 logits is the real / bona-fide class. The models expect
**raw BGR floats in [0, 255]**: upstream commented out `div(255)`, so do not
divide by 255 and do not apply mean/std normalization to the 80x80 crop.
Dividing by 255 marks every face as an attack, and swapping BGR/RGB is
fail-open (attacks pass as real).

**Downloading (required).**

```bash
python scripts/download_models.py
# fetches YuNet, SFace, and both MiniFASNet models; aborts if any fails
```

All four manifest entries are required. The downloader streams to a `.part`
file, checks SHA-256 and byte size before renaming, refuses oversized
responses, and exits nonzero when a required artifact cannot be fetched. The
ONNX binaries are not committed; they are published as `models-minifasnet-v1`
release assets, and that release asset SHA-256 is authoritative because
re-exports with a different torch version may differ in bytes.

**Exporting (maintainers).** PyTorch, ONNX, and ONNX Runtime are export-only
dependencies (`[export]`), never runtime dependencies:

```bash
python -m pip install -e ".[export]"
git clone https://github.com/minivision-ai/Silent-Face-Anti-Spoofing \
  /tmp/Silent-Face-Anti-Spoofing
git -C /tmp/Silent-Face-Anti-Spoofing checkout \
  b6d5f04ad78778917853b25c778acef6d5626d15
python scripts/export_minifasnet_onnx.py \
  --silent-face-root /tmp/Silent-Face-Anti-Spoofing
# -> models/2.7_80x80_MiniFASNetV2.onnx
# -> models/4_0_0_80x80_MiniFASNetV1SE.onnx
```

`--silent-face-root` falls back to `$SILENT_FACE_ROOT` (or
`$FACECHECK_SILENT_FACE_ROOT`). The script verifies each `.pth` against the
pinned SHA-256 before loading, asserts PyTorch vs ONNX Runtime parity below
`1e-4` on seeded synthetic inputs, and writes
`tests/fixtures/minifasnet_parity.json` only when every model passes. The
fixture records the upstream commit, per-model `.pth` digests, and the
torch/onnx/onnxruntime versions. The architecture is vendored in
`scripts/_minifasnet_arch.py`; only the `.pth` checkpoints are read from the
clone. Attribution is in [`NOTICE`](NOTICE) and
[`third_party/Silent-Face-Anti-Spoofing.LICENSE`](third_party/Silent-Face-Anti-Spoofing.LICENSE).

**Semantic goldens.** The crop/preprocessing path is checked against real
photographs kept outside the repository:

```bash
export FACECHECK_PAD_GOLDEN_DIR=/path/to/pad-golden
python scripts/export_minifasnet_onnx.py \
  --silent-face-root /tmp/Silent-Face-Anti-Spoofing \
  --golden-dir "$FACECHECK_PAD_GOLDEN_DIR" --update-golden
pytest -m minifasnet

pytest   # with the variable set, the golden tests also run
```

The directory holds a `manifest.json` (schema:
[`docs/pad_golden_manifest.md`](docs/pad_golden_manifest.md)) and images.
Maintainers supply consented real/print/screen photos and no names; images are
never committed or shipped. The tests reproduce the upstream `CropImage` port
from `scripts/_crop_image.py`, require the ONNX files, and skip cleanly when
`FACECHECK_PAD_GOLDEN_DIR` is unset or missing. Background and open risks are
recorded in
[`docs/adr/0002-presentation-attack-detection.md`](docs/adr/0002-presentation-attack-detection.md).

The runtime building blocks now live under `src/facecheck/liveness/`: a
`LivenessDetector` protocol and `LivenessResult`, a runtime `CropImage` port
(kept bit-identical to the scripts reference), a detection-to-full-res box
mapper, and `MiniFASNetLivenessDetector`. The detector scores a full-resolution
image plus a face box and reports the averaged real probability. It applies no
threshold and is not wired into enroll or verify; full PAD integration remains
Proposed.

## Development

```bash
make install   # venv + editable install with dev/eval dependencies
make models    # download/verify ONNX models
make test      # pytest
make lint      # ruff
make eval      # fetch LFW + run the benchmark
make serve     # uvicorn with reload
```

CI runs on GitHub Actions for Python 3.11 and 3.12: install, lint, download and
verify models (cached), then `pytest`. The model download step is required; if
it fails, the job fails rather than silently skipping the model-backed tests.
