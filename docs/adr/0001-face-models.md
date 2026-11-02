# ADR 0001: Face detection and recognition models

- Status: Accepted
- Date: 2026-11-02
- Deciders: facecheck maintainers
- Related: `scripts/model_checksums.json`, `eval/run_lfw.py`

## Context

facecheck needs a face detector and a face embedding model for 1:1 verification
in an SFA attendance prototype. The service must run on commodity CPUs, ship as
a small Python service, and remain a viable candidate for commercial use. Two
model files are downloaded at setup time and never committed:

- detector: YuNet (`face_detection_yunet_2023mar.onnx`, MIT)
- recognizer: SFace (`face_recognition_sface_2021dec.onnx`, Apache-2.0 model files)

Both are loaded through OpenCV's `FaceDetectorYN` and `FaceRecognizerSF`
(`opencv-contrib-python`), so no PyTorch/ONNX Runtime dependency is required.

## Options considered

### 1. InsightFace ArcFace (e.g. buffalo_l / antelopev2)

The state of the art for open face recognition and the default choice for many
face-matching projects. Rejected for v0 because the pretrained weights are a
licensing minefield: InsightFace code is MIT, but the model zoo distribution
terms are often "non-commercial research only", and the training sets
(MS1M, Glint360K, and derivatives) have their own restrictions. Shipping those
weights in a product that may be sold to an SFA vendor is an unacceptable
unreviewed risk.

### 2. dlib / face_recognition

Simple and well known, but the ResNet model is older and less accurate than
modern embeddings, dlib builds are slower on CPU-only images, and the training
data provenance is opaque. The Python packaging story is also worse.

### 3. Cloud face APIs

Best raw accuracy, but attendance is privacy-sensitive biometric processing
under Indonesia's UU PDP. Sending face images to a third party raises consent,
retention, and cross-border transfer questions, adds per-call cost, and fails
offline check-in scenarios (Bali field offices with poor connectivity).

### 4. OpenCV Zoo: YuNet + SFace (chosen)

- YuNet 2023mar is a small fixed-shape ONNX detector supported by OpenCV 4.x,
  MIT licensed.
- SFace 2021dec produces a 128-dimensional embedding; the model files are
  Apache-2.0. `FaceRecognizerSF.alignCrop` performs the five-landmark alignment,
  and `FaceRecognizerSF.match` implements cosine matching.
- CPU inference is fast enough for an attendance endpoint, the ONNX files are
  ~39 MB total, and installation is one pip package.

## Decision

Use YuNet 2023mar for detection and SFace 2021dec for recognition via
`opencv-contrib-python`.

Implementation details:

- Detection runs at the image's native resolution; faces are rejected unless
  exactly one face is present with a shortest side of at least 80 px
  (configurable via `FACECHECK_MIN_FACE_SIZE`).
- Embeddings are L2-normalized float32 vectors of dimension 128, stored as raw
  bytes in SQLite/Postgres. No vector database is used.
- Enrollment stores 3-5 independent templates per person and verification takes
  the maximum cosine similarity against those templates. Averaging templates
  was rejected because it blurs pose/lighting variation.
- The default cosine match threshold starts at 0.363 (OpenCV's documented SFace
  cosine threshold) and was re-checked on the LFW evaluation in
  `eval/reports/lfw_report.md`. The 848 usable test pairs give 0.9945 ROC AUC;
  at 0.363 the threshold produced 0.00% FAR and 98.09% TPR. The FAR <= 1%
  operating point (0.310) only raises TPR to 98.33% while accepting four
  impostors, so 0.363 is retained as the production default. The chosen value is
  stored in `Settings.match_threshold` and reported by `GET /v1/meta`.

## Consequences

Positive:

- Permissive model file licenses and a dependency stack that is easy to audit.
- Small artifact size, CPU-only inference, no framework lock-in.
- Deterministic, reproducible evaluation against LFW.

Negative / open risks:

- SFace model files are Apache-2.0, but the **training data licensing is not
  established by that license**. The SFace paper uses large web-collected face
  corpora; commercial deployment requires legal review of the model card,
  training-data provenance, and any applicable dataset terms. Apache-2.0 on the
  weights is not a blanket commercial clearance.
- Accuracy is below current ArcFace-class models. For a v0 attendance prototype
  the trade-off is acceptable; the embedding layer is a protocol
  (`FaceEmbedder`), so a better model can be swapped in later.
- LFW is a historical, predominantly Western celebrity dataset. It says little
  about Indonesian SPG uniforms, hijab, cheap-phone cameras, or outdoor Bali
  lighting. See the README limitations and roadmap.
- Model upgrades change the embedding space. Templates record `model_version`,
  and verification refuses to compare templates from a different model version
  (HTTP 409) rather than silently producing meaningless scores.

## References and verification

- Model URLs and SHA-256 digests: `scripts/model_checksums.json`.
- Downloader: `scripts/download_models.py` (pinned digests, mirror fallback).
- OpenCV Zoo model pages:
  - https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet
  - https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface
- SFace paper: Zhong et al., "SFace: Sigmoid-Constrained Hypersphere Loss for
  Robust Face Recognition" (2021).
