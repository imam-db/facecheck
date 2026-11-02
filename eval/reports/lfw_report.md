# LFW 1:1 face verification report

Generated: 2026-11-02 06:52 UTC

- Model: `face_detection_yunet_2023mar+face_recognition_sface_2021dec`
- Dataset: LFW official test pairs (pairsDevTest.txt), 1000 pairs
- Pairs evaluated: **848** (418 same / 430 different)
- Pairs skipped by the quality gate: 152
- Skip reasons: `face_too_small`: 43, `multiple_faces`: 93, `no_face`: 16

## Headline metrics

| Metric | Value |
| --- | --- |
| ROC AUC | 0.9945 |
| EER | 1.77% |
| EER threshold | 0.2872 |
| Recommended threshold (FAR <= 1%) | **0.3100** |
| Youden J threshold | 0.3506 |

## Production threshold

The service ships with threshold **0.3630** (`FACECHECK_MATCH_THRESHOLD`). On this evaluation subset it gives TPR 98.09%, FAR 0.00%, FRR 1.91% (840/848 pairs correct).

The FAR <= 1% point at 0.3100 accepts slightly more genuine pairs but also admits impostors; for attendance, false accepts are the costlier error, so the conservative operating point is kept.

Highest impostor score: 0.3363; lowest genuine score: -0.0401. At 0.3630 the margin above the highest observed impostor is 0.0267.

## Operating points

| Threshold | TPR (FAR target) | FAR | FRR | Accuracy |
| --- | --- | --- | --- | --- |
| 0.2872 | 98.33% | 1.63% | 1.67% | 98.35% |
| 0.3000 | 98.33% | 1.16% | 1.67% | 98.58% |
| 0.3100 | 98.33% | 0.93% | 1.67% | 98.70% |
| 0.3506 | 98.09% | 0.00% | 1.91% | 99.06% |
| 0.3630 | 98.09% | 0.00% | 1.91% | 99.06% |
| 0.4000 | 97.61% | 0.00% | 2.39% | 98.82% |
| 0.4500 | 96.89% | 0.00% | 3.11% | 98.47% |
| 0.5000 | 94.02% | 0.00% | 5.98% | 97.05% |

## Score distribution

| Class | Mean | Median | Std | n |
| --- | --- | --- | --- | --- |
| same person | 0.6661 | 0.6765 | 0.1195 | 418 |
| different person | 0.0826 | 0.0775 | 0.0957 | 430 |

## CPU latency (single process, warm models)

| Measurement | Mean | Median | p95 |
| --- | --- | --- | --- |
| Detect + align + embed (per image) | 8.62 ms | 7.95 ms | 11.12 ms |
| Verification (two images + cosine) | 17.27 ms | 16.22 ms | 21.76 ms |

## Environment

- platform: Linux-6.12.94+-x86_64-with-glibc2.41
- processor: unknown
- cpu_count: 8
- python: 3.13.5
- opencv: 4.14.0
- numpy: 2.5.4

## Method notes

- Each LFW pair is treated as one enroll template and one probe; the score is cosine similarity of L2-normalized SFace embeddings.
- The production service enrolls 3-5 images per person and verifies with the maximum cosine across those templates. This benchmark is therefore conservative for real enrollment: it has no template averaging or multi-frame voting.
- Pairs where either image fails the quality gate (no face, more than one face, or shortest face side below the configured minimum) are excluded and counted.
- Threshold selection: the recommended threshold is the first threshold on the score grid with FAR <= 1% (highest TPR satisfying the false-accept budget). Youden J and EER thresholds are reported for comparison.
