# LFW 1:1 face verification report

Generated: 2026-11-02 07:54 UTC

- Model: `face_detection_yunet_2023mar@8f2383e4+face_recognition_sface_2021dec@0ba9fbfa`
- Git revision: `7634368f9b968336a1a69f7096c2b56be59f81bb`
- Dataset images: `data/lfw_home/lfw_deepfunneled`
- Threshold fitted on: `pairsDevTrain.txt`
- Metrics reported on: `pairsDevTest.txt`

## Protocol

- Train split: 1877 usable of 2200 pairs (938 same / 939 different), 323 skipped (`face_too_small` 72, `multiple_faces` 205, `no_face` 46).
- Test split: 848 usable of 1000 pairs (418 same / 430 different), 152 skipped (`face_too_small` 43, `multiple_faces` 93, `no_face` 16).
- FAR target for threshold fitting: 1% on train.
- The production default (0.363) is OpenCV's a-priori SFace cosine threshold and was **not** tuned on the test split.

## Headline test metrics

| Metric | Value |
| --- | --- |
| ROC AUC | 0.9945 |
| EER | 1.77% |
| EER threshold | 0.2872 |
| Failure to acquire (FTA) | 15.20% (152/1000 pairs) |
| Generalized FRR at 0.363 | **18.00%** (90/500 genuine pairs) |
| FAR at 0.363 | 0.00% (0/430 impostor pairs) |
| FAR 95% upper bound (rule of three) | **0.69%** |
| FAR 95% Wilson interval | 0.00% - 0.89% |
| FRR at 0.363 | 1.91% |
| FRR 95% Wilson interval | 0.97% - 3.73% |

Zero false accepts in a finite impostor sample does not mean FAR is zero. With 0 accepts in `n` trials the one-sided 95% upper bound is `1 - 0.05^(1/n)`, approximately `3/n` (the rule of three).

## Thresholds

The service ships with **0.3630** (`FACECHECK_MATCH_THRESHOLD`), OpenCV's documented SFace cosine threshold. That value is an a-priori default, not a result tuned on the test split.

Fitting on the train split for FAR <= 1% gives **0.3003** (train TPR 99.36%, train FAR 0.96%). Applied unchanged to the test split it gives TPR 98.33%, FAR 0.93%, generalized FRR 17.80%.

Best Youden J on the test split is 0.3506 (J=0.983); it is reported for reference and is not used to pick the shipped threshold.

Averaging the k enrollment templates would lower the impostor score for a lucky frame, but max-over-k gives each impostor k independent chances, so it is optimistic for FAR (FAR can grow roughly k-times) while it is conservative for FRR. The single-template LFW protocol does not measure that effect.

## Operating points (test split)

| Threshold | TPR | FAR | FRR | Accuracy |
| --- | --- | --- | --- | --- |
| 0.3000 | 98.33% | 1.16% | 1.67% | 98.58% |
| 0.3003 | 98.33% | 0.93% | 1.67% | 98.70% |
| 0.3506 | 98.09% | 0.00% | 1.91% | 99.06% |
| 0.3630 | 98.09% | 0.00% | 1.91% | 99.06% |
| 0.4000 | 97.61% | 0.00% | 2.39% | 98.82% |
| 0.4500 | 96.89% | 0.00% | 3.11% | 98.47% |
| 0.5000 | 94.02% | 0.00% | 5.98% | 97.05% |

## Score distribution (test split)

| Class | Mean | Median | Std | n |
| --- | --- | --- | --- | --- |
| same person | 0.6661 | 0.6765 | 0.1195 | 418 |
| different person | 0.0826 | 0.0775 | 0.0957 | 430 |

## CPU latency (single process, warm models)

| Measurement | Mean | Median | p95 |
| --- | --- | --- | --- |
| Detect + align + embed (LFW 250px) | 8.56 ms | 8.10 ms | 10.82 ms |
| Verification (two images + cosine) | 17.15 ms | 16.54 ms | 20.99 ms |
| JPEG decode + detect + align + embed, 1280x720 (downscaled to 960px) | 19.73 ms | 18.85 ms | 23.89 ms |
| JPEG decode + detect + align + embed, 4000x3000 (downscaled to 960px) | 73.90 ms | 71.14 ms | 91.79 ms |

## Environment

- platform: Linux-6.12.94+-x86_64-with-glibc2.41
- processor: unknown
- cpu_count: 8
- python: 3.13.5
- opencv: 4.14.0
- numpy: 2.5.4
- detect_max_side: 960
- min_face_size: 80
- min_face_frac: 0.15
- latency_benchmarks: JPEG decode + detect + align + embed for 720p and 12 MP

## Model files

- `face_detection_yunet_2023mar.onnx`: `8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4`
- `face_recognition_sface_2021dec.onnx`: `0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79`

## Method notes

- Each LFW pair is one enroll template and one probe; the score is cosine similarity of L2-normalized SFace embeddings.
- Images are downscaled so the longest side is at most 960 px before detection, alignment, and embedding. A face must be at least the configured pixel minimum and at least the configured fraction of the frame's short side.
- Pairs where either image fails the quality gate are skipped. FTA counts all skipped pairs; generalized FRR counts skipped genuine pairs as failures, so a detector that cannot acquire faces does not look good.
- The 720p and 12 MP latency rows encode the synthetic frame as JPEG and time `imdecode` plus detect + align + embed, matching the upload path.
- Confidence intervals are Wilson score intervals at 95%; the FAR bound uses the exact zero-event rule of three.
- The production service enrolls 3-5 templates and takes the maximum cosine. That is conservative for FRR but optimistic for FAR, which this single-template benchmark cannot quantify.
