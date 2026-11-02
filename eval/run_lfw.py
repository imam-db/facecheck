"""LFW 1:1 verification benchmark for the YuNet + SFace pipeline.

The protocol is split fitting: the operating threshold is selected on
``pairsDevTrain.txt`` (1100 same / 1100 different pairs) and every reported
metric is measured on ``pairsDevTest.txt`` (500 / 500). This keeps the test
split out of threshold selection.

For each pair both images go through detect + align + embed and are compared
with cosine similarity. The report includes ROC AUC, EER, FTA, ordinary and
generalized FRR, a rule-of-three upper bound on FAR, Wilson confidence
intervals, CPU latency, and the git/model revisions used.

Expected layout (created by ``scripts/fetch_lfw.py``)::

    data/lfw_home/pairsDevTrain.txt
    data/lfw_home/pairsDevTest.txt
    data/lfw_home/lfw_deepfunneled/<name>/<name>_0001.jpg

Usage::

    python -m eval.run_lfw                # train tune + test report
    python -m eval.run_lfw --limit 200    # quick subset of each split
"""

from __future__ import annotations

import argparse
import contextlib
import json
import platform
import subprocess
import sys
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np

from eval.metrics import (
    Metrics,
    choose_eer,
    choose_far_threshold,
    choose_youden,
    failure_to_acquire,
    generalized_frr,
    roc_auc,
    roc_curve,
    rule_of_three,
    scores_at,
    wilson_interval,
)
from facecheck.config import Settings
from facecheck.embeddings import (
    OpenCVYuNetSFaceEmbedder,
    decode_image_bgr,
    file_sha256,
)
from facecheck.quality import QualityError
from facecheck.similarity import cosine_similarity

REPO_ROOT = Path(__file__).resolve().parents[1]

THRESHOLD_GRID = (0.30, 0.363, 0.40, 0.45, 0.50)
FAR_TARGET = 0.01


@dataclass
class Pair:
    image_a: np.ndarray
    image_b: np.ndarray
    label: int
    pair_id: int


@dataclass
class SplitResult:
    source: str
    pairs_total: int
    pairs_skipped: int
    skipped_reasons: dict[str, int]
    skipped_genuine: int
    skipped_impostor: int
    scores: np.ndarray
    labels: np.ndarray
    embed_times: list[float]
    pair_times: list[float]

    @property
    def pairs_used(self) -> int:
        return int(self.scores.size)

    @property
    def positives(self) -> int:
        return int(np.sum(self.labels == 1))

    @property
    def negatives(self) -> int:
        return int(np.sum(self.labels == 0))


@dataclass
class SplitReport:
    source: str
    pairs_total: int
    pairs_used: int
    pairs_skipped: int
    skipped_reasons: dict[str, int]
    skipped_genuine: int
    skipped_impostor: int
    positives: int
    negatives: int
    fta: float | None
    roc_auc: float | None
    eer: float | None
    eer_threshold: float | None


@dataclass
class EvalReport:
    generated_at: str
    git_sha: str
    git_dirty: bool
    model_version: str
    model_sha256: dict[str, str]
    images_dir: str
    train_pairs_file: str
    test_pairs_file: str
    far_target: float
    default_threshold: float
    train: SplitReport
    test: SplitReport
    recommended_threshold: float
    recommended_threshold_basis: str
    recommended_threshold_train_metrics: Metrics
    youden_threshold: float
    youden_index: float
    production_metrics: Metrics
    production_far_upper_95_rule_of_three: float | None
    production_far_ci_95_wilson: tuple[float, float] | None
    production_frr_ci_95_wilson: tuple[float, float] | None
    production_generalized_frr: float | None
    production_generalized_frr_ci_95_wilson: tuple[float, float] | None
    recommended_metrics: Metrics
    recommended_far_upper_95_rule_of_three: float | None
    recommended_far_ci_95_wilson: tuple[float, float] | None
    recommended_frr_ci_95_wilson: tuple[float, float] | None
    recommended_generalized_frr: float | None
    recommended_generalized_frr_ci_95_wilson: tuple[float, float] | None
    operating_points: list[Metrics] = field(default_factory=list)
    score_stats: dict[str, dict[str, float]] = field(default_factory=dict)
    latency_ms: dict[str, dict[str, float]] = field(default_factory=dict)
    environment: dict[str, object] = field(default_factory=dict)


def _resolve_dataset_paths(
    images_dir: Path | None,
    pairs_file: Path | None,
    train_pairs_file: Path | None,
) -> tuple[Path, Path, Path]:
    if images_dir is None:
        for candidate in (
            REPO_ROOT / "data/lfw_home/lfw_deepfunneled",
            REPO_ROOT / "data/lfw_home/lfw_funneled",
            REPO_ROOT / "data/lfw_home/lfw",
        ):
            if candidate.is_dir():
                images_dir = candidate
                break
    if images_dir is None:
        raise SystemExit(
            "no local LFW image tree found. Run `python scripts/fetch_lfw.py` "
            "to download the checksum-pinned LFW layout."
        )

    if pairs_file is None:
        pairs_file = REPO_ROOT / "data/lfw_home/pairsDevTest.txt"
    if train_pairs_file is None:
        train_pairs_file = REPO_ROOT / "data/lfw_home/pairsDevTrain.txt"

    missing = [path for path in (pairs_file, train_pairs_file) if not path.is_file()]
    if missing:
        raise SystemExit(
            "missing LFW pairs file(s): "
            + ", ".join(str(path) for path in missing)
            + ".\nRun `python scripts/fetch_lfw.py`; threshold fitting uses the "
            "train pairs and reporting uses the test pairs."
        )
    return images_dir, pairs_file, train_pairs_file


def _read_pairs_file(pairs_file: Path, images_dir: Path) -> list[tuple[Path, Path, int]]:
    lines = [line.rstrip("\n") for line in pairs_file.read_text(encoding="utf-8").splitlines()]
    lines = [line for line in lines if line.strip()]
    declared = int(lines[0].strip())
    matched_lines = lines[1 : 1 + declared]
    mismatched_lines = lines[1 + declared : 1 + 2 * declared]
    if len(matched_lines) != declared or len(mismatched_lines) != declared:
        raise ValueError(
            f"{pairs_file} declares {declared} pairs per class but contains "
            f"{len(matched_lines)} matched / {len(mismatched_lines)} mismatched"
        )

    def image_path(name: str, index: str) -> Path:
        return images_dir / name / f"{name}_{int(index):04d}.jpg"

    pairs: list[tuple[Path, Path, int]] = []
    for line in matched_lines:
        name, first, second = line.split("\t")
        pairs.append((image_path(name, first), image_path(name, second), 1))
    for line in mismatched_lines:
        first_name, first, second_name, second = line.split("\t")
        pairs.append((image_path(first_name, first), image_path(second_name, second), 0))
    return pairs


def display_path(path: Path) -> str:
    """Render a path relative to the repository when possible.

    Committed reports should not embed one machine's absolute layout.
    """

    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.name


def iter_pairs(
    images_dir: Path,
    pairs_file: Path,
    limit: int,
) -> tuple[Iterator[Pair], str]:
    records = _read_pairs_file(pairs_file, images_dir)
    if limit > 0 and limit < len(records):
        # Keep the subset balanced: matched pairs are listed first, so naive
        # truncation would produce an all-positive set and an undefined AUC.
        matched = [record for record in records if record[2] == 1]
        mismatched = [record for record in records if record[2] == 0]
        n_matched = min(len(matched), (limit + 1) // 2)
        n_mismatched = min(len(mismatched), limit // 2)
        records = matched[:n_matched] + mismatched[:n_mismatched]

    def from_files() -> Iterator[Pair]:
        for pair_id, (path_a, path_b, label) in enumerate(records):
            image_a = cv2.imread(str(path_a), cv2.IMREAD_COLOR)
            image_b = cv2.imread(str(path_b), cv2.IMREAD_COLOR)
            if image_a is None:
                raise SystemExit(f"could not read {path_a}")
            if image_b is None:
                raise SystemExit(f"could not read {path_b}")
            yield Pair(image_a=image_a, image_b=image_b, label=label, pair_id=pair_id)

    source = f"LFW {pairs_file.name} ({len(records)} pairs)"
    return from_files(), source


def _evaluate(
    embedder: OpenCVYuNetSFaceEmbedder,
    pairs: Iterator[Pair],
    source: str,
    *,
    warmup: list[bool],
) -> SplitResult:
    scores: list[float] = []
    labels: list[int] = []
    embed_times: list[float] = []
    pair_times: list[float] = []
    skipped_reasons: dict[str, int] = {}
    pairs_total = 0
    pairs_skipped = 0
    skipped_genuine = 0
    skipped_impostor = 0

    for pair in pairs:
        pairs_total += 1
        if not warmup[0]:
            for image in (pair.image_a, pair.image_b):
                with contextlib.suppress(QualityError):
                    embedder.detect_and_embed(image)
            warmup[0] = True

        started = time.perf_counter()
        try:
            start_a = time.perf_counter()
            result_a = embedder.detect_and_embed(pair.image_a)
            embed_times.append((time.perf_counter() - start_a) * 1000.0)

            start_b = time.perf_counter()
            result_b = embedder.detect_and_embed(pair.image_b)
            embed_times.append((time.perf_counter() - start_b) * 1000.0)
        except QualityError as exc:
            pairs_skipped += 1
            skipped_reasons[exc.reason] = skipped_reasons.get(exc.reason, 0) + 1
            if pair.label == 1:
                skipped_genuine += 1
            else:
                skipped_impostor += 1
            continue

        score = cosine_similarity(result_a.embedding, result_b.embedding)
        pair_times.append((time.perf_counter() - started) * 1000.0)
        scores.append(score)
        labels.append(pair.label)

        if pairs_total % 100 == 0:
            print(f"  processed {pairs_total} pairs ({pairs_skipped} skipped)")

    if not scores:
        raise SystemExit("no pairs survived the quality gate; nothing to evaluate")

    return SplitResult(
        source=source,
        pairs_total=pairs_total,
        pairs_skipped=pairs_skipped,
        skipped_reasons=skipped_reasons,
        skipped_genuine=skipped_genuine,
        skipped_impostor=skipped_impostor,
        scores=np.asarray(scores, dtype=np.float64),
        labels=np.asarray(labels, dtype=np.int64),
        embed_times=embed_times,
        pair_times=pair_times,
    )


def _split_report(result: SplitResult) -> SplitReport:
    auc = roc_auc(result.scores, result.labels)
    eer_threshold, eer = choose_eer(result.scores, result.labels)
    return SplitReport(
        source=result.source,
        pairs_total=result.pairs_total,
        pairs_used=result.pairs_used,
        pairs_skipped=result.pairs_skipped,
        skipped_reasons=result.skipped_reasons,
        skipped_genuine=result.skipped_genuine,
        skipped_impostor=result.skipped_impostor,
        positives=result.positives,
        negatives=result.negatives,
        fta=failure_to_acquire(result.pairs_skipped, result.pairs_total),
        roc_auc=auc,
        eer=eer,
        eer_threshold=eer_threshold,
    )


def _latency_summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "n": int(array.size),
    }


def _encode_jpeg(image: np.ndarray, quality: int = 95) -> bytes:
    success, buffer = cv2.imencode(
        ".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    )
    if not success:
        raise RuntimeError("could not encode the synthetic frame as JPEG")
    return buffer.tobytes()


def _timed_decode_embed(
    embedder: OpenCVYuNetSFaceEmbedder, encoded: bytes, repeats: int = 7
) -> dict[str, float] | None:
    """Time the full request path: JPEG decode, then detect + align + embed."""

    try:
        embedder.detect_and_embed(decode_image_bgr(encoded))
    except QualityError:
        return None
    timings: list[float] = []
    for _ in range(repeats):
        started = time.perf_counter()
        image = decode_image_bgr(encoded)
        embedder.detect_and_embed(image)
        timings.append((time.perf_counter() - started) * 1000.0)
    return _latency_summary(timings)


def _synthetic_frame(source: np.ndarray, width: int, height: int) -> np.ndarray:
    """Paste a real LFW face into a frame of the requested pixel dimensions."""

    frame = np.full((height, width, 3), (96, 96, 96), dtype=np.uint8)
    tile = int(min(width, height) * 0.6)
    face = cv2.resize(source, (tile, tile), interpolation=cv2.INTER_AREA)
    x = (width - tile) // 2
    y = (height - tile) // 2
    frame[y : y + tile, x : x + tile] = face
    return frame


def _latency_benchmarks(
    embedder: OpenCVYuNetSFaceEmbedder, source: np.ndarray
) -> dict[str, dict[str, float]]:
    benchmarks: dict[str, dict[str, float]] = {}
    for name, width, height in (("720p", 1280, 720), ("12mp", 4000, 3000)):
        encoded = _encode_jpeg(_synthetic_frame(source, width, height))
        summary = _timed_decode_embed(embedder, encoded)
        if summary is not None:
            summary["pixels"] = float(width * height)
            benchmarks[name] = summary
    return benchmarks


def _fmt_pct(value: float | None, digits: int = 2) -> str:
    return "n/a" if value is None else f"{value * 100:.{digits}f}%"


def _fmt_num(value: float | None, digits: int = 4) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _fmt_interval(interval: tuple[float, float] | None) -> str:
    if interval is None:
        return "n/a"
    return f"{interval[0] * 100:.2f}% - {interval[1] * 100:.2f}%"


def _fmt_reasons(reasons: dict[str, int]) -> str:
    if not reasons:
        return "none"
    return ", ".join(f"`{name}` {count}" for name, count in sorted(reasons.items()))


def _git_state() -> tuple[str, bool]:
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown", False
    return sha or "unknown", bool(status)


def render_markdown(report: EvalReport) -> str:
    train = report.train
    test = report.test
    production = report.production_metrics
    recommended = report.recommended_metrics

    lines = [
        "# LFW 1:1 face verification report",
        "",
        f"Generated: {report.generated_at}",
        "",
        f"- Model: `{report.model_version}`",
        f"- Git revision: `{report.git_sha}`"
        + (" (working tree dirty)" if report.git_dirty else ""),
        f"- Dataset images: `{report.images_dir}`",
        f"- Threshold fitted on: `{Path(report.train_pairs_file).name}`",
        f"- Metrics reported on: `{Path(report.test_pairs_file).name}`",
        "",
        "## Protocol",
        "",
        f"- Train split: {train.pairs_used} usable of {train.pairs_total} pairs "
        f"({train.positives} same / {train.negatives} different), "
        f"{train.pairs_skipped} skipped ({_fmt_reasons(train.skipped_reasons)}).",
        f"- Test split: {test.pairs_used} usable of {test.pairs_total} pairs "
        f"({test.positives} same / {test.negatives} different), "
        f"{test.pairs_skipped} skipped ({_fmt_reasons(test.skipped_reasons)}).",
        f"- FAR target for threshold fitting: {report.far_target * 100:.0f}% on train.",
        "- The production default (0.363) is OpenCV's a-priori SFace cosine "
        "threshold and was **not** tuned on the test split.",
        "",
        "## Headline test metrics",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| ROC AUC | {_fmt_num(test.roc_auc)} |",
        f"| EER | {_fmt_pct(test.eer)} |",
        f"| EER threshold | {_fmt_num(test.eer_threshold)} |",
        f"| Failure to acquire (FTA) | {_fmt_pct(test.fta)} "
        f"({test.pairs_skipped}/{test.pairs_total} pairs) |",
        f"| Generalized FRR at {report.default_threshold:.3f} | "
        f"**{_fmt_pct(report.production_generalized_frr)}** "
        f"({test.skipped_genuine + production.false_rejects}/"
        f"{test.positives + test.skipped_genuine} genuine pairs) |",
        f"| FAR at {report.default_threshold:.3f} | {_fmt_pct(production.far)} "
        f"({production.false_accepts}/{production.negatives} impostor pairs) |",
        f"| FAR 95% upper bound (rule of three) | "
        f"**{_fmt_pct(report.production_far_upper_95_rule_of_three)}** |",
        f"| FAR 95% Wilson interval | {_fmt_interval(report.production_far_ci_95_wilson)} |",
        f"| FRR at {report.default_threshold:.3f} | {_fmt_pct(production.frr)} |",
        f"| FRR 95% Wilson interval | {_fmt_interval(report.production_frr_ci_95_wilson)} |",
        "",
        "Zero false accepts in a finite impostor sample does not mean FAR is "
        "zero. With 0 accepts in `n` trials the one-sided 95% upper bound is "
        "`1 - 0.05^(1/n)`, approximately `3/n` (the rule of three).",
        "",
        "## Thresholds",
        "",
        f"The service ships with **{report.default_threshold:.4f}** "
        "(`FACECHECK_MATCH_THRESHOLD`), OpenCV's documented SFace cosine "
        "threshold. That value is an a-priori default, not a result tuned on "
        "the test split.",
        "",
        f"Fitting on the train split for FAR <= {report.far_target * 100:.0f}% "
        f"gives **{report.recommended_threshold:.4f}** "
        f"(train TPR {_fmt_pct(report.recommended_threshold_train_metrics.tpr)}, "
        f"train FAR {_fmt_pct(report.recommended_threshold_train_metrics.far)}). "
        f"Applied unchanged to the test split it gives TPR "
        f"{_fmt_pct(recommended.tpr)}, FAR {_fmt_pct(recommended.far)}, "
        f"generalized FRR {_fmt_pct(report.recommended_generalized_frr)}.",
        "",
        f"Best Youden J on the test split is {_fmt_num(report.youden_threshold)} "
        f"(J={_fmt_num(report.youden_index, 3)}); it is reported for reference "
        "and is not used to pick the shipped threshold.",
        "",
        "Averaging the k enrollment templates would lower the impostor score "
        "for a lucky frame, but max-over-k gives each impostor k independent "
        "chances, so it is optimistic for FAR (FAR can grow roughly k-times) "
        "while it is conservative for FRR. The single-template LFW protocol "
        "does not measure that effect.",
        "",
        "## Operating points (test split)",
        "",
        "| Threshold | TPR | FAR | FRR | Accuracy |",
        "| --- | --- | --- | --- | --- |",
    ]
    for point in report.operating_points:
        lines.append(
            f"| {point.threshold:.4f} | {_fmt_pct(point.tpr)} | {_fmt_pct(point.far)} | "
            f"{_fmt_pct(point.frr)} | {_fmt_pct(point.accuracy)} |"
        )

    if report.score_stats:
        lines += [
            "",
            "## Score distribution (test split)",
            "",
            "| Class | Mean | Median | Std | n |",
            "| --- | --- | --- | --- | --- |",
        ]
        for name, stats in report.score_stats.items():
            lines.append(
                f"| {name} | {stats['mean']:.4f} | {stats['median']:.4f} | "
                f"{stats['std']:.4f} | {int(stats['n'])} |"
            )

    latency = report.latency_ms
    lines += [
        "",
        "## CPU latency (single process, warm models)",
        "",
        "| Measurement | Mean | Median | p95 |",
        "| --- | --- | --- | --- |",
        f"| Detect + align + embed (LFW 250px) | {latency['embed']['mean']:.2f} ms | "
        f"{latency['embed']['median']:.2f} ms | {latency['embed']['p95']:.2f} ms |",
        f"| Verification (two images + cosine) | {latency['pair']['mean']:.2f} ms | "
        f"{latency['pair']['median']:.2f} ms | {latency['pair']['p95']:.2f} ms |",
    ]
    for name, label in (
        ("720p", "JPEG decode + detect + align + embed, 1280x720"),
        ("12mp", "JPEG decode + detect + align + embed, 4000x3000"),
    ):
        if name in latency:
            summary = latency[name]
            lines.append(
                f"| {label} (downscaled to "
                f"{report.environment.get('detect_max_side', 'n/a')}px) | "
                f"{summary['mean']:.2f} ms | {summary['median']:.2f} ms | "
                f"{summary['p95']:.2f} ms |"
            )

    lines += ["", "## Environment", ""]
    for key, value in report.environment.items():
        lines.append(f"- {key}: {value}")
    lines += ["", "## Model files", ""]
    for filename, digest in report.model_sha256.items():
        lines.append(f"- `{filename}`: `{digest}`")

    lines += [
        "",
        "## Method notes",
        "",
        "- Each LFW pair is one enroll template and one probe; the score is cosine "
        "similarity of L2-normalized SFace embeddings.",
        "- Images are downscaled so the longest side is at most "
        f"{report.environment.get('detect_max_side', 'n/a')} px before detection, "
        "alignment, and embedding. A face must be at least the configured pixel "
        "minimum and at least the configured fraction of the frame's short side.",
        "- Pairs where either image fails the quality gate are skipped. FTA counts "
        "all skipped pairs; generalized FRR counts skipped genuine pairs as "
        "failures, so a detector that cannot acquire faces does not look good.",
        "- The 720p and 12 MP latency rows encode the synthetic frame as JPEG and "
        "time `imdecode` plus detect + align + embed, matching the upload path.",
        "- Confidence intervals are Wilson score intervals at 95%; the FAR bound "
        "uses the exact zero-event rule of three.",
        "- The production service enrolls 3-5 templates and takes the maximum "
        "cosine. That is conservative for FRR but optimistic for FAR, which this "
        "single-template benchmark cannot quantify.",
    ]
    return "\n".join(lines) + "\n"


def _plot(
    report_dir: Path,
    scores: np.ndarray,
    labels: np.ndarray,
    report: EvalReport,
) -> None:
    roc = roc_curve(scores, labels)
    if roc is None:
        return

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fpr, tpr, _ = roc
    fig, axis = plt.subplots(figsize=(5, 5))
    axis.plot(fpr, tpr, label=f"YuNet+SFace (AUC={report.test.roc_auc:.4f})")
    axis.plot([0, 1], [0, 1], linestyle="--", color="grey", label="chance")
    axis.scatter(
        [report.production_metrics.far],
        [report.production_metrics.tpr],
        color="tab:red",
        label=f"shipped ({report.default_threshold:.3f})",
    )
    axis.scatter(
        [report.recommended_metrics.far],
        [report.recommended_metrics.tpr],
        color="tab:green",
        label=f"train FAR<=1% ({report.recommended_threshold:.3f})",
    )
    axis.set_xlabel("False accept rate (FPR)")
    axis.set_ylabel("True accept rate (TPR)")
    axis.set_title("LFW 1:1 verification ROC (test split)")
    axis.legend(loc="lower right")
    axis.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(report_dir / "lfw_roc.png", dpi=150)
    plt.close(fig)

    fnmr = 1.0 - tpr
    fig, axis = plt.subplots(figsize=(5, 5))
    axis.plot(
        np.clip(fpr, 1e-4, 1.0),
        fnmr,
        label=f"YuNet+SFace (EER={report.test.eer * 100:.2f}%)"
        if report.test.eer is not None
        else "YuNet+SFace",
    )
    axis.scatter(
        [max(report.production_metrics.far, 1e-4)],
        [report.production_metrics.frr],
        color="tab:red",
        label=f"shipped ({report.default_threshold:.3f})",
    )
    axis.set_xscale("log")
    axis.set_xlim(1e-4, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.set_xlabel("False match rate (FMR, log scale)")
    axis.set_ylabel("False non-match rate (FNMR)")
    axis.set_title("LFW DET curve (test split)")
    axis.legend(loc="upper right")
    axis.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(report_dir / "lfw_det.png", dpi=150)
    plt.close(fig)


def _performance_block(
    result: SplitResult,
    metrics: Metrics,
) -> dict[str, object]:
    attempted_genuine = result.positives + result.skipped_genuine
    return {
        "metrics": metrics,
        "far_upper_95_rule_of_three": (
            rule_of_three(metrics.negatives) if metrics.false_accepts == 0 else None
        ),
        "far_ci_95_wilson": wilson_interval(metrics.false_accepts, metrics.negatives),
        "frr_ci_95_wilson": wilson_interval(metrics.false_rejects, metrics.positives),
        "generalized_frr": generalized_frr(
            result.skipped_genuine, metrics.false_rejects, attempted_genuine
        ),
        "generalized_frr_ci_95_wilson": wilson_interval(
            result.skipped_genuine + metrics.false_rejects, attempted_genuine
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate YuNet+SFace on LFW pairs")
    parser.add_argument("--images-dir", type=Path, default=None)
    parser.add_argument("--pairs-file", type=Path, default=None, help="test pairs file")
    parser.add_argument(
        "--train-pairs-file", type=Path, default=None, help="pairs used for threshold fitting"
    )
    parser.add_argument("--limit", type=int, default=0, help="evaluate only the first N pairs")
    parser.add_argument("--report-dir", type=Path, default=REPO_ROOT / "eval/reports")
    parser.add_argument("--no-plot", action="store_true", help="skip ROC/DET PNGs")
    parser.add_argument(
        "--min-face-size", type=int, default=None, help="override the absolute quality gate"
    )
    args = parser.parse_args(argv)

    settings = Settings(_env_file=None)
    if args.min_face_size is not None:
        settings = settings.model_copy(update={"min_face_size": args.min_face_size})

    images_dir, pairs_file, train_pairs_file = _resolve_dataset_paths(
        args.images_dir, args.pairs_file, args.train_pairs_file
    )

    embedder = OpenCVYuNetSFaceEmbedder.from_settings(settings)
    print(f"model: {embedder.model_version}")
    print(f"quality gate: min side {settings.min_face_size}px and "
          f"{settings.min_face_frac:.0%} of the frame, exactly one face")
    print(f"detect max side: {settings.detect_max_side}px")

    warmup = [False]
    train_pairs, train_source = iter_pairs(images_dir, train_pairs_file, args.limit)
    print(f"train: {train_source}")
    train = _evaluate(embedder, train_pairs, train_source, warmup=warmup)

    test_pairs, test_source = iter_pairs(images_dir, pairs_file, args.limit)
    print(f"test: {test_source}")
    test = _evaluate(embedder, test_pairs, test_source, warmup=warmup)

    recommended_threshold, recommended_train_metrics = choose_far_threshold(
        train.scores, train.labels, FAR_TARGET
    )
    if recommended_threshold is None or recommended_train_metrics is None:
        raise SystemExit("train split needs both genuine and impostor pairs to fit a threshold")

    production_metrics = scores_at(test.scores, test.labels, settings.match_threshold)
    recommended_metrics = scores_at(test.scores, test.labels, recommended_threshold)
    youden_threshold, youden_metrics = choose_youden(test.scores, test.labels)

    production_block = _performance_block(test, production_metrics)
    recommended_block = _performance_block(test, recommended_metrics)

    operating_points: list[Metrics] = []
    seen: set[float] = set()
    for threshold in (
        *THRESHOLD_GRID,
        recommended_threshold,
        youden_threshold,
        settings.match_threshold,
    ):
        rounded = round(float(threshold), 6)
        if rounded in seen:
            continue
        seen.add(rounded)
        operating_points.append(scores_at(test.scores, test.labels, rounded))
    operating_points.sort(key=lambda point: point.threshold)

    same_scores = test.scores[test.labels == 1]
    different_scores = test.scores[test.labels == 0]
    score_stats = {
        "same person": {
            "mean": float(np.mean(same_scores)),
            "median": float(np.median(same_scores)),
            "std": float(np.std(same_scores)),
            "n": float(same_scores.size),
        },
        "different person": {
            "mean": float(np.mean(different_scores)),
            "median": float(np.median(different_scores)),
            "std": float(np.std(different_scores)),
            "n": float(different_scores.size),
        },
    }

    latency: dict[str, dict[str, float]] = {
        "embed": _latency_summary(test.embed_times),
        "pair": _latency_summary(test.pair_times),
    }
    sample_image = cv2.imread(
        str(_read_pairs_file(pairs_file, images_dir)[0][0]), cv2.IMREAD_COLOR
    )
    if sample_image is not None:
        latency.update(_latency_benchmarks(embedder, sample_image))

    git_sha, git_dirty = _git_state()
    report = EvalReport(
        generated_at=datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
        git_sha=git_sha,
        git_dirty=git_dirty,
        model_version=embedder.model_version,
        model_sha256={
            settings.detector_file: file_sha256(settings.detector_path),
            settings.recognizer_file: file_sha256(settings.recognizer_path),
        },
        images_dir=display_path(images_dir),
        train_pairs_file=display_path(train_pairs_file),
        test_pairs_file=display_path(pairs_file),
        far_target=FAR_TARGET,
        default_threshold=settings.match_threshold,
        train=_split_report(train),
        test=_split_report(test),
        recommended_threshold=recommended_threshold,
        recommended_threshold_basis=f"FAR <= {FAR_TARGET * 100:.0f}% on train",
        recommended_threshold_train_metrics=recommended_train_metrics,
        youden_threshold=youden_threshold,
        youden_index=youden_metrics.tpr - youden_metrics.far,
        production_metrics=production_metrics,
        production_far_upper_95_rule_of_three=production_block["far_upper_95_rule_of_three"],  # type: ignore[arg-type]
        production_far_ci_95_wilson=production_block["far_ci_95_wilson"],  # type: ignore[arg-type]
        production_frr_ci_95_wilson=production_block["frr_ci_95_wilson"],  # type: ignore[arg-type]
        production_generalized_frr=production_block["generalized_frr"],  # type: ignore[arg-type]
        production_generalized_frr_ci_95_wilson=production_block["generalized_frr_ci_95_wilson"],  # type: ignore[arg-type]
        recommended_metrics=recommended_metrics,
        recommended_far_upper_95_rule_of_three=recommended_block["far_upper_95_rule_of_three"],  # type: ignore[arg-type]
        recommended_far_ci_95_wilson=recommended_block["far_ci_95_wilson"],  # type: ignore[arg-type]
        recommended_frr_ci_95_wilson=recommended_block["frr_ci_95_wilson"],  # type: ignore[arg-type]
        recommended_generalized_frr=recommended_block["generalized_frr"],  # type: ignore[arg-type]
        recommended_generalized_frr_ci_95_wilson=recommended_block["generalized_frr_ci_95_wilson"],  # type: ignore[arg-type]
        operating_points=operating_points,
        score_stats=score_stats,
        latency_ms=latency,
        environment={
            "platform": platform.platform(),
            "processor": platform.processor() or "unknown",
            "cpu_count": __import__("os").cpu_count(),
            "python": platform.python_version(),
            "opencv": cv2.__version__,
            "numpy": np.__version__,
            "detect_max_side": settings.detect_max_side,
            "min_face_size": settings.min_face_size,
            "min_face_frac": settings.min_face_frac,
            "latency_benchmarks": "JPEG decode + detect + align + embed for 720p and 12 MP",
        },
    )

    report_dir = args.report_dir
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "lfw_report.md").write_text(render_markdown(report), encoding="utf-8")
    (report_dir / "lfw_metrics.json").write_text(
        json.dumps(asdict(report), indent=2) + "\n", encoding="utf-8"
    )

    if not args.no_plot:
        _plot(report_dir, test.scores, test.labels, report)

    print()
    print(f"train pairs used: {train.pairs_used} (skipped {train.pairs_skipped})")
    print(f"test pairs used: {test.pairs_used} (skipped {test.pairs_skipped})")
    print(f"test ROC AUC: {_fmt_num(report.test.roc_auc)}")
    print(f"test EER: {_fmt_pct(report.test.eer)} at threshold {_fmt_num(report.test.eer_threshold)}")
    print(
        f"threshold from train (FAR <= 1%): {_fmt_num(recommended_threshold)} -> "
        f"test TPR {_fmt_pct(recommended_metrics.tpr)}, "
        f"FAR {_fmt_pct(recommended_metrics.far)}, "
        f"generalized FRR {_fmt_pct(report.recommended_generalized_frr)}"
    )
    print(
        f"shipped {settings.match_threshold:.3f}: "
        f"FAR {_fmt_pct(production_metrics.far)} "
        f"(95% upper bound {_fmt_pct(report.production_far_upper_95_rule_of_three)}), "
        f"FRR {_fmt_pct(production_metrics.frr)}, "
        f"generalized FRR {_fmt_pct(report.production_generalized_frr)}"
    )
    print(f"wrote {report_dir / 'lfw_report.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
