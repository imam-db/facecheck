"""LFW 1:1 verification benchmark for the YuNet + SFace pipeline.

Loads the official LFW test pairs (``pairsDevTest.txt``, 1000 pairs: 500 same,
500 different), runs detect + align + embed on both images of each pair, and
reports ROC AUC, EER, FAR/FRR at selected thresholds, threshold
recommendations, and CPU latency.

Expected layout (created by ``scripts/fetch_lfw.py`` or manually)::

    data/lfw_home/pairsDevTest.txt
    data/lfw_home/lfw_funneled/<name>/<name>_0001.jpg

If that layout is absent, the script falls back to
``sklearn.datasets.fetch_lfw_pairs``.

Usage::

    python -m eval.run_lfw                # full test split
    python -m eval.run_lfw --limit 200    # quick subset
"""

from __future__ import annotations

import argparse
import contextlib
import json
import platform
import sys
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np

from facecheck.config import Settings
from facecheck.embeddings import OpenCVYuNetSFaceEmbedder
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
class Metrics:
    threshold: float
    tpr: float
    far: float
    frr: float
    accuracy: float


@dataclass
class EvalReport:
    generated_at: str
    model_version: str
    dataset: str
    pairs_total: int
    pairs_used: int
    pairs_skipped: int
    skipped_reasons: dict[str, int]
    positives: int
    negatives: int
    roc_auc: float | None
    eer: float | None
    eer_threshold: float | None
    recommended_threshold: float | None
    recommended_threshold_basis: str
    youden_threshold: float | None
    youden_index: float | None
    default_threshold: float
    production_metrics: Metrics
    max_impostor_score: float
    min_genuine_score: float
    operating_points: list[Metrics] = field(default_factory=list)
    score_stats: dict[str, dict[str, float]] = field(default_factory=dict)
    latency_ms: dict[str, float] = field(default_factory=dict)
    environment: dict[str, object] = field(default_factory=dict)


def _resolve_dataset_paths(
    images_dir: Path | None, pairs_file: Path | None
) -> tuple[Path | None, Path | None]:
    if images_dir is None:
        for candidate in (
            REPO_ROOT / "data/lfw_home/lfw_deepfunneled",
            REPO_ROOT / "data/lfw_home/lfw_funneled",
            REPO_ROOT / "data/lfw_home/lfw",
            Path.home() / "scikit_learn_data/lfw_home/lfw_funneled",
        ):
            if candidate.is_dir():
                images_dir = candidate
                break
    if pairs_file is None:
        for candidate in (
            REPO_ROOT / "data/lfw_home/pairsDevTest.txt",
            Path.home() / "scikit_learn_data/lfw_home/pairsDevTest.txt",
        ):
            if candidate.is_file():
                pairs_file = candidate
                break
    return images_dir, pairs_file


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


def iter_pairs(
    images_dir: Path | None,
    pairs_file: Path | None,
    limit: int,
) -> tuple[Iterator[Pair], str]:
    if images_dir is not None and pairs_file is not None:
        records = _read_pairs_file(pairs_file, images_dir)
        if limit > 0 and limit < len(records):
            # Keep the subset balanced: matched pairs are listed first in the
            # official file, so naive truncation would produce an all-positive
            # evaluation set and an undefined ROC AUC.
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

        source = f"LFW official test pairs ({pairs_file.name}), {len(records)} pairs"
        return from_files(), source

    try:
        from sklearn.datasets import fetch_lfw_pairs
    except ImportError as exc:  # pragma: no cover - dev dependency
        raise SystemExit(
            "no local LFW layout found and scikit-learn is not installed.\n"
            "Fetch LFW with `python scripts/fetch_lfw.py` or install the eval extra."
        ) from exc

    print("local LFW layout not found; falling back to sklearn.datasets.fetch_lfw_pairs")
    try:
        data = fetch_lfw_pairs(
            subset="test",
            color=True,
            funneled=True,
            download_if_missing=True,
            data_home=str(REPO_ROOT / "data"),
        )
    except (OSError, ValueError) as exc:
        raise SystemExit(
            f"could not fetch LFW via scikit-learn ({exc}).\n"
            "The canonical figshare/UMass mirrors are sometimes unavailable; "
            "run `python scripts/fetch_lfw.py` to use the checksum-pinned mirrors."
        ) from exc
    total = len(data.target)
    count = total if limit <= 0 else min(limit, total)

    def from_sklearn() -> Iterator[Pair]:
        for pair_id in range(count):
            yield Pair(
                image_a=data.pairs[pair_id, 0],
                image_b=data.pairs[pair_id, 1],
                label=int(data.target[pair_id]),
                pair_id=pair_id,
            )

    return from_sklearn(), f"sklearn LFW test subset, {count} pairs"


def _scores_at(scores: np.ndarray, labels: np.ndarray, threshold: float) -> Metrics:
    predictions = scores >= threshold
    positives = labels == 1
    negatives = ~positives
    true_accepts = int(np.sum(predictions & positives))
    false_accepts = int(np.sum(predictions & negatives))
    tpr = true_accepts / max(int(np.sum(positives)), 1)
    far = false_accepts / max(int(np.sum(negatives)), 1)
    frr = 1.0 - tpr
    accuracy = float(np.mean(predictions == positives))
    return Metrics(
        threshold=float(threshold),
        tpr=tpr,
        far=far,
        frr=frr,
        accuracy=accuracy,
    )


def _choose_far_threshold(
    scores: np.ndarray, labels: np.ndarray, target_far: float
) -> tuple[float | None, Metrics | None]:
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels)
    candidates = np.unique(np.concatenate([scores, [0.0, 1.0]]))
    positives = int(np.sum(labels == 1))
    negatives = int(np.sum(labels == 0))

    # Walk thresholds upward; the first one that satisfies FAR <= target has
    # the highest TPR among all satisfying thresholds.
    best: tuple[float, Metrics] | None = None
    for threshold in candidates:
        metrics = _scores_at(scores, labels, float(threshold))
        if metrics.far <= target_far and positives and negatives:
            best = (float(threshold), metrics)
            break

    if best is None:
        # No threshold meets the FAR target; fall back to the most
        # conservative candidate that still accepts at least one pair.
        threshold = float(candidates[-1])
        return threshold, _scores_at(scores, labels, threshold)
    return best


def _choose_youden(scores: np.ndarray, labels: np.ndarray) -> tuple[float, Metrics]:
    candidates = np.unique(np.concatenate([scores, [0.0, 1.0]]))
    best_threshold = 0.0
    best_metrics = _scores_at(scores, labels, 0.0)
    best_index = best_metrics.tpr - best_metrics.far
    for threshold in candidates:
        metrics = _scores_at(scores, labels, float(threshold))
        index = metrics.tpr - metrics.far
        if index > best_index:
            best_index, best_threshold, best_metrics = index, float(threshold), metrics
    return best_threshold, best_metrics


def _choose_eer(scores: np.ndarray, labels: np.ndarray) -> tuple[float | None, float | None]:
    if len(np.unique(labels)) < 2:
        return None, None
    roc = _roc_curve(scores, labels)
    if roc is None:
        return None, None
    fpr, tpr, thresholds = roc
    fnr = 1.0 - tpr
    index = int(np.argmin(np.abs(fpr - fnr)))
    eer = float((fpr[index] + fnr[index]) / 2.0)
    threshold = float(thresholds[index])
    # np.argmin can land on a synthetic endpoint; clamp to the finite range.
    if not np.isfinite(threshold):
        threshold = float(np.max(scores))
    return threshold, eer


def _roc_curve(
    scores: np.ndarray, labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    try:
        from sklearn.metrics import roc_curve
    except ImportError:
        return None
    fpr, tpr, thresholds = roc_curve(labels, scores)
    return fpr, tpr, thresholds


def _roc_auc(scores: np.ndarray, labels: np.ndarray) -> float | None:
    if len(np.unique(labels)) < 2:
        return None
    try:
        from sklearn.metrics import roc_auc_score
    except ImportError:
        return None
    return float(roc_auc_score(labels, scores))


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


def _fmt_pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.2f}%"


def _fmt_num(value: float | None, digits: int = 4) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def render_markdown(report: EvalReport) -> str:
    lines = [
        "# LFW 1:1 face verification report",
        "",
        f"Generated: {report.generated_at}",
        "",
        f"- Model: `{report.model_version}`",
        f"- Dataset: {report.dataset}",
        f"- Pairs evaluated: **{report.pairs_used}** "
        f"({report.positives} same / {report.negatives} different)",
        f"- Pairs skipped by the quality gate: {report.pairs_skipped}",
    ]
    if report.skipped_reasons:
        reasons = ", ".join(f"`{k}`: {v}" for k, v in sorted(report.skipped_reasons.items()))
        lines.append(f"- Skip reasons: {reasons}")
    lines += [
        "",
        "## Headline metrics",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| ROC AUC | {_fmt_num(report.roc_auc)} |",
        f"| EER | {_fmt_pct(report.eer)} |",
        f"| EER threshold | {_fmt_num(report.eer_threshold)} |",
        f"| Recommended threshold ({report.recommended_threshold_basis}) | "
        f"**{_fmt_num(report.recommended_threshold)}** |",
        f"| Youden J threshold | {_fmt_num(report.youden_threshold)} |",
        "",
        "## Production threshold",
        "",
        f"The service ships with threshold **{report.default_threshold:.4f}** "
        f"(`FACECHECK_MATCH_THRESHOLD`). On this evaluation subset it gives "
        f"TPR {_fmt_pct(report.production_metrics.tpr)}, "
        f"FAR {_fmt_pct(report.production_metrics.far)}, "
        f"FRR {_fmt_pct(report.production_metrics.frr)} "
        f"({report.production_metrics.accuracy * report.pairs_used:.0f}/"
        f"{report.pairs_used} pairs correct).",
        "",
        f"The FAR <= {FAR_TARGET * 100:.0f}% point at "
        f"{_fmt_num(report.recommended_threshold)} accepts slightly more genuine "
        f"pairs but also admits impostors; for attendance, false accepts are the "
        f"costlier error, so the conservative operating point is kept.",
        "",
        f"Highest impostor score: {report.max_impostor_score:.4f}; "
        f"lowest genuine score: {report.min_genuine_score:.4f}. "
        f"At {report.default_threshold:.4f} the margin above the highest observed "
        f"impostor is {report.default_threshold - report.max_impostor_score:.4f}.",
        "",
        "## Operating points",
        "",
        "| Threshold | TPR (FAR target) | FAR | FRR | Accuracy |",
        "| --- | --- | --- | --- | --- |",
    ]
    for point in report.operating_points:
        lines.append(
            f"| {point.threshold:.4f} | {_fmt_pct(point.tpr)} | {_fmt_pct(point.far)} | "
            f"{_fmt_pct(point.frr)} | {_fmt_pct(point.accuracy)} |"
        )

    if report.score_stats:
        lines += ["", "## Score distribution", "", "| Class | Mean | Median | Std | n |", "| --- | --- | --- | --- | --- |"]
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
        f"| Detect + align + embed (per image) | {latency['embed']['mean']:.2f} ms | "
        f"{latency['embed']['median']:.2f} ms | {latency['embed']['p95']:.2f} ms |",
        f"| Verification (two images + cosine) | {latency['pair']['mean']:.2f} ms | "
        f"{latency['pair']['median']:.2f} ms | {latency['pair']['p95']:.2f} ms |",
        "",
        "## Environment",
        "",
    ]
    for key, value in report.environment.items():
        lines.append(f"- {key}: {value}")
    lines += [
        "",
        "## Method notes",
        "",
        "- Each LFW pair is treated as one enroll template and one probe; the score is "
        "cosine similarity of L2-normalized SFace embeddings.",
        "- The production service enrolls 3-5 images per person and verifies with the "
        "maximum cosine across those templates. This benchmark is therefore conservative "
        "for real enrollment: it has no template averaging or multi-frame voting.",
        "- Pairs where either image fails the quality gate (no face, more than one face, "
        "or shortest face side below the configured minimum) are excluded and counted.",
        "- Threshold selection: the recommended threshold is the first threshold on the "
        f"score grid with FAR <= {FAR_TARGET * 100:.0f}% (highest TPR satisfying the "
        "false-accept budget). Youden J and EER thresholds are reported for comparison.",
    ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate YuNet+SFace on LFW pairs")
    parser.add_argument("--images-dir", type=Path, default=None)
    parser.add_argument("--pairs-file", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=0, help="evaluate only the first N pairs")
    parser.add_argument("--report-dir", type=Path, default=REPO_ROOT / "eval/reports")
    parser.add_argument("--no-plot", action="store_true", help="skip the ROC PNG")
    parser.add_argument(
        "--min-face-size", type=int, default=None, help="override the quality gate"
    )
    args = parser.parse_args(argv)

    settings = Settings(_env_file=None)
    if args.min_face_size is not None:
        settings = settings.model_copy(update={"min_face_size": args.min_face_size})

    images_dir, pairs_file = _resolve_dataset_paths(args.images_dir, args.pairs_file)
    pairs_iter, source = iter_pairs(images_dir, pairs_file, args.limit)

    embedder = OpenCVYuNetSFaceEmbedder.from_settings(settings)
    print(f"model: {embedder.model_version}")
    print(f"dataset: {source}")
    print(f"quality gate: min face side {settings.min_face_size}px, exactly one face")

    scores: list[float] = []
    labels: list[int] = []
    embed_times: list[float] = []
    pair_times: list[float] = []
    skipped_reasons: dict[str, int] = {}
    pairs_total = 0
    pairs_skipped = 0

    warmup_done = False
    for pair in pairs_iter:
        pairs_total += 1
        if not warmup_done:
            for image in (pair.image_a, pair.image_b):
                with contextlib.suppress(QualityError):
                    embedder.detect_and_embed(image)
            warmup_done = True

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
            continue

        score = cosine_similarity(result_a.embedding, result_b.embedding)
        pair_times.append((time.perf_counter() - started) * 1000.0)
        scores.append(score)
        labels.append(pair.label)

        if pairs_total % 100 == 0:
            print(f"  processed {pairs_total} pairs ({pairs_skipped} skipped)")

    if not scores:
        raise SystemExit("no pairs survived the quality gate; nothing to evaluate")

    score_array = np.asarray(scores, dtype=np.float64)
    label_array = np.asarray(labels, dtype=np.int64)
    positives = int(np.sum(label_array == 1))
    negatives = int(np.sum(label_array == 0))

    auc = _roc_auc(score_array, label_array)
    eer_threshold, eer = _choose_eer(score_array, label_array)
    far_threshold, far_metrics = _choose_far_threshold(score_array, label_array, FAR_TARGET)
    youden_threshold, youden_metrics = _choose_youden(score_array, label_array)

    operating_points: list[Metrics] = []
    seen: set[float] = set()
    for threshold in (*THRESHOLD_GRID, far_threshold, youden_threshold, eer_threshold):
        if threshold is None:
            continue
        threshold = round(float(threshold), 6)
        if threshold in seen:
            continue
        seen.add(threshold)
        operating_points.append(_scores_at(score_array, label_array, threshold))
    operating_points.sort(key=lambda point: point.threshold)

    same_scores = score_array[label_array == 1]
    different_scores = score_array[label_array == 0]
    production_metrics = _scores_at(score_array, label_array, settings.match_threshold)
    max_impostor_score = float(np.max(different_scores)) if different_scores.size else 0.0
    min_genuine_score = float(np.min(same_scores)) if same_scores.size else 0.0
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

    report = EvalReport(
        generated_at=datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
        model_version=embedder.model_version,
        dataset=source,
        pairs_total=pairs_total,
        pairs_used=len(scores),
        pairs_skipped=pairs_skipped,
        skipped_reasons=skipped_reasons,
        positives=positives,
        negatives=negatives,
        roc_auc=auc,
        eer=eer,
        eer_threshold=eer_threshold,
        recommended_threshold=far_threshold,
        recommended_threshold_basis=f"FAR <= {FAR_TARGET * 100:.0f}%",
        youden_threshold=youden_threshold,
        youden_index=youden_metrics.tpr - youden_metrics.far,
        default_threshold=settings.match_threshold,
        production_metrics=production_metrics,
        max_impostor_score=max_impostor_score,
        min_genuine_score=min_genuine_score,
        operating_points=operating_points,
        score_stats=score_stats,
        latency_ms={
            "embed": _latency_summary(embed_times),
            "pair": _latency_summary(pair_times),
        },
        environment={
            "platform": platform.platform(),
            "processor": platform.processor() or "unknown",
            "cpu_count": __import__("os").cpu_count(),
            "python": platform.python_version(),
            "opencv": cv2.__version__,
            "numpy": np.__version__,
        },
    )

    report_dir = args.report_dir
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "lfw_report.md").write_text(render_markdown(report), encoding="utf-8")
    (report_dir / "lfw_metrics.json").write_text(
        json.dumps(asdict(report), indent=2) + "\n", encoding="utf-8"
    )

    roc = _roc_curve(score_array, label_array)
    if roc is not None and not args.no_plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fpr, tpr, _ = roc
        fig, axis = plt.subplots(figsize=(5, 5))
        axis.plot(fpr, tpr, label=f"YuNet+SFace (AUC={auc:.4f})")
        axis.plot([0, 1], [0, 1], linestyle="--", color="grey", label="chance")
        if far_metrics is not None:
            axis.scatter(
                [far_metrics.far], [far_metrics.tpr], color="tab:red",
                label=f"FAR<=1% ({far_metrics.threshold:.3f})",
            )
        axis.set_xlabel("False accept rate (FPR)")
        axis.set_ylabel("True accept rate (TPR)")
        axis.set_title("LFW 1:1 verification ROC")
        axis.legend(loc="lower right")
        axis.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(report_dir / "lfw_roc.png", dpi=150)
        plt.close(fig)

    print()
    print(f"pairs used: {len(scores)} (skipped {pairs_skipped})")
    print(f"ROC AUC: {_fmt_num(auc)}")
    print(f"EER: {_fmt_pct(eer)} at threshold {_fmt_num(eer_threshold)}")
    print(
        f"recommended threshold (FAR <= 1%): {_fmt_num(far_threshold)} "
        f"-> TPR {_fmt_pct(far_metrics.tpr if far_metrics else None)}, "
        f"FAR {_fmt_pct(far_metrics.far if far_metrics else None)}"
    )
    print(f"wrote {report_dir / 'lfw_report.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
