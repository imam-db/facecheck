"""Presentation-attack detection (PAD) evaluation harness.

The harness scores an out-of-repo capture directory listed in a manifest CSV
and writes aggregate JSON and Markdown reports. It follows the presentation
unit of ISO/IEC 30107-3: one row is one presentation, not one video frame. It
never writes image paths or per-presentation scores into the reports.

Production detection uses the configured YuNet threshold. A presentation that
fails that gate is counted as a detection failure and excluded from APCER and
BPCER. An optional diagnostic pass reruns those rows with a lower detection
threshold and reports its PAD scores in a separate section.

Usage::

    FACECHECK_PAD_CAPTURE_DIR=/path/to/captures \\
    python scripts/pad_eval.py --manifest /path/to/captures/manifest.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

import cv2
import numpy as np

from eval.metrics import (
    choose_apcer_threshold,
    choose_bpcer_threshold,
    pad_rates_at,
    rule_of_three,
    wilson_interval,
)
from facecheck.config import Settings
from facecheck.embeddings import FaceEmbedder, ModelNotFoundError, build_embedder
from facecheck.liveness import (
    LivenessDetector,
    LivenessModelNotFoundError,
    MiniFASNetLivenessDetector,
)
from facecheck.quality import QualityError

CAPTURE_DIR_ENV = "FACECHECK_PAD_CAPTURE_DIR"
BONA_FIDE = "bona_fide"
ATTACK_CLASSES = ("print", "phone_screen", "laptop_screen")
PRESENTATION_CLASSES = (BONA_FIDE, *ATTACK_CLASSES)
REQUIRED_COLUMNS = (
    "path",
    "subject_id",
    "session_id",
    "class",
    "attack_instrument",
    "capture_device",
    "condition",
    "consent",
)
MIN_ATTACKS_FOR_APCER_TARGET = 10
DEFAULT_REPORT_JSON = "pad_report.json"
DEFAULT_REPORT_MD = "pad_report.md"
SCHEMA_VERSION = 1


class ManifestError(ValueError):
    """Raised when the capture manifest is missing or malformed."""


class ImageLoadError(RuntimeError):
    """Raised when an evaluation image cannot be decoded."""


@dataclass(frozen=True)
class ManifestRow:
    relative_path: Path
    subject_id: str
    session_id: str
    presentation_class: str
    attack_instrument: str
    capture_device: str
    condition: str


@dataclass(frozen=True)
class Manifest:
    rows: list[ManifestRow]
    total_rows: int
    skipped_unconsented: int


@dataclass
class PresentationOutcome:
    presentation_class: str
    subject_id: str
    session_id: str
    detected: bool
    score: float | None
    diagnostic_score: float | None
    failure_reason: str | None
    error: str | None
    original_size: tuple[int, int] | None
    used_size: tuple[int, int] | None
    total_ms: float
    detection_ms: float
    liveness_ms: float


def _parse_consent(value: str | None) -> bool:
    return (value or "").strip().lower() in {"true", "1", "yes", "y"}


def _safe_relative_path(value: str, row_number: int) -> Path:
    text = value.strip().replace("\\", "/")
    if not text:
        raise ManifestError(f"row {row_number}: path is empty")
    relative = PurePosixPath(text)
    if relative.is_absolute() or ".." in relative.parts:
        raise ManifestError(
            f"row {row_number}: path must be relative to the capture directory"
        )
    return Path(*relative.parts)


def _required(values: dict[str, str], column: str, row_number: int) -> str:
    value = values.get(column, "").strip()
    if not value:
        raise ManifestError(f"row {row_number}: missing {column}")
    return value


def parse_manifest(path: str | Path) -> Manifest:
    """Read the capture manifest CSV and keep only consented presentations."""

    path = Path(path)
    if not path.is_file():
        raise ManifestError(f"manifest {path} does not exist")

    rows: list[ManifestRow] = []
    skipped_unconsented = 0
    total_rows = 0
    seen_paths: set[str] = set()

    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ManifestError(f"manifest {path} has no header row")
        fieldnames = {(name or "").strip() for name in reader.fieldnames}
        missing = [column for column in REQUIRED_COLUMNS if column not in fieldnames]
        if missing:
            raise ManifestError(
                f"manifest {path} is missing column(s): {', '.join(missing)}"
            )

        for row_number, raw in enumerate(reader, start=2):
            values = {
                (key or "").strip(): (value or "").strip()
                for key, value in raw.items()
                if key is not None
            }
            if not any(values.values()):
                continue
            total_rows += 1

            if not _parse_consent(values.get("consent")):
                skipped_unconsented += 1
                continue

            relative_path = _safe_relative_path(values.get("path", ""), row_number)
            path_key = relative_path.as_posix()
            if path_key in seen_paths:
                raise ManifestError(
                    f"row {row_number}: duplicate path {path_key!r}; "
                    "each presentation must be listed once"
                )
            seen_paths.add(path_key)

            presentation_class = _required(values, "class", row_number)
            if presentation_class not in PRESENTATION_CLASSES:
                raise ManifestError(
                    f"row {row_number}: unknown class {presentation_class!r}; "
                    f"expected one of {', '.join(PRESENTATION_CLASSES)}"
                )

            rows.append(
                ManifestRow(
                    relative_path=relative_path,
                    subject_id=_required(values, "subject_id", row_number),
                    session_id=_required(values, "session_id", row_number),
                    presentation_class=presentation_class,
                    attack_instrument=values.get("attack_instrument", ""),
                    capture_device=values.get("capture_device", ""),
                    condition=values.get("condition", ""),
                )
            )

    return Manifest(
        rows=rows,
        total_rows=total_rows,
        skipped_unconsented=skipped_unconsented,
    )


def load_image_bgr(path: str | Path) -> np.ndarray:
    """Decode an evaluation image without the API's 25 MP upload guard.

    ``facecheck.embeddings.decode_image_bgr`` rejects images above
    ``DEFAULT_MAX_PIXELS`` before decoding. Evaluation needs native phone
    photos, so it loads through OpenCV directly and optionally downscales with
    ``--max-side`` instead.
    """

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise ImageLoadError("could not decode evaluation image")
    return image


def downscale_max_side(image: np.ndarray, max_side: int) -> tuple[np.ndarray, float]:
    """Downscale so the longest side is at most ``max_side``.

    Returns the image and the width scale (``used_width / original_width``, or
    ``1.0`` when no resize happens). ``max_side <= 0`` keeps full resolution.
    """

    if max_side <= 0:
        return image, 1.0
    height, width = image.shape[:2]
    longest = max(height, width)
    if longest <= max_side:
        return image, 1.0
    scale = max_side / float(longest)
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    resized = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    return resized, size[0] / float(width)


def _diagnostic_score(
    embedder: FaceEmbedder | None,
    detector: LivenessDetector | None,
    image: np.ndarray,
) -> float | None:
    if embedder is None or detector is None:
        return None
    try:
        embedding = embedder.detect_and_embed(image)
        return float(detector.score_face(image, embedding).score)
    except Exception:  # noqa: BLE001 - diagnostic failures are reported as missing scores
        return None


def evaluate_rows(
    rows: Sequence[ManifestRow],
    data_dir: str | Path,
    *,
    embedder: FaceEmbedder,
    detector: LivenessDetector,
    diagnostic_embedder: FaceEmbedder | None = None,
    diagnostic_detector: LivenessDetector | None = None,
    max_side: int = 0,
) -> list[PresentationOutcome]:
    """Score every presentation once, keeping detection failures separate."""

    data_dir = Path(data_dir)
    outcomes: list[PresentationOutcome] = []

    for row in rows:
        started = time.perf_counter()
        try:
            image = load_image_bgr(data_dir / row.relative_path)
        except ImageLoadError as exc:
            outcomes.append(
                PresentationOutcome(
                    presentation_class=row.presentation_class,
                    subject_id=row.subject_id,
                    session_id=row.session_id,
                    detected=False,
                    score=None,
                    diagnostic_score=None,
                    failure_reason=None,
                    error=type(exc).__name__,
                    original_size=None,
                    used_size=None,
                    total_ms=(time.perf_counter() - started) * 1000.0,
                    detection_ms=0.0,
                    liveness_ms=0.0,
                )
            )
            continue

        original_size = (image.shape[1], image.shape[0])
        used, _scale = downscale_max_side(image, max_side)
        used_size = (used.shape[1], used.shape[0])

        detection_started = time.perf_counter()
        try:
            embedding = embedder.detect_and_embed(used)
        except QualityError as exc:
            detection_ms = (time.perf_counter() - detection_started) * 1000.0
            outcomes.append(
                PresentationOutcome(
                    presentation_class=row.presentation_class,
                    subject_id=row.subject_id,
                    session_id=row.session_id,
                    detected=False,
                    score=None,
                    diagnostic_score=_diagnostic_score(
                        diagnostic_embedder, diagnostic_detector, used
                    ),
                    failure_reason=exc.reason,
                    error=None,
                    original_size=original_size,
                    used_size=used_size,
                    total_ms=(time.perf_counter() - started) * 1000.0,
                    detection_ms=detection_ms,
                    liveness_ms=0.0,
                )
            )
            continue
        except Exception as exc:  # noqa: BLE001 - report unexpected backend failures
            detection_ms = (time.perf_counter() - detection_started) * 1000.0
            outcomes.append(
                PresentationOutcome(
                    presentation_class=row.presentation_class,
                    subject_id=row.subject_id,
                    session_id=row.session_id,
                    detected=False,
                    score=None,
                    diagnostic_score=None,
                    failure_reason=None,
                    error=type(exc).__name__,
                    original_size=original_size,
                    used_size=used_size,
                    total_ms=(time.perf_counter() - started) * 1000.0,
                    detection_ms=detection_ms,
                    liveness_ms=0.0,
                )
            )
            continue

        detection_ms = (time.perf_counter() - detection_started) * 1000.0
        liveness_started = time.perf_counter()
        try:
            result = detector.score_face(used, embedding)
        except Exception as exc:  # noqa: BLE001 - report unexpected backend failures
            outcomes.append(
                PresentationOutcome(
                    presentation_class=row.presentation_class,
                    subject_id=row.subject_id,
                    session_id=row.session_id,
                    detected=True,
                    score=None,
                    diagnostic_score=None,
                    failure_reason=None,
                    error=type(exc).__name__,
                    original_size=original_size,
                    used_size=used_size,
                    total_ms=(time.perf_counter() - started) * 1000.0,
                    detection_ms=detection_ms,
                    liveness_ms=(time.perf_counter() - liveness_started) * 1000.0,
                )
            )
            continue

        outcomes.append(
            PresentationOutcome(
                presentation_class=row.presentation_class,
                subject_id=row.subject_id,
                session_id=row.session_id,
                detected=True,
                score=float(result.score),
                diagnostic_score=None,
                failure_reason=None,
                error=None,
                original_size=original_size,
                used_size=used_size,
                total_ms=(time.perf_counter() - started) * 1000.0,
                detection_ms=detection_ms,
                liveness_ms=(time.perf_counter() - liveness_started) * 1000.0,
            )
        )

    return outcomes


def _scores_by_class(outcomes: Sequence[PresentationOutcome]) -> dict[str, list[float]]:
    scores = {presentation_class: [] for presentation_class in PRESENTATION_CLASSES}
    for outcome in outcomes:
        if outcome.score is not None:
            scores[outcome.presentation_class].append(float(outcome.score))
    return scores


def _class_summary(
    outcomes: Sequence[PresentationOutcome], presentation_class: str
) -> dict[str, Any]:
    items = [
        outcome
        for outcome in outcomes
        if outcome.presentation_class == presentation_class
    ]
    presentations = len(items)
    load_failures = sum(1 for item in items if item.original_size is None)
    detection_failures = sum(1 for item in items if item.failure_reason is not None)
    detection_attempts = presentations - load_failures
    return {
        "presentations": presentations,
        "scored": sum(1 for item in items if item.score is not None),
        "load_failures": load_failures,
        "detection_failures": detection_failures,
        "detection_attempts": detection_attempts,
        "detection_failure_rate": (
            detection_failures / detection_attempts if detection_attempts else None
        ),
        "errors": sum(1 for item in items if item.error is not None),
    }


def _worst_apcer(per_class: dict[str, dict[str, Any]]) -> tuple[str | None, float | None]:
    worst_name: str | None = None
    worst_value: float | None = None
    for name, block in per_class.items():
        value = block.get("apcer")
        if value is not None and (worst_value is None or value > worst_value):
            worst_name = name
            worst_value = float(value)
    return worst_name, worst_value


def _rates_block(
    bona_scores: Sequence[float],
    attack_scores: dict[str, list[float]],
    threshold: float,
) -> dict[str, Any]:
    all_attacks = [
        score for name in ATTACK_CLASSES for score in attack_scores[name]
    ]
    overall = pad_rates_at(bona_scores, all_attacks, threshold)

    per_class: dict[str, dict[str, Any]] = {}
    for presentation_class in ATTACK_CLASSES:
        rates = pad_rates_at(bona_scores, attack_scores[presentation_class], threshold)
        per_class[presentation_class] = {
            "presentations": rates.attack_presentations,
            "accepted": rates.attack_accepted,
            "apcer": rates.apcer,
            "apcer_ci_95_wilson": wilson_interval(
                rates.attack_accepted, rates.attack_presentations
            ),
            "apcer_zero_error_rule_of_three": (
                rule_of_three(rates.attack_presentations)
                if rates.attack_presentations and rates.attack_accepted == 0
                else None
            ),
        }

    worst_name, worst_value = _worst_apcer(per_class)
    bpcer_interval = wilson_interval(overall.bona_rejected, overall.bona_presentations)
    apcer_interval = wilson_interval(
        overall.attack_accepted, overall.attack_presentations
    )
    return {
        "threshold": float(threshold),
        "bona_presentations": overall.bona_presentations,
        "bona_rejected": overall.bona_rejected,
        "bpcer": overall.bpcer,
        "bpcer_ci_95_wilson": bpcer_interval,
        "bpcer_zero_error_rule_of_three": (
            rule_of_three(overall.bona_presentations)
            if overall.bona_presentations and overall.bona_rejected == 0
            else None
        ),
        "attack_presentations": overall.attack_presentations,
        "attack_accepted": overall.attack_accepted,
        "apcer": overall.apcer,
        "apcer_ci_95_wilson": apcer_interval,
        "apcer_zero_error_rule_of_three": (
            rule_of_three(overall.attack_presentations)
            if overall.attack_presentations and overall.attack_accepted == 0
            else None
        ),
        "apcer_per_class": per_class,
        "apcer_worst_class": worst_name,
        "apcer_worst": worst_value,
        "iapmr": None,
        "iapmr_status": "not_measured_without_matcher",
    }


def _compact_rates(
    bona_scores: Sequence[float],
    attack_scores: dict[str, list[float]],
    threshold: float,
) -> dict[str, Any]:
    all_attacks = [
        score for name in ATTACK_CLASSES for score in attack_scores[name]
    ]
    overall = pad_rates_at(bona_scores, all_attacks, threshold)
    per_class = {
        presentation_class: pad_rates_at(
            bona_scores, attack_scores[presentation_class], threshold
        )
        for presentation_class in ATTACK_CLASSES
    }
    worst_name: str | None = None
    worst_value: float | None = None
    for name, rates in per_class.items():
        if rates.apcer is not None and (worst_value is None or rates.apcer > worst_value):
            worst_name = name
            worst_value = rates.apcer
    return {
        "threshold": float(threshold),
        "bpcer": overall.bpcer,
        "apcer": overall.apcer,
        "apcer_worst_class": worst_name,
        "apcer_worst": worst_value,
    }


def _score_stats(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0, "mean": None, "median": None, "min": None, "max": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "n": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def _latency_summary(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0, "mean": None, "p50": None, "p95": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "n": int(array.size),
        "mean": float(array.mean()),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
    }


def build_report(
    outcomes: Sequence[PresentationOutcome],
    *,
    manifest: Manifest,
    production_threshold: float,
    diagnostic_threshold: float | None,
    diagnostic_enabled: bool,
    max_side: int,
    bpcer_target: float,
    apcer_target: float,
    embedder_version: str,
    detector_version: str,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Build an aggregate-only report dictionary from scored presentations."""

    scores = _scores_by_class(outcomes)
    bona_scores = scores[BONA_FIDE]
    all_attack_scores = [score for name in ATTACK_CLASSES for score in scores[name]]
    rates = _rates_block(bona_scores, scores, production_threshold)

    class_counts = Counter(outcome.presentation_class for outcome in outcomes)
    manifest_block = {
        "data_rows": manifest.total_rows,
        "included_presentations": len(manifest.rows),
        "skipped_unconsented": manifest.skipped_unconsented,
        "class_counts": {
            presentation_class: class_counts.get(presentation_class, 0)
            for presentation_class in PRESENTATION_CLASSES
        },
        "subjects": len({row.subject_id for row in manifest.rows}),
        "sessions": len({row.session_id for row in manifest.rows}),
        "attack_instruments": dict(
            sorted(
                Counter(
                    row.attack_instrument
                    for row in manifest.rows
                    if row.presentation_class != BONA_FIDE and row.attack_instrument
                ).items()
            )
        ),
        "capture_devices": dict(
            sorted(
                Counter(
                    row.capture_device for row in manifest.rows if row.capture_device
                ).items()
            )
        ),
    }

    bpcer_threshold = choose_bpcer_threshold(bona_scores, bpcer_target)
    bpcer_point = (
        _rates_block(bona_scores, scores, bpcer_threshold)
        if bpcer_threshold is not None
        else None
    )
    bpcer_status = "selected" if bpcer_threshold is not None else "no_bona_fide_scores"

    apcer_threshold: float | None = None
    apcer_point: dict[str, Any] | None = None
    if len(all_attack_scores) < MIN_ATTACKS_FOR_APCER_TARGET:
        apcer_status = (
            f"insufficient_attack_scores (need at least "
            f"{MIN_ATTACKS_FOR_APCER_TARGET})"
        )
    else:
        apcer_threshold = choose_apcer_threshold(all_attack_scores, apcer_target)
        apcer_point = (
            _rates_block(bona_scores, scores, apcer_threshold)
            if apcer_threshold is not None
            else None
        )
        apcer_status = "selected" if apcer_threshold is not None else "no_attack_scores"

    sweep_values = {round(0.05 * index, 4) for index in range(1, 20)}
    sweep_values |= {round(score, 4) for score in all_attack_scores + bona_scores}
    for value in (production_threshold, bpcer_threshold, apcer_threshold):
        if value is not None:
            sweep_values.add(round(float(value), 4))
    threshold_sweep = [
        _compact_rates(bona_scores, scores, threshold)
        for threshold in sorted(sweep_values)
    ]

    diagnostic_classes: dict[str, dict[str, Any]] = {}
    for presentation_class in PRESENTATION_CLASSES:
        failed = [
            outcome
            for outcome in outcomes
            if outcome.presentation_class == presentation_class
            and outcome.failure_reason is not None
        ]
        recovered = [
            float(outcome.diagnostic_score)
            for outcome in failed
            if outcome.diagnostic_score is not None
        ]
        diagnostic_classes[presentation_class] = {
            "failed_production_detection": len(failed),
            "recovered_with_diagnostic_threshold": len(recovered),
            "still_failed": len(failed) - len(recovered),
            "scores": _score_stats(recovered),
        }

    scored = [outcome for outcome in outcomes if outcome.score is not None]
    loaded = [outcome for outcome in outcomes if outcome.original_size is not None]
    downscaled = [
        outcome for outcome in loaded if outcome.original_size != outcome.used_size
    ]
    latencies = {
        "full_pipeline": _latency_summary([outcome.total_ms for outcome in scored]),
        "detection_and_embedding": _latency_summary(
            [outcome.detection_ms for outcome in scored]
        ),
        "liveness_scoring": _latency_summary(
            [outcome.liveness_ms for outcome in scored]
        ),
    }

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at
        or datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
        "protocol": {
            "metric_family": "ISO/IEC 30107-3",
            "unit": "presentation",
            "acceptance_rule": "pad_score >= threshold",
            "production_detector_threshold": production_threshold,
            "diagnostic_detector_threshold": (
                diagnostic_threshold if diagnostic_enabled else None
            ),
            "max_side": max_side,
            "bpcer_target": bpcer_target,
            "apcer_target": apcer_target,
            "min_attacks_for_apcer_target": MIN_ATTACKS_FOR_APCER_TARGET,
        },
        "model_versions": {
            "embedder": embedder_version,
            "liveness": detector_version,
        },
        "manifest": manifest_block,
        "presentation_summary": {
            presentation_class: _class_summary(outcomes, presentation_class)
            for presentation_class in PRESENTATION_CLASSES
        },
        "rates": rates,
        "operating_points": {
            "bpcer_target": bpcer_target,
            "bpcer_target_status": bpcer_status,
            "bpcer_target_threshold": bpcer_threshold,
            "bpcer_target_rates": bpcer_point,
            "apcer_target": apcer_target,
            "apcer_target_status": apcer_status,
            "apcer_target_threshold": apcer_threshold,
            "apcer_target_rates": apcer_point,
        },
        "threshold_sweep": threshold_sweep,
        "diagnostic": {
            "enabled": diagnostic_enabled,
            "note": (
                "Diagnostic scores are excluded from production APCER/BPCER; "
                "they only show whether a lower face-detection threshold recovers "
                "rows that failed the production gate."
            ),
            "production_detector_threshold": production_threshold,
            "diagnostic_detector_threshold": (
                diagnostic_threshold if diagnostic_enabled else None
            ),
            "per_class": diagnostic_classes,
        },
        "latency_ms": latencies,
        "resolution": {
            "max_side_setting": max_side,
            "loaded_presentations": len(loaded),
            "downscaled_presentations": len(downscaled),
            "max_original_side_px": (
                max(max(outcome.original_size or (0, 0)) for outcome in loaded)
                if loaded
                else None
            ),
            "max_used_side_px": (
                max(max(outcome.used_size or (0, 0)) for outcome in loaded)
                if loaded
                else None
            ),
        },
        "caveats": [
            "One row is one presentation. Frames from one video or session are "
            "one presentation for these metrics unless the manifest lists each "
            "as an independent presentation.",
            "Calibration and test sets must be separated by session_id (or day), "
            "not by randomly splitting frames.",
            "BPCER from a small or single-subject capture does not generalize "
            "beyond the subjects, devices, and lighting in that capture.",
            "Zero observed errors do not imply zero risk; see the rule-of-three "
            "upper bounds in the rates block.",
            "IAPMR is not measured because the face matcher is not wired into "
            "this harness.",
            "Detection failures are excluded from APCER and BPCER and reported "
            "separately per presentation class.",
            "The harness loads images with OpenCV and is not subject to the API's "
            "25 MP upload limit; use --max-side to compare downscaled latency.",
        ],
    }


def _fmt_pct(value: float | None, digits: int = 2) -> str:
    return "n/a" if value is None else f"{value * 100:.{digits}f}%"


def _fmt_num(value: float | None, digits: int = 4) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _fmt_interval(interval: tuple[float, float] | None) -> str:
    if interval is None:
        return "n/a"
    return f"{interval[0] * 100:.2f}% - {interval[1] * 100:.2f}%"


def render_markdown(report: dict[str, Any]) -> str:
    """Render the aggregate report as Markdown."""

    rates = report["rates"]
    operating = report["operating_points"]
    summary = report["presentation_summary"]
    protocol = report["protocol"]
    manifest = report["manifest"]

    lines = [
        "# PAD capture evaluation report",
        "",
        f"Generated: {report['generated_at']}",
        "",
        "This report contains aggregates only. It does not contain image paths, "
        "subject identifiers, or per-presentation scores.",
        "",
        "## Protocol",
        "",
        f"- Metric family: {protocol['metric_family']}",
        f"- Unit: {protocol['unit']} (a row is a presentation, not a frame)",
        f"- Accept rule: `{protocol['acceptance_rule']}`",
        f"- Production detector threshold: "
        f"{protocol['production_detector_threshold']:.3f}",
        f"- Diagnostic detector threshold: "
        f"{_fmt_num(protocol['diagnostic_detector_threshold'], 3)}",
        f"- Evaluation max side: "
        f"{protocol['max_side']} (0 = full resolution)",
        f"- BPCER target: {_fmt_pct(protocol['bpcer_target'])}",
        f"- APCER target: {_fmt_pct(protocol['apcer_target'])}",
        f"- Embedder: `{report['model_versions']['embedder']}`",
        f"- Liveness detector: `{report['model_versions']['liveness']}`",
        "",
        "## Capture summary",
        "",
        f"- Manifest data rows: {manifest['data_rows']}",
        f"- Included presentations: {manifest['included_presentations']}",
        f"- Skipped, consent not true: {manifest['skipped_unconsented']}",
        f"- Subjects: {manifest['subjects']}",
        f"- Sessions: {manifest['sessions']}",
        "",
        "| Class | Presentations | Scored | Detection failures | "
        "Detection failure rate | Load failures |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for presentation_class in PRESENTATION_CLASSES:
        block = summary[presentation_class]
        lines.append(
            f"| {presentation_class} | {block['presentations']} | "
            f"{block['scored']} | {block['detection_failures']} | "
            f"{_fmt_pct(block['detection_failure_rate'])} | "
            f"{block['load_failures']} |"
        )

    lines += [
        "",
        f"## Rates at production threshold {rates['threshold']:.3f}",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| BPCER | **{_fmt_pct(rates['bpcer'])}** "
        f"({rates['bona_rejected']}/{rates['bona_presentations']} bona-fide "
        "presentations) |",
        f"| BPCER 95% Wilson interval | "
        f"{_fmt_interval(rates['bpcer_ci_95_wilson'])} |",
        f"| BPCER zero-error 95% upper bound (rule of three) | "
        f"{_fmt_pct(rates['bpcer_zero_error_rule_of_three'])} |",
        f"| APCER overall | **{_fmt_pct(rates['apcer'])}** "
        f"({rates['attack_accepted']}/{rates['attack_presentations']} attack "
        "presentations) |",
        f"| APCER 95% Wilson interval | "
        f"{_fmt_interval(rates['apcer_ci_95_wilson'])} |",
        f"| APCER zero-error 95% upper bound (rule of three) | "
        f"{_fmt_pct(rates['apcer_zero_error_rule_of_three'])} |",
        f"| Worst-case APCER | {_fmt_pct(rates['apcer_worst'])} "
        f"(`{rates['apcer_worst_class'] or 'n/a'}`) |",
        f"| IAPMR | n/a ({rates['iapmr_status']}) |",
        "",
        "Per attack class:",
        "",
        "| Class | Presentations | Accepted | APCER | 95% Wilson | "
        "Zero-error upper bound |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for presentation_class in ATTACK_CLASSES:
        block = rates["apcer_per_class"][presentation_class]
        lines.append(
            f"| {presentation_class} | {block['presentations']} | "
            f"{block['accepted']} | {_fmt_pct(block['apcer'])} | "
            f"{_fmt_interval(block['apcer_ci_95_wilson'])} | "
            f"{_fmt_pct(block['apcer_zero_error_rule_of_three'])} |"
        )

    lines += [
        "",
        "## Operating points",
        "",
        f"BPCER target {_fmt_pct(operating['bpcer_target'])}: "
        f"{operating['bpcer_target_status']}.",
        "",
    ]
    bpcer_point = operating["bpcer_target_rates"]
    if bpcer_point is not None:
        lines += [
            "| Threshold | BPCER | APCER | Worst-case APCER |",
            "| --- | --- | --- | --- |",
            f"| {bpcer_point['threshold']:.4f} | {_fmt_pct(bpcer_point['bpcer'])} | "
            f"{_fmt_pct(bpcer_point['apcer'])} | {_fmt_pct(bpcer_point['apcer_worst'])} |",
        ]
    else:
        lines.append("No bona-fide scores were available to select a threshold.")

    lines += [
        "",
        f"APCER target {_fmt_pct(operating['apcer_target'])}: "
        f"{operating['apcer_target_status']}.",
        "",
    ]
    apcer_point = operating["apcer_target_rates"]
    if apcer_point is not None:
        lines += [
            "| Threshold | BPCER | APCER | Worst-case APCER |",
            "| --- | --- | --- | --- |",
            f"| {apcer_point['threshold']:.4f} | {_fmt_pct(apcer_point['bpcer'])} | "
            f"{_fmt_pct(apcer_point['apcer'])} | {_fmt_pct(apcer_point['apcer_worst'])} |",
        ]
    else:
        lines.append(
            f"At least {protocol['min_attacks_for_apcer_target']} scored attack "
            "presentations are required for this operating point."
        )

    lines += [
        "",
        "## Threshold sweep",
        "",
        "| Threshold | BPCER | APCER | Worst-case APCER |",
        "| --- | --- | --- | --- |",
    ]
    for point in report["threshold_sweep"]:
        lines.append(
            f"| {point['threshold']:.4f} | {_fmt_pct(point['bpcer'])} | "
            f"{_fmt_pct(point['apcer'])} | {_fmt_pct(point['apcer_worst'])} |"
        )

    diagnostic = report["diagnostic"]
    lines += [
        "",
        "## Diagnostic pass (excluded from production metrics)",
        "",
        f"- Enabled: {diagnostic['enabled']}",
        f"- Production detector threshold: "
        f"{diagnostic['production_detector_threshold']:.3f}",
        f"- Diagnostic detector threshold: "
        f"{_fmt_num(diagnostic['diagnostic_detector_threshold'], 3)}",
        "",
        diagnostic["note"],
        "",
        "| Class | Failed production detection | Recovered | Still failed | "
        "Recovered score mean | Median | Min | Max |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for presentation_class in PRESENTATION_CLASSES:
        block = diagnostic["per_class"][presentation_class]
        stats = block["scores"]
        lines.append(
            f"| {presentation_class} | {block['failed_production_detection']} | "
            f"{block['recovered_with_diagnostic_threshold']} | "
            f"{block['still_failed']} | {_fmt_num(stats['mean'])} | "
            f"{_fmt_num(stats['median'])} | {_fmt_num(stats['min'])} | "
            f"{_fmt_num(stats['max'])} |"
        )

    lines += [
        "",
        "## Latency",
        "",
        "| Stage | n | Mean | p50 | p95 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for name, label in (
        ("full_pipeline", "Load + detect/embed + PAD"),
        ("detection_and_embedding", "Detect + align + embed"),
        ("liveness_scoring", "MiniFASNet score"),
    ):
        block = report["latency_ms"][name]
        lines.append(
            f"| {label} | {block['n']} | {_fmt_num(block['mean'], 2)} ms | "
            f"{_fmt_num(block['p50'], 2)} ms | {_fmt_num(block['p95'], 2)} ms |"
        )

    resolution = report["resolution"]
    lines += [
        "",
        "## Resolution",
        "",
        f"- Max side setting: {resolution['max_side_setting']}",
        f"- Loaded presentations: {resolution['loaded_presentations']}",
        f"- Downscaled presentations: {resolution['downscaled_presentations']}",
        f"- Largest original side: {resolution['max_original_side_px']} px",
        f"- Largest used side: {resolution['max_used_side_px']} px",
        "",
        "## Caveats",
        "",
    ]
    for caveat in report["caveats"]:
        lines.append(f"- {caveat}")
    return "\n".join(lines) + "\n"


def run_evaluation(
    manifest_path: str | Path,
    data_dir: str | Path,
    *,
    embedder: FaceEmbedder,
    detector: LivenessDetector,
    diagnostic_embedder: FaceEmbedder | None = None,
    diagnostic_detector: LivenessDetector | None = None,
    production_threshold: float = 0.9,
    diagnostic_threshold: float | None = 0.5,
    max_side: int = 0,
    bpcer_target: float = 0.05,
    apcer_target: float = 0.10,
) -> dict[str, Any]:
    """Parse the manifest, score every presentation, and build the report."""

    manifest = parse_manifest(manifest_path)
    if not manifest.rows:
        raise ManifestError(
            "manifest has no usable presentations: every row was skipped or the "
            "file is empty"
        )

    diagnostic_enabled = (
        diagnostic_embedder is not None
        and diagnostic_detector is not None
        and diagnostic_threshold is not None
        and diagnostic_threshold < production_threshold
    )
    outcomes = evaluate_rows(
        manifest.rows,
        data_dir,
        embedder=embedder,
        detector=detector,
        diagnostic_embedder=diagnostic_embedder if diagnostic_enabled else None,
        diagnostic_detector=diagnostic_detector if diagnostic_enabled else None,
        max_side=max_side,
    )
    return build_report(
        outcomes,
        manifest=manifest,
        production_threshold=production_threshold,
        diagnostic_threshold=diagnostic_threshold,
        diagnostic_enabled=diagnostic_enabled,
        max_side=max_side,
        bpcer_target=bpcer_target,
        apcer_target=apcer_target,
        embedder_version=embedder.model_version,
        detector_version=detector.model_version,
    )


def _validate_threshold(name: str, value: float) -> None:
    if not 0.0 < value <= 1.0:
        raise SystemExit(f"{name} must be in (0, 1], got {value}")


def _validate_target(name: str, value: float) -> None:
    if not 0.0 <= value <= 1.0:
        raise SystemExit(f"{name} must be in [0, 1], got {value}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Score a consented PAD capture directory and write reports"
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help=f"capture root (default: ${CAPTURE_DIR_ENV} or the manifest directory)",
    )
    parser.add_argument("--detector-threshold", type=float, default=0.9)
    parser.add_argument("--diagnostic-detector-threshold", type=float, default=0.5)
    parser.add_argument(
        "--max-side",
        type=int,
        default=0,
        help="downscale so the longest side is at most this; 0 keeps full resolution",
    )
    parser.add_argument("--bpcer-target", type=float, default=0.05)
    parser.add_argument("--apcer-target", type=float, default=0.10)
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument("--out-md", type=Path, default=None)
    args = parser.parse_args(argv)

    _validate_threshold("--detector-threshold", args.detector_threshold)
    _validate_threshold(
        "--diagnostic-detector-threshold", args.diagnostic_detector_threshold
    )
    _validate_target("--bpcer-target", args.bpcer_target)
    _validate_target("--apcer-target", args.apcer_target)
    if args.max_side < 0:
        raise SystemExit("--max-side must be >= 0")

    if not args.manifest.is_file():
        print(f"missing manifest {args.manifest}", file=sys.stderr)
        return 2

    data_dir = args.data_dir or os.environ.get(CAPTURE_DIR_ENV)
    data_dir = Path(data_dir) if data_dir else args.manifest.parent
    if not data_dir.is_dir():
        print(f"capture directory {data_dir} does not exist", file=sys.stderr)
        return 2

    try:
        manifest = parse_manifest(args.manifest)
    except ManifestError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if not manifest.rows:
        print(
            "manifest has no usable presentations: every row was skipped or the "
            "file is empty",
            file=sys.stderr,
        )
        return 2

    settings = Settings(
        _env_file=None, detector_score_threshold=args.detector_threshold
    )
    try:
        embedder = build_embedder(settings)
    except ModelNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    try:
        detector = MiniFASNetLivenessDetector.from_settings(settings)
    except LivenessModelNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    diagnostic_embedder = None
    diagnostic_detector = None
    if args.diagnostic_detector_threshold < args.detector_threshold:
        diagnostic_settings = settings.model_copy(
            update={
                "detector_score_threshold": args.diagnostic_detector_threshold,
            }
        )
        diagnostic_embedder = build_embedder(diagnostic_settings)
        diagnostic_detector = MiniFASNetLivenessDetector.from_settings(
            diagnostic_settings
        )

    try:
        report = run_evaluation(
            args.manifest,
            data_dir,
            embedder=embedder,
            detector=detector,
            diagnostic_embedder=diagnostic_embedder,
            diagnostic_detector=diagnostic_detector,
            production_threshold=args.detector_threshold,
            diagnostic_threshold=args.diagnostic_detector_threshold,
            max_side=args.max_side,
            bpcer_target=args.bpcer_target,
            apcer_target=args.apcer_target,
        )
    except ManifestError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    out_json = args.out_json or args.manifest.parent / DEFAULT_REPORT_JSON
    out_md = args.out_md or args.manifest.parent / DEFAULT_REPORT_MD
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    out_md.write_text(render_markdown(report), encoding="utf-8")

    rates = report["rates"]
    print(
        f"presentations: {report['manifest']['included_presentations']} included, "
        f"{report['manifest']['skipped_unconsented']} skipped for consent"
    )
    print(
        f"scored: {rates['bona_presentations']} bona-fide / "
        f"{rates['attack_presentations']} attack"
    )
    print(
        f"BPCER at {rates['threshold']:.3f}: {_fmt_pct(rates['bpcer'])}; "
        f"APCER overall: {_fmt_pct(rates['apcer'])}; "
        f"worst class: {_fmt_pct(rates['apcer_worst'])} "
        f"({rates['apcer_worst_class'] or 'n/a'})"
    )
    print(f"wrote {out_json}")
    print(f"wrote {out_md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
