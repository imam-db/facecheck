"""Metric helpers for the LFW and PAD evaluation harnesses.

The functions here are intentionally small and pure so they can be unit
tested on synthetic scores without touching the models or the dataset.

PAD helpers follow ISO/IEC 30107-3 terminology. A presentation is *accepted*
when its PAD real-class probability is at or above the threshold, so an attack
accepted at that threshold contributes to APCER.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Metrics:
    threshold: float
    tpr: float
    far: float
    frr: float
    accuracy: float
    true_accepts: int = 0
    false_accepts: int = 0
    true_rejects: int = 0
    false_rejects: int = 0
    positives: int = 0
    negatives: int = 0


def scores_at(scores: np.ndarray, labels: np.ndarray, threshold: float) -> Metrics:
    predictions = scores >= threshold
    positives = labels == 1
    negatives = ~positives
    true_accepts = int(np.sum(predictions & positives))
    false_accepts = int(np.sum(predictions & negatives))
    true_rejects = int(np.sum(~predictions & negatives))
    false_rejects = int(np.sum(~predictions & positives))
    n_positives = int(np.sum(positives))
    n_negatives = int(np.sum(negatives))
    tpr = true_accepts / max(n_positives, 1)
    far = false_accepts / max(n_negatives, 1)
    total = max(int(scores.size), 1)
    return Metrics(
        threshold=float(threshold),
        tpr=tpr,
        far=far,
        frr=1.0 - tpr,
        accuracy=(true_accepts + true_rejects) / total,
        true_accepts=true_accepts,
        false_accepts=false_accepts,
        true_rejects=true_rejects,
        false_rejects=false_rejects,
        positives=n_positives,
        negatives=n_negatives,
    )


def choose_far_threshold(
    scores: np.ndarray, labels: np.ndarray, target_far: float
) -> tuple[float | None, Metrics | None]:
    """Pick the highest-TPR threshold whose FAR is within the target."""

    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels)
    candidates = np.unique(np.concatenate([scores, [0.0, 1.0]]))
    positives = int(np.sum(labels == 1))
    negatives = int(np.sum(labels == 0))
    if not positives or not negatives:
        return None, None

    for threshold in candidates:
        metrics = scores_at(scores, labels, float(threshold))
        if metrics.far <= target_far:
            return float(threshold), metrics

    threshold = float(candidates[-1])
    return threshold, scores_at(scores, labels, threshold)


def choose_youden(scores: np.ndarray, labels: np.ndarray) -> tuple[float, Metrics]:
    candidates = np.unique(np.concatenate([scores, [0.0, 1.0]]))
    best_threshold = 0.0
    best_metrics = scores_at(scores, labels, 0.0)
    best_index = best_metrics.tpr - best_metrics.far
    for threshold in candidates:
        metrics = scores_at(scores, labels, float(threshold))
        index = metrics.tpr - metrics.far
        if index > best_index:
            best_index, best_threshold, best_metrics = index, float(threshold), metrics
    return best_threshold, best_metrics


def roc_curve(
    scores: np.ndarray, labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    try:
        from sklearn.metrics import roc_curve as sklearn_roc_curve
    except ImportError:
        return None
    fpr, tpr, thresholds = sklearn_roc_curve(labels, scores)
    return fpr, tpr, thresholds


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float | None:
    if len(np.unique(labels)) < 2:
        return None
    try:
        from sklearn.metrics import roc_auc_score
    except ImportError:
        return None
    return float(roc_auc_score(labels, scores))


def choose_eer(scores: np.ndarray, labels: np.ndarray) -> tuple[float | None, float | None]:
    if len(np.unique(labels)) < 2:
        return None, None
    roc = roc_curve(scores, labels)
    if roc is None:
        return None, None
    fpr, tpr, thresholds = roc
    fnr = 1.0 - tpr
    index = int(np.argmin(np.abs(fpr - fnr)))
    eer = float((fpr[index] + fnr[index]) / 2.0)
    threshold = float(thresholds[index])
    if not np.isfinite(threshold):
        threshold = float(np.max(scores))
    return threshold, eer


def rule_of_three(impostors: int) -> float | None:
    """95% upper bound on FAR when zero false accepts are observed.

    For ``n`` independent impostor trials and zero accepts, the exact
    one-sided 95% upper bound is ``1 - 0.05 ** (1/n)``, which is approximately
    ``3/n`` for large ``n`` (the "rule of three").
    """

    if impostors <= 0:
        return None
    return 1.0 - 0.05 ** (1.0 / impostors)


def wilson_interval(successes: int, trials: int, z: float = 1.96) -> tuple[float, float] | None:
    """Wilson score interval for a binomial proportion."""

    if trials <= 0:
        return None
    p = successes / trials
    denominator = 1.0 + z * z / trials
    centre = (p + z * z / (2.0 * trials)) / denominator
    margin = (
        z
        * math.sqrt(p * (1.0 - p) / trials + z * z / (4.0 * trials * trials))
        / denominator
    )
    return max(0.0, centre - margin), min(1.0, centre + margin)


def generalized_frr(
    skipped_genuine: int, false_rejects: int, attempted_genuine: int
) -> float | None:
    """FRR that counts skipped genuine pairs as failures."""

    if attempted_genuine <= 0:
        return None
    return (skipped_genuine + false_rejects) / attempted_genuine


def failure_to_acquire(skipped_pairs: int, attempted_pairs: int) -> float | None:
    if attempted_pairs <= 0:
        return None
    return skipped_pairs / attempted_pairs


def apcer(attack_accepted: int, attack_presentations: int) -> float | None:
    """Attack Presentation Classification Error Rate (ISO/IEC 30107-3).

    The fraction of attack presentations that the PAD accepted as bona fide.
    Returns ``None`` when no attack presentations were scored.
    """

    if attack_presentations <= 0:
        return None
    return attack_accepted / attack_presentations


def bpcer(bona_rejected: int, bona_presentations: int) -> float | None:
    """Bona Fide Presentation Classification Error Rate (ISO/IEC 30107-3).

    The fraction of bona-fide presentations that the PAD rejected. Returns
    ``None`` when no bona-fide presentations were scored.
    """

    if bona_presentations <= 0:
        return None
    return bona_rejected / bona_presentations


def iapmr(attacks_that_matched: int, attack_presentations: int) -> float | None:
    """Impostor Attack Presentation Match Rate (ISO/IEC 30107-3).

    This harness has no matcher wired into it, so it reports IAPMR as ``None``.
    The helper is still the named accepted/total rate for callers that do have
    match outcomes, and it is numerically the same as the PAD accept rate on
    attacks when the stimulus is a PAD score.
    """

    if attack_presentations <= 0:
        return None
    return attacks_that_matched / attack_presentations


@dataclass(frozen=True)
class PadRates:
    """PAD accept/reject counts and rates for one threshold."""

    threshold: float
    bona_presentations: int
    bona_rejected: int
    bpcer: float | None
    attack_presentations: int
    attack_accepted: int
    apcer: float | None


def _score_array(scores: Iterable[float] | np.ndarray) -> np.ndarray:
    values = scores if isinstance(scores, np.ndarray) else list(scores)
    return np.asarray(values, dtype=np.float64).reshape(-1)


def pad_rates_at(
    bona_scores: Iterable[float] | np.ndarray,
    attack_scores: Iterable[float] | np.ndarray,
    threshold: float,
) -> PadRates:
    """Count accepted/rejected presentations at ``threshold``.

    A presentation is accepted when its real-class probability is
    ``>= threshold``. That is the conservative direction for a PAD: an attack
    that scores exactly at the threshold is counted as accepted.
    """

    bona = _score_array(bona_scores)
    attacks = _score_array(attack_scores)
    bona_rejected = int(np.sum(bona < threshold))
    attack_accepted = int(np.sum(attacks >= threshold))
    return PadRates(
        threshold=float(threshold),
        bona_presentations=int(bona.size),
        bona_rejected=bona_rejected,
        bpcer=bpcer(bona_rejected, int(bona.size)),
        attack_presentations=int(attacks.size),
        attack_accepted=attack_accepted,
        apcer=apcer(attack_accepted, int(attacks.size)),
    )


def pad_threshold_sweep(
    bona_scores: Iterable[float] | np.ndarray,
    attack_scores: Iterable[float] | np.ndarray,
    thresholds: Iterable[float],
) -> list[PadRates]:
    """Evaluate unique thresholds in ascending order, for a report table."""

    return [
        pad_rates_at(bona_scores, attack_scores, float(threshold))
        for threshold in sorted({float(value) for value in thresholds})
    ]


def choose_bpcer_threshold(
    bona_scores: Iterable[float] | np.ndarray, bpcer_target: float
) -> float | None:
    """Highest observed threshold whose BPCER is within ``bpcer_target``.

    Raising the threshold rejects more bona-fide presentations, so BPCER is
    non-decreasing in the threshold. Walking the observed scores from high to
    low returns the most attack-resistant operating point that still meets the
    bona-fide error budget. The result is capped at the highest observed score
    rather than returning an unbounded value.
    """

    if not 0.0 <= bpcer_target <= 1.0:
        raise ValueError("bpcer_target must be between 0 and 1")
    scores = _score_array(bona_scores)
    if scores.size == 0:
        return None

    for threshold in np.unique(scores)[::-1]:
        rate = bpcer(int(np.sum(scores < threshold)), int(scores.size))
        if rate is not None and rate <= bpcer_target:
            return float(threshold)
    return float(scores.min())


def choose_apcer_threshold(
    attack_scores: Iterable[float] | np.ndarray, apcer_target: float
) -> float | None:
    """Lowest threshold whose APCER is within ``apcer_target``.

    Lowering the threshold accepts more presentations and therefore raises
    APCER. Walking candidates upward returns the lowest threshold that satisfies
    the attack-accept budget, which is also the point with the lowest BPCER.
    Candidates include ``0.0`` and a value just above the highest score so a
    zero-accept budget is reachable.
    """

    if not 0.0 <= apcer_target <= 1.0:
        raise ValueError("apcer_target must be between 0 and 1")
    scores = _score_array(attack_scores)
    if scores.size == 0:
        return None

    candidates = [0.0, *(float(value) for value in np.unique(scores))]
    candidates.append(math.nextafter(float(scores.max()), math.inf))
    for threshold in candidates:
        rate = apcer(int(np.sum(scores >= threshold)), int(scores.size))
        if rate is not None and rate <= apcer_target:
            return threshold
    return math.nextafter(float(scores.max()), math.inf)
