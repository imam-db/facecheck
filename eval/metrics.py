"""Metric helpers for the LFW evaluation harness.

The functions here are intentionally small and pure so they can be unit
tested on synthetic scores without touching the models or the dataset.
"""

from __future__ import annotations

import math
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
