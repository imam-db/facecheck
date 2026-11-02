"""Unit tests for the evaluation metric helpers on synthetic scores."""

from __future__ import annotations

import numpy as np
import pytest
from eval.metrics import (
    choose_far_threshold,
    choose_youden,
    failure_to_acquire,
    generalized_frr,
    roc_auc,
    rule_of_three,
    scores_at,
    wilson_interval,
)


def test_scores_at_counts_and_rates() -> None:
    scores = np.array([0.9, 0.8, 0.2, 0.1])
    labels = np.array([1, 1, 0, 0])
    metrics = scores_at(scores, labels, 0.5)
    assert metrics.threshold == 0.5
    assert metrics.true_accepts == 2
    assert metrics.false_accepts == 0
    assert metrics.true_rejects == 2
    assert metrics.false_rejects == 0
    assert metrics.tpr == pytest.approx(1.0)
    assert metrics.far == pytest.approx(0.0)
    assert metrics.frr == pytest.approx(0.0)
    assert metrics.accuracy == pytest.approx(1.0)


def test_scores_at_is_inclusive_at_the_threshold() -> None:
    metrics = scores_at(np.array([0.363]), np.array([1]), 0.363)
    assert metrics.true_accepts == 1
    assert metrics.tpr == 1.0


def test_choose_far_threshold_respects_the_budget() -> None:
    scores = np.array([0.9, 0.8, 0.4, 0.3])
    labels = np.array([1, 1, 0, 0])
    threshold, metrics = choose_far_threshold(scores, labels, 0.0)
    assert threshold == pytest.approx(0.8)
    assert metrics is not None
    assert metrics.far == 0.0
    assert metrics.tpr == pytest.approx(1.0)


def test_choose_far_threshold_returns_none_for_single_class() -> None:
    threshold, metrics = choose_far_threshold(
        np.array([0.9, 0.8]), np.array([1, 1]), 0.01
    )
    assert threshold is None
    assert metrics is None


def test_choose_youden_prefers_separating_threshold() -> None:
    scores = np.array([0.9, 0.8, 0.2, 0.1])
    labels = np.array([1, 1, 0, 0])
    threshold, metrics = choose_youden(scores, labels)
    assert threshold == pytest.approx(0.8)
    assert metrics.tpr - metrics.far == pytest.approx(1.0)


def test_rule_of_three_matches_the_approximation() -> None:
    bound = rule_of_three(430)
    assert bound == pytest.approx(3 / 430, rel=0.01)
    assert bound == pytest.approx(0.00694, abs=1e-5)
    assert rule_of_three(0) is None


def test_wilson_interval_for_zero_events() -> None:
    interval = wilson_interval(0, 430)
    assert interval is not None
    low, high = interval
    assert low == 0.0
    assert 0.0 < high < 0.01
    assert high == pytest.approx(0.00885, abs=1e-4)


def test_wilson_interval_contains_the_estimate() -> None:
    interval = wilson_interval(50, 100)
    assert interval is not None
    low, high = interval
    assert low < 0.5 < high
    assert 0.39 < low < 0.41
    assert 0.59 < high < 0.61
    assert wilson_interval(0, 0) is None


def test_generalized_frr_counts_skipped_genuine_pairs() -> None:
    assert generalized_frr(5, 10, 100) == pytest.approx(0.15)
    assert generalized_frr(0, 0, 0) is None


def test_failure_to_acquire() -> None:
    assert failure_to_acquire(152, 1000) == pytest.approx(0.152)
    assert failure_to_acquire(0, 0) is None


def test_roc_auc_on_synthetic_scores() -> None:
    scores = np.array([0.9, 0.8, 0.2, 0.1])
    labels = np.array([1, 1, 0, 0])
    assert roc_auc(scores, labels) == pytest.approx(1.0)
