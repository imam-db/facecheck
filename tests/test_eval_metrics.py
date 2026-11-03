"""Unit tests for the evaluation metric helpers on synthetic scores."""

from __future__ import annotations

import numpy as np
import pytest
from eval.metrics import (
    apcer,
    bpcer,
    choose_apcer_threshold,
    choose_bpcer_threshold,
    choose_far_threshold,
    choose_youden,
    failure_to_acquire,
    generalized_frr,
    iapmr,
    pad_rates_at,
    pad_threshold_sweep,
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


def test_pad_rate_helpers_and_empty_denominators() -> None:
    assert apcer(3, 10) == pytest.approx(0.3)
    assert bpcer(1, 4) == pytest.approx(0.25)
    assert iapmr(2, 5) == pytest.approx(0.4)
    assert apcer(0, 0) is None
    assert bpcer(0, 0) is None
    assert iapmr(0, 0) is None


def test_pad_rates_at_accepts_at_or_above_the_threshold() -> None:
    rates = pad_rates_at(
        bona_scores=[0.9, 0.5, 0.1],
        attack_scores=[0.9, 0.4, 0.2],
        threshold=0.5,
    )
    assert rates.bona_presentations == 3
    assert rates.bona_rejected == 1
    assert rates.bpcer == pytest.approx(1 / 3)
    assert rates.attack_presentations == 3
    assert rates.attack_accepted == 1
    assert rates.apcer == pytest.approx(1 / 3)


def test_pad_rates_with_no_bona_or_attacks() -> None:
    rates = pad_rates_at([], [], 0.5)
    assert rates.bpcer is None
    assert rates.apcer is None
    assert rates.bona_presentations == 0
    assert rates.attack_presentations == 0


def test_choose_bpcer_threshold_picks_the_highest_threshold_in_budget() -> None:
    scores = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    threshold = choose_bpcer_threshold(scores, 0.2)
    assert threshold == pytest.approx(0.3)
    assert pad_rates_at(scores, [], threshold).bpcer == pytest.approx(0.2)

    assert choose_bpcer_threshold(scores, 0.0) == pytest.approx(0.1)
    assert choose_bpcer_threshold([], 0.05) is None


def test_choose_bpcer_threshold_handles_duplicate_scores() -> None:
    scores = [0.2, 0.2, 0.2, 0.9]
    assert choose_bpcer_threshold(scores, 0.5) == pytest.approx(0.2)


def test_choose_apcer_threshold_picks_the_lowest_threshold_in_budget() -> None:
    scores = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    threshold = choose_apcer_threshold(scores, 0.2)
    assert threshold == pytest.approx(0.9)
    assert pad_rates_at([], scores, threshold).apcer == pytest.approx(0.2)

    assert choose_apcer_threshold([], 0.1) is None


def test_choose_apcer_threshold_can_reach_zero_accepts() -> None:
    threshold = choose_apcer_threshold([0.2, 0.6, 0.9], 0.0)
    assert threshold is not None
    assert threshold > 0.9
    assert pad_rates_at([], [0.2, 0.6, 0.9], threshold).apcer == pytest.approx(0.0)


def test_threshold_target_helpers_reject_bad_targets() -> None:
    with pytest.raises(ValueError):
        choose_bpcer_threshold([0.5], -0.1)
    with pytest.raises(ValueError):
        choose_apcer_threshold([0.5], 1.1)


def test_pad_threshold_sweep_sorts_and_deduplicates() -> None:
    sweep = pad_threshold_sweep([0.2, 0.8], [0.4, 0.9], [0.9, 0.5, 0.5])
    assert [point.threshold for point in sweep] == pytest.approx([0.5, 0.9])
    assert sweep[0].bpcer == pytest.approx(0.5)
    assert sweep[0].apcer == pytest.approx(0.5)
