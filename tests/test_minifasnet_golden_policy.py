"""Unit tests for the PAD golden assertion policy.

These run without ``FACECHECK_PAD_GOLDEN_DIR``, images, or MiniFASNet models, so
CI still covers the discriminative gate that the semantic golden tests apply.
"""

from __future__ import annotations

import pytest

from tests.pad_golden_policy import (
    discriminative_coverage,
    is_discriminative,
    missing_discriminative_labels,
    should_assert_div255,
    should_assert_rgb_swap_gain,
)

MISSED_ATTACK = {
    "id": "synthetic_missed_attack",
    "label": "spoof",
    "expected_real_prob": 0.99,
    "discriminative": False,
}
DISCRIMINATIVE_SPOOF = {
    "id": "synthetic_spoof",
    "label": "spoof",
    "expected_real_prob": 0.05,
    "discriminative": True,
}
DISCRIMINATIVE_REAL = {
    "id": "synthetic_real",
    "label": "real",
    "expected_real_prob": 0.99,
    "discriminative": True,
}


def test_flag_defaults_to_false_when_absent() -> None:
    assert is_discriminative({"id": "unmarked", "label": "spoof"}) is False
    assert should_assert_div255({"id": "unmarked", "label": "real"}) is False
    assert should_assert_rgb_swap_gain({"id": "unmarked", "label": "spoof"}) is False


def test_missed_attack_skips_per_case_contrast_assertions() -> None:
    # A spoof the model scores as real cannot show a /255 collapse or an
    # RGB-swap gain, so both contrast assertions must not fire.
    assert should_assert_div255(MISSED_ATTACK) is False
    assert should_assert_rgb_swap_gain(MISSED_ATTACK) is False


def test_discriminative_spoof_applies_contrast_assertions() -> None:
    assert should_assert_div255(DISCRIMINATIVE_SPOOF) is True
    assert should_assert_rgb_swap_gain(DISCRIMINATIVE_SPOOF) is True


def test_discriminative_real_applies_only_div255() -> None:
    assert should_assert_div255(DISCRIMINATIVE_REAL) is True
    # Real cases make no RGB-swap gain claim.
    assert should_assert_rgb_swap_gain(DISCRIMINATIVE_REAL) is False


def test_truthy_flags_are_accepted() -> None:
    assert is_discriminative({"label": "real", "discriminative": 1}) is True
    assert is_discriminative({"label": "real", "discriminative": "yes"}) is True


def test_coverage_requires_discriminative_real_and_spoof() -> None:
    missing = missing_discriminative_labels([MISSED_ATTACK])
    assert missing == ("real", "spoof")

    missing = missing_discriminative_labels([MISSED_ATTACK, DISCRIMINATIVE_REAL])
    assert missing == ("spoof",)

    missing = missing_discriminative_labels(
        [MISSED_ATTACK, DISCRIMINATIVE_REAL, DISCRIMINATIVE_SPOOF]
    )
    assert missing == ()


def test_coverage_counts_by_label() -> None:
    coverage = discriminative_coverage([MISSED_ATTACK, DISCRIMINATIVE_REAL, DISCRIMINATIVE_SPOOF])
    assert coverage == {"real": 1, "spoof": 1}


@pytest.mark.parametrize(
    "cases",
    [
        [],
        [{"label": "real", "discriminative": True}],
        [{"label": "spoof", "discriminative": True}],
        [MISSED_ATTACK, DISCRIMINATIVE_REAL],
        [MISSED_ATTACK, DISCRIMINATIVE_SPOOF],
    ],
)
def test_set_level_coverage_rejects_incomplete_sets(cases: list[dict[str, object]]) -> None:
    assert missing_discriminative_labels(cases) != ()
