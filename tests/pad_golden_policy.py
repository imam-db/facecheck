"""Policy helpers for the PAD golden manifest.

A case is **discriminative** when the model already gets its label right on the
correct raw-BGR crop, so a preprocessing regression (``/255`` scaling or a
BGR/RGB swap) changes the score in an observable way. Missed-attack cases (a
spoof the model scores as real) are not discriminative: the contrast assertions
cannot distinguish a regression from an inherently hard image, so they would
fail for reasons unrelated to preprocessing.

The flag defaults to ``False`` when absent so new cases are safe unless they
are deliberately marked. These helpers are pure functions; they need no
images, models, or ``FACECHECK_PAD_GOLDEN_DIR``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

DISCRIMINATIVE_KEY = "discriminative"
DISCRIMINATIVE_LABELS = ("real", "spoof")


def is_discriminative(case: Mapping[str, Any]) -> bool:
    """Whether a case opted into the preprocessing contrast assertions."""

    return bool(case.get(DISCRIMINATIVE_KEY, False))


def should_assert_div255(case: Mapping[str, Any]) -> bool:
    """``/255`` collapses the real probability; only check discriminative cases."""

    return is_discriminative(case)


def should_assert_rgb_swap_gain(case: Mapping[str, Any]) -> bool:
    """The fail-open gain assertion only makes sense for discriminative spoofs."""

    return is_discriminative(case) and case.get("label") == "spoof"


def discriminative_coverage(
    cases: Iterable[Mapping[str, Any]],
) -> dict[str, int]:
    """Count discriminative cases per ground-truth label."""

    coverage = dict.fromkeys(DISCRIMINATIVE_LABELS, 0)
    for case in cases:
        label = case.get("label")
        if label in coverage and is_discriminative(case):
            coverage[label] += 1
    return coverage


def missing_discriminative_labels(
    cases: Iterable[Mapping[str, Any]],
) -> tuple[str, ...]:
    """Labels that have no discriminative case in the set."""

    coverage = discriminative_coverage(cases)
    return tuple(label for label in DISCRIMINATIVE_LABELS if coverage[label] == 0)
