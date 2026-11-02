from __future__ import annotations

import numpy as np
import pytest

from facecheck.similarity import (
    MATCH,
    NO_MATCH,
    best_cosine,
    cosine_similarity,
    decide,
    l2_normalize,
)


def test_l2_normalize_returns_unit_norm_float32() -> None:
    vector = np.array([3.0, 4.0], dtype=np.float64)
    normalized = l2_normalize(vector)
    assert normalized.dtype == np.float32
    assert np.isclose(np.linalg.norm(normalized), 1.0)


def test_l2_normalize_rejects_zero_vector() -> None:
    with pytest.raises(ValueError, match="zero vector"):
        l2_normalize(np.zeros(8, dtype=np.float32))


def test_cosine_similarity_of_identical_vectors_is_one() -> None:
    vector = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    assert cosine_similarity(vector, vector) == pytest.approx(1.0, abs=1e-6)


def test_cosine_similarity_of_orthogonal_vectors_is_zero() -> None:
    assert cosine_similarity(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == pytest.approx(0.0)


def test_cosine_similarity_of_opposite_vectors_is_minus_one() -> None:
    assert cosine_similarity(np.array([1.0, 2.0]), np.array([-1.0, -2.0])) == pytest.approx(-1.0)


def test_cosine_similarity_is_scale_invariant() -> None:
    left = np.array([1.0, 2.0, 3.0])
    right = np.array([2.0, 4.0, 6.0])
    assert cosine_similarity(left, right) == pytest.approx(1.0)


def test_cosine_similarity_rejects_dimension_mismatch() -> None:
    with pytest.raises(ValueError, match="dimensions differ"):
        cosine_similarity(np.zeros(4), np.zeros(5))


def test_cosine_similarity_of_zero_vector_is_zero() -> None:
    assert cosine_similarity(np.zeros(4), np.ones(4)) == 0.0


@pytest.mark.parametrize(
    ("score", "threshold", "expected"),
    [
        (0.5, 0.363, MATCH),
        (0.363, 0.363, MATCH),
        (0.3629, 0.363, NO_MATCH),
        (-0.1, 0.0, NO_MATCH),
    ],
)
def test_decide(score: float, threshold: float, expected: str) -> None:
    assert decide(score, threshold) == expected


def test_best_cosine_takes_maximum_across_templates() -> None:
    query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    templates = [
        np.array([0.0, 1.0, 0.0], dtype=np.float32),
        np.array([0.8, 0.6, 0.0], dtype=np.float32),
        np.array([0.6, 0.8, 0.0], dtype=np.float32),
    ]
    assert best_cosine(query, templates) == pytest.approx(0.8)


def test_best_cosine_requires_templates() -> None:
    with pytest.raises(ValueError, match="no templates"):
        best_cosine(np.ones(4), [])
