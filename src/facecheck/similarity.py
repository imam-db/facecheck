"""Cosine similarity and match decisions.

Embeddings are L2-normalized float32 vectors, so the cosine similarity of two
embeddings is their dot product. The functions here are pure NumPy and do not
import OpenCV, which keeps them fast to test.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

MATCH = "match"
NO_MATCH = "no_match"


def l2_normalize(vector: np.ndarray) -> np.ndarray:
    """Return ``vector`` scaled to unit L2 norm as float32."""

    array = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(array))
    if norm == 0.0:
        raise ValueError("cannot normalize a zero vector")
    return (array / norm).astype(np.float32, copy=False)


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    """Cosine similarity in ``[-1, 1]`` for two vectors of equal length."""

    a = np.asarray(left, dtype=np.float32).reshape(-1)
    b = np.asarray(right, dtype=np.float32).reshape(-1)
    if a.shape != b.shape:
        raise ValueError(f"embedding dimensions differ: {a.shape[0]} != {b.shape[0]}")
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator == 0.0:
        return 0.0
    return float(np.clip(np.dot(a, b) / denominator, -1.0, 1.0))


def decide(score: float, threshold: float) -> str:
    """Map a similarity score to ``match`` or ``no_match`` (inclusive threshold)."""

    return MATCH if score >= threshold else NO_MATCH


def best_cosine(
    query: np.ndarray,
    templates: Iterable[np.ndarray],
) -> float:
    """Highest cosine similarity against any template.

    facecheck enrolls every accepted frame as its own template and verifies
    with the maximum similarity across a person's templates. Averaging
    templates was rejected because it blurs pose and lighting variation that
    multiple templates are meant to capture.
    """

    scores = [cosine_similarity(query, template) for template in templates]
    if not scores:
        raise ValueError("no templates to compare against")
    return max(scores)
