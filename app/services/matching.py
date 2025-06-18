"""Utilities to compute similarity scores between embeddings."""

from typing import Iterable
import numpy as np


def cosine_score(vec1: Iterable[float], vec2: Iterable[float]) -> float:
    """Return cosine similarity score (0-100) between two vectors."""
    a = np.array(list(vec1))
    b = np.array(list(vec2))
    if a.size == 0 or b.size == 0:
        raise ValueError("Vectors must be non-empty")
    score = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    return round(score * 100, 2)
