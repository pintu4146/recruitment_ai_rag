"""Utilities to compute similarity scores between embeddings."""

from typing import Iterable
import numpy as np
import mlflow

from app.core.logger import logger


def cosine_score(vec1: Iterable[float], vec2: Iterable[float]) -> float:
    """Return cosine similarity score (0-100) between two vectors."""
    logger.debug("Computing cosine similarity")
    a = np.array(list(vec1))
    b = np.array(list(vec2))
    if a.size == 0 or b.size == 0:
        logger.error("Empty vectors passed to cosine_score")
        raise ValueError("Vectors must be non-empty")
    score = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    score = round(score * 100, 2)
    mlflow.log_metric("similarity_score", score)
    return score
