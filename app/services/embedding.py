"""Utility functions for generating text embeddings."""

from functools import lru_cache
from typing import List

from sentence_transformers import SentenceTransformer

from app.core.logger import logger


@lru_cache(maxsize=1)
def _get_model(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    """Load and cache the sentence transformer model."""
    logger.debug(f"Loading sentence transformer model: {model_name}")
    return SentenceTransformer(model_name)


def embed_text(text: str) -> List[float]:
    """Return embedding vector for a chunk of text."""
    logger.debug("Generating embedding")
    model = _get_model()
    embedding = model.encode(text, show_progress_bar=False)
    return embedding.tolist()
