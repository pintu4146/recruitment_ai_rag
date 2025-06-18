"""Utility functions for generating text embeddings."""

from functools import lru_cache
from typing import List

from sentence_transformers import SentenceTransformer


@lru_cache(maxsize=1)
def _get_model(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    """Load and cache the sentence transformer model."""
    return SentenceTransformer(model_name)


def embed_text(text: str) -> List[float]:
    """Return embedding vector for a chunk of text."""
    model = _get_model()
    embedding = model.encode(text, show_progress_bar=False)
    return embedding.tolist()
