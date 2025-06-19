"""Utility functions for generating text embeddings."""

from functools import lru_cache
from typing import List

from app.core.logger import logger
from app.core.config import settings
from app.embedding.factory import get_embedding_encoder
from app.embedding.encoder import BaseEmbeddingEncoder


@lru_cache(maxsize=1)
def _get_encoder() -> BaseEmbeddingEncoder:
    """Load and cache the configured embedding encoder."""
    logger.debug(f"Loading embedding encoder: {settings.EMBEDDING_MODEL}")
    return get_embedding_encoder(settings.EMBEDDING_MODEL)


def embed_text(text: str) -> List[float]:
    """Return embedding vector for a chunk of text using the configured encoder."""
    logger.debug("Generating embedding")
    encoder = _get_encoder()
    embedding = encoder.encode(text)
    return embedding
