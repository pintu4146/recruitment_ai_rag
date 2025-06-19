"""Thin wrapper around a ChromaDB persistent client."""
from __future__ import annotations

from chromadb import PersistentClient

from app.core.config import settings
from app.core.logger import logger

_client: PersistentClient | None = None


def get_client() -> PersistentClient:
    """Return a cached ChromaDB client in persistent (local) mode."""
    global _client
    if _client is None:
        logger.info(f"Initializing ChromaDB client at {settings.CHROMA_DIR}")
        _client = PersistentClient(path=settings.CHROMA_DIR)
    return _client


def get_collection(name: str) -> any:
    """Get or create a collection by name."""
    logger.debug(f"Fetching collection: {name}")
    client = get_client()
    return client.get_or_create_collection(name)
