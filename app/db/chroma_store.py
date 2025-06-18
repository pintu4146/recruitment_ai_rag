"""Thin wrapper around a ChromaDB persistent client."""
from __future__ import annotations

from chromadb import PersistentClient

from app.core.config import settings

_client: PersistentClient | None = None


def get_client() -> PersistentClient:
    """Return a cached ChromaDB client in persistent (local) mode."""
    global _client
    if _client is None:
        _client = PersistentClient(path=settings.CHROMA_DIR)
    return _client


def get_collection(name: str) -> any:
    """Get or create a collection by name."""
    client = get_client()
    return client.get_or_create_collection(name)
