from __future__ import annotations

"""Embedding encoder interfaces and implementations."""
from abc import ABC, abstractmethod
from functools import lru_cache
from typing import Iterable, List, Union

from sentence_transformers import SentenceTransformer


class BaseEmbeddingEncoder(ABC):
    """Abstract base class for embedding encoders."""

    @abstractmethod
    def encode(self, text: Union[str, List[str]]) -> List[float]:
        """Return embedding vector for the given text."""
        raise NotImplementedError


class MiniLMEncoder(BaseEmbeddingEncoder):
    """Encoder using sentence-transformers MiniLM model."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model = _load_model(model_name)

    def encode(self, text: Union[str, List[str]]) -> List[float]:
        embedding = self.model.encode(text, show_progress_bar=False)
        # sentence-transformers returns numpy array or list for str/list inputs
        return embedding.tolist() if hasattr(embedding, "tolist") else embedding


class E5Encoder(BaseEmbeddingEncoder):
    """Encoder using the E5 sentence-transformer model."""

    def __init__(self, model_name: str = "intfloat/e5-large") -> None:
        self.model = _load_model(model_name)

    def encode(self, text: Union[str, List[str]]) -> List[float]:
        embedding = self.model.encode(text, show_progress_bar=False)
        return embedding.tolist() if hasattr(embedding, "tolist") else embedding


# Optional stub implementations for future extension
class OpenAIEncoder(BaseEmbeddingEncoder):
    """Stub for OpenAI embedding encoder."""

    def __init__(self, model_name: str = "text-embedding-ada-002") -> None:
        self.model_name = model_name

    def encode(self, text: Union[str, List[str]]) -> List[float]:
        raise NotImplementedError("OpenAIEncoder not implemented in this example")


class BGEEncoder(BaseEmbeddingEncoder):
    """Stub for BGE embedding encoder."""

    def __init__(self, model_name: str = "bge-base-en") -> None:
        self.model_name = model_name

    def encode(self, text: Union[str, List[str]]) -> List[float]:
        raise NotImplementedError("BGEEncoder not implemented in this example")


class NVEmbedEncoder(BaseEmbeddingEncoder):
    """Stub for NVIDIA embedding encoder."""

    def __init__(self, model_name: str = "nv-embed-v2") -> None:
        self.model_name = model_name

    def encode(self, text: Union[str, List[str]]) -> List[float]:
        raise NotImplementedError("NVEmbedEncoder not implemented in this example")


@lru_cache(maxsize=None)
def _load_model(model_name: str) -> SentenceTransformer:
    """Load and cache a SentenceTransformer model."""
    return SentenceTransformer(model_name)

