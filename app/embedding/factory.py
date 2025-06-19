"""Factory utilities to obtain embedding encoder instances."""
from __future__ import annotations

from typing import Dict, Type

from .encoder import (
    BaseEmbeddingEncoder,
    MiniLMEncoder,
    E5Encoder,
    OpenAIEncoder,
    BGEEncoder,
    NVEmbedEncoder,
)


_ENCODER_REGISTRY: Dict[str, Type[BaseEmbeddingEncoder]] = {
    "MiniLM": MiniLMEncoder,
    "E5": E5Encoder,
    "OpenAI": OpenAIEncoder,
    "BGE": BGEEncoder,
    "NVEmbed": NVEmbedEncoder,
}


def get_embedding_encoder(name: str) -> BaseEmbeddingEncoder:
    """Return an embedding encoder instance by name."""
    try:
        encoder_cls = _ENCODER_REGISTRY[name]
    except KeyError as exc:
        raise ValueError(f"Unknown embedding model: {name}") from exc
    return encoder_cls()

