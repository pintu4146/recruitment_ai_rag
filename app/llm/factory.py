"""Factory helpers for retriever LLMs."""
from __future__ import annotations

from typing import Dict, Type

from .interfaces import (
    BaseRetrieverLLM,
    GeminiRetrieverLLM,
    OpenAIRetrieverLLM,
    CohereRetrieverLLM,
)


_LLM_REGISTRY: Dict[str, Type[BaseRetrieverLLM]] = {
    "Gemini": GeminiRetrieverLLM,
    "OpenAI": OpenAIRetrieverLLM,
    "Cohere": CohereRetrieverLLM,
}


def get_retriever_llm(name: str) -> BaseRetrieverLLM:
    """Return a retriever LLM instance by name."""
    try:
        llm_cls = _LLM_REGISTRY[name]
    except KeyError as exc:
        raise ValueError(f"Unknown LLM provider: {name}") from exc
    return llm_cls()

