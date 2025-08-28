"""LLM retriever interface definitions."""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import google.generativeai as genai

from app.core.config import settings
from app.core.logger import logger


genai.configure(api_key=settings.GEMINI_API_KEY)


class BaseRetrieverLLM(ABC):
    """Abstract interface for retrieval-augmented generation LLMs."""

    @abstractmethod
    def generate(self, query: str, context: str) -> str:
        """Generate text given a query and retrieved context."""
        raise NotImplementedError


class GeminiRetrieverLLM(BaseRetrieverLLM):
    """Retriever implementation using Gemini models."""

    PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"
    MATCHING_PROMPT = (PROMPTS_DIR / "matching_prompt.txt").read_text()

    def __init__(self, model_name: str = "gemini-2.0-flash") -> None:
        self.model = genai.GenerativeModel(model_name)

    def generate(self, query: str, context: str) -> str:
        prompt = self.MATCHING_PROMPT.format(jd=query, resume=context)
        logger.debug("Calling Gemini model")
        response = self.model.generate_content(prompt)
        return response.text


# Stubs for additional providers
class OpenAIRetrieverLLM(BaseRetrieverLLM):
    def generate(self, query: str, context: str) -> str:
        raise NotImplementedError("OpenAI retriever not implemented")


class CohereRetrieverLLM(BaseRetrieverLLM):
    def generate(self, query: str, context: str) -> str:
        raise NotImplementedError("Cohere retriever not implemented")


