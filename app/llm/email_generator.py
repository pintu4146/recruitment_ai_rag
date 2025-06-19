"""Utilities to generate candidate emails using Gemini."""

from pathlib import Path

import google.generativeai as genai
from langchain.prompts import PromptTemplate
import mlflow

from app.core.logger import logger

from app.mlflow.prompts import PROMPT_REGISTRY

from app.core.config import settings


# Configure Gemini with the API key
genai.configure(api_key=settings.GEMINI_API_KEY)

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"

INTERVIEW_PROMPT = PromptTemplate.from_template(
    (PROMPTS_DIR / "interview_email_prompt.txt").read_text()
)

REJECTION_PROMPT = PromptTemplate.from_template(
    (PROMPTS_DIR / "rejection_email_prompt.txt").read_text()
)


@mlflow.trace()
def generate_interview_email(candidate_name: str, role: str) -> str:
    """Generate an interview invitation email."""
    logger.info(f"Generating interview email for {candidate_name}")

    prompt = INTERVIEW_PROMPT.format(candidate_name=candidate_name, role=role)
    mlflow.log_param("prompt", PROMPT_REGISTRY["interview_email"])
    model = genai.GenerativeModel("gemini-pro")
    try:
        response = model.generate_content(prompt)
    except Exception:
        logger.exception("Gemini interview email generation failed")
        raise
    return response.text.strip()


@mlflow.trace()
def generate_rejection_email(candidate_name: str, role: str) -> str:
    """Generate a rejection email."""
    logger.info(f"Generating rejection email for {candidate_name}")

    prompt = REJECTION_PROMPT.format(candidate_name=candidate_name, role=role)
    mlflow.log_param("prompt", PROMPT_REGISTRY["rejection_email"])
    model = genai.GenerativeModel("gemini-pro")
    try:
        response = model.generate_content(prompt)
    except Exception:
        logger.exception("Gemini rejection email generation failed")
        raise
    return response.text.strip()

