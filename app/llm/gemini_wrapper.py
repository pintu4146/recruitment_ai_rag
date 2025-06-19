"""Wrapper utilities for interacting with the Gemini API."""

import google.generativeai as genai
from pathlib import Path
from langchain.prompts import PromptTemplate
import mlflow

from app.core.logger import logger

from app.mlflow.prompts import PROMPT_REGISTRY

from app.core.config import settings


genai.configure(api_key=settings.GEMINI_API_KEY)

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"

JD_PROMPT_TEMPLATE = PromptTemplate.from_template(
    (PROMPTS_DIR / "jd_prompt.txt").read_text()
)

REMARKS_PROMPT_TEMPLATE = PromptTemplate.from_template(
    (PROMPTS_DIR / "remarks_prompt.txt").read_text()
)

MATCH_PROMPT_TEMPLATE = PromptTemplate.from_template(
    (PROMPTS_DIR / "matching_prompt.txt").read_text()
)


@mlflow.trace()
def generate_job_description(role: str, tech_stack: str | None = None) -> str:
    """Generate a job description for the given role using Gemini."""
    logger.info(f"Generating job description for role={role}")
    prompt = JD_PROMPT_TEMPLATE.format(role=role, tech_stack=tech_stack or "")
    mlflow.log_param("prompt", PROMPT_REGISTRY["generate_jd"])
    model = genai.GenerativeModel("gemini-pro")
    try:
        response = model.generate_content(prompt)
    except Exception:  # broad catch to log unexpected LLM errors
        logger.exception("Gemini JD generation failed")
        raise
    return response.text


@mlflow.trace()
def generate_candidate_remarks(missing: list[str], strong: list[str]) -> str:
    """Generate a short remark highlighting missing and strong skills."""

    logger.info("Generating remarks")
    prompt = REMARKS_PROMPT_TEMPLATE.format(
        missing=", ".join(missing) if missing else "none",
        strong=", ".join(strong) if strong else "none",
    )
    mlflow.log_param("prompt", PROMPT_REGISTRY["generate_remark"])
    model = genai.GenerativeModel("gemini-pro")
    try:
        response = model.generate_content(prompt)
    except Exception:
        logger.exception("Gemini remark generation failed")
        raise
    return response.text.strip()


@mlflow.trace()
def assess_resume_with_jd(jd: str, resume: str, *, top_p: float = 0.8) -> str:
    """Evaluate a resume against a JD and return a short summary."""

    logger.info("Assessing resume against JD")
    prompt = MATCH_PROMPT_TEMPLATE.format(jd=jd, resume=resume)
    mlflow.log_param("prompt", PROMPT_REGISTRY["match_resume"])
    model = genai.GenerativeModel("gemini-pro")
    try:
        response = model.generate_content(
            prompt,
            generation_config={"top_p": top_p},
        )
    except Exception:
        logger.exception("Gemini resume assessment failed")
        raise
    return response.text.strip()

