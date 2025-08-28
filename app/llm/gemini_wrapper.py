"""Wrapper utilities for interacting with the Gemini API."""

from pathlib import Path
from langchain.prompts import PromptTemplate
import mlflow

from app.core.logger import logger

from app.mlflow.prompts import PROMPT_REGISTRY

from app.core.config import settings
from app.llm.factory import get_retriever_llm

import google.generativeai as genai
genai.configure(api_key=settings.GEMINI_API_KEY)

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"

JD_PROMPT_TEMPLATE = PromptTemplate.from_template(
    (PROMPTS_DIR / "jd_prompt.txt").read_text()
)

REMARKS_PROMPT_TEMPLATE = PromptTemplate.from_template(
    (PROMPTS_DIR / "remarks_prompt.txt").read_text()
)


@mlflow.trace()
def generate_job_description(
        role: str,
        tech_stack_must_have: str | None = None,
        company_name: str | None = None,
        good_to_have: str | None = None,
        employment_type: str | None = None,
        industry: str | None = None,
        location: str | None = None
) -> str:
    """Generate a job description using Gemini LLM."""

    logger.info(f"Generating job description for role={role}, company={company_name or 'N/A'}")

    prompt = JD_PROMPT_TEMPLATE.format(
        role=role,
        tech_stack=tech_stack_must_have or "Not specified",
        company_name=company_name or "A reputed company",
        good_to_have=good_to_have or "N/A",
        employment_type=employment_type or "Full-time",
        industry=industry or "General",
        location=location or "Remote"
    )

    mlflow.log_param("jd_prompt_template", PROMPT_REGISTRY["generate_jd"])

    model = genai.GenerativeModel("gemini-2.0-flash")

    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception:
        logger.exception("Gemini JD generation failed")
        raise


@mlflow.trace()
def generate_candidate_remarks(missing: list[str], strong: list[str]) -> str:
    """Generate a short remark highlighting missing and strong skills."""

    logger.info("Generating remarks")
    prompt = REMARKS_PROMPT_TEMPLATE.format(
        missing=", ".join(missing) if missing else "none",
        strong=", ".join(strong) if strong else "none",
    )
    mlflow.log_param("prompt", PROMPT_REGISTRY["generate_remark"])
    model = genai.GenerativeModel("gemini-2.0-flash")
    try:
        response = model.generate_content(prompt)
    except Exception:
        logger.exception("Gemini remark generation failed")
        raise
    return response.text.strip()


@mlflow.trace()
def assess_resume_with_jd(jd: str, resume: str, *, top_p: float = 0.8) -> str:
    """Evaluate a resume against a JD using the configured retriever LLM."""

    logger.info("Assessing resume against JD")
    mlflow.log_param("prompt_template", PROMPT_REGISTRY["match_resume"])
    llm = get_retriever_llm(settings.RETRIEVAL_MODEL)
    try:
        response = llm.generate(jd, resume)
    except Exception:
        logger.exception("Resume assessment failed")
        raise
    return response.strip()
