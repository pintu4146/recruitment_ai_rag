"""Wrapper utilities for interacting with the Gemini API."""

import google.generativeai as genai
from pathlib import Path
from langchain.prompts import PromptTemplate

from app.core.config import settings


genai.configure(api_key=settings.GEMINI_API_KEY)

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"

JD_PROMPT_TEMPLATE = PromptTemplate.from_template(
    (PROMPTS_DIR / "jd_prompt.txt").read_text()
)

REMARKS_PROMPT_TEMPLATE = PromptTemplate.from_template(
    (PROMPTS_DIR / "remarks_prompt.txt").read_text()
)


def generate_job_description(role: str, tech_stack: str | None = None) -> str:
    """Generate a job description for the given role using Gemini."""
    prompt = JD_PROMPT_TEMPLATE.format(role=role, tech_stack=tech_stack or "")
    model = genai.GenerativeModel("gemini-pro")
    response = model.generate_content(prompt)
    return response.text


def generate_candidate_remarks(missing: list[str], strong: list[str]) -> str:
    """Generate a short remark highlighting missing and strong skills."""

    prompt = REMARKS_PROMPT_TEMPLATE.format(
        missing=", ".join(missing) if missing else "none",
        strong=", ".join(strong) if strong else "none",
    )
    model = genai.GenerativeModel("gemini-pro")
    response = model.generate_content(prompt)
    return response.text.strip()
