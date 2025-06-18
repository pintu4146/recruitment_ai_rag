"""Wrapper utilities for interacting with the Gemini API."""

import google.generativeai as genai
from pathlib import Path
from langchain.prompts import PromptTemplate

from app.core.config import settings


genai.configure(api_key=settings.GEMINI_API_KEY)

PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "jd_prompt.txt"
JD_PROMPT_TEMPLATE = PromptTemplate.from_template(PROMPT_PATH.read_text())


def generate_job_description(role: str, tech_stack: str | None = None) -> str:
    """Generate a job description for the given role using Gemini."""
    prompt = JD_PROMPT_TEMPLATE.format(role=role, tech_stack=tech_stack or "")
    model = genai.GenerativeModel("gemini-pro")
    response = model.generate_content(prompt)
    return response.text
