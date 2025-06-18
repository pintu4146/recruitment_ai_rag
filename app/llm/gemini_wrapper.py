"""Wrapper utilities for interacting with the Gemini API."""

import google.generativeai as genai

from app.core.config import settings


genai.configure(api_key=settings.GEMINI_API_KEY)

PROMPT_TEMPLATE = (
    "You are an HR assistant. Generate a detailed job description for the "
    "role: {role}."
)


def generate_job_description(role: str) -> str:
    """Generate a job description for the given role using Gemini."""
    prompt = PROMPT_TEMPLATE.format(role=role)
    model = genai.GenerativeModel("gemini-pro")
    response = model.generate_content(prompt)
    return response.text
