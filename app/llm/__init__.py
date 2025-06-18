"""Convenience exports for LLM utilities."""

from .gemini_wrapper import (
    generate_job_description,
    generate_candidate_remarks,
    assess_resume_with_jd,
)
from .email_generator import generate_interview_email, generate_rejection_email

__all__ = [
    "generate_job_description",
    "generate_candidate_remarks",
    "assess_resume_with_jd",
    "generate_interview_email",
    "generate_rejection_email",
]
