"""Convenience exports for LLM utilities."""

from .gemini_wrapper import generate_job_description, generate_candidate_remarks
from .email_generator import generate_interview_email, generate_rejection_email

__all__ = [
    "generate_job_description",
    "generate_candidate_remarks",
    "generate_interview_email",
    "generate_rejection_email",
]
