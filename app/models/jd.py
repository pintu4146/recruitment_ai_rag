from __future__ import annotations

from pydantic import BaseModel, Field


class JDGenerationRequest(BaseModel):
    """Input model for JD generation via Gemini."""

    role: str = Field(..., example="Software Engineer")
    tech_stack: str | None = Field(
        None,
        example="Python, FastAPI, PostgreSQL",
        description="Comma separated list of required technologies",
    )
    company_name: str | None = Field(None,
                                     example="E2M")
