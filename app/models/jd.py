from __future__ import annotations

from pydantic import BaseModel, Field


class JDGenerationRequest(BaseModel):
    """Input model for JD generation via Gemini."""

    job_title: str = Field(..., example="Software Engineer")
    year_experience: float

    tech_stack_must_have: str = Field(
        None,
        example="Python, FastAPI, PostgreSQL",
        description="Comma separated list of required technologies",
    )
    good_to_have: str | None = Field(
        None,
        example="docker, kubernetes, ci/cd",
        description="Comma separated list of good to have  technologies",
    )
    company_name: str | None = Field(None,
                                     example="E2M")
    employment_type: str = Field(...,
                                 example='Full_time')
    industry: str = Field(...,
                          example='IT')
    location: str = Field(...,
                          example='indore')
