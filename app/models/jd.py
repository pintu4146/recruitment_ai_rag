from pydantic import BaseModel, Field


class JDGenerationRequest(BaseModel):
    """Input model for JD generation via Gemini."""

    role: str = Field(..., example="Software Engineer")
