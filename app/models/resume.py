from pydantic import BaseModel, Field, EmailStr


class ResumeMetadata(BaseModel):
    """Metadata accompanying an uploaded resume."""

    candidate_name: str = Field(..., example="John Doe")
    email: EmailStr | None = Field(
        None,
        example="pin21k19@gmail.com",
        description="Candidate contact email",
    )
    notes: str | None = Field(None, example="Referred by Alice")

