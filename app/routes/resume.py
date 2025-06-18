from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Form

from app.models.resume import ResumeMetadata
from app.utils.file_parser import extract_text_from_upload

router = APIRouter()


async def _metadata_from_form(
    candidate_name: str = Form(...),
    email: str | None = Form(None),
    notes: str | None = Form(None),
) -> ResumeMetadata:
    """Construct ResumeMetadata from multipart form fields."""

    return ResumeMetadata(candidate_name=candidate_name, email=email, notes=notes)


@router.post("/upload")
async def upload_resume(
    file: UploadFile = File(...),
    metadata: ResumeMetadata = Depends(_metadata_from_form),
):
    """Upload a single resume and return parsed text."""
    try:
        text = await extract_text_from_upload(file)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"metadata": metadata.model_dump(), "text": text}
