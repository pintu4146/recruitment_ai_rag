from fastapi import APIRouter, UploadFile, File, HTTPException

from app.models.jd import JDGenerationRequest
from app.utils.file_parser import extract_text_from_upload
from app.llm.gemini_wrapper import generate_job_description
from app.llm.output_parser import parse_jd_response

router = APIRouter()


@router.post("/upload")
async def upload_jd(file: UploadFile = File(...)):
    """Upload a JD file and return extracted text."""
    try:
        text = await extract_text_from_upload(file)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"text": text}


@router.post("/generate")
def generate_jd(payload: JDGenerationRequest):
    """Generate a JD using the Gemini API."""
    jd_text = generate_job_description(payload.role, payload.tech_stack)
    parsed = parse_jd_response(jd_text)
    return {"jd": jd_text, "parsed": parsed}
