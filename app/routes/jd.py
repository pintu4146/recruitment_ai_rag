from typing import List

from fastapi import APIRouter, UploadFile, File, HTTPException

from app.core.logger import logger

from app.models.jd import JDGenerationRequest
from app.utils.file_parser import extract_text_from_upload
from app.llm.gemini_wrapper import generate_job_description
from app.llm.output_parser import parse_jd_response

router = APIRouter()


@router.post("/upload")
async def upload_jd(files: List[UploadFile] = File(...)):
    """Upload one or more JD files and return extracted texts."""

    texts: list[str] = []
    for file in files:
        try:
            text = await extract_text_from_upload(file)
        except ValueError as exc:
            logger.exception("JD upload failed")
            raise HTTPException(status_code=400, detail=str(exc))
        texts.append(text)
    return {"texts": texts}


@router.post("/generate")
def generate_jd(payload: JDGenerationRequest):
    """Generate a JD using the Gemini API."""
    try:
        jd_text = generate_job_description(payload.job_title, payload.tech_stack_must_have
                                           , payload.company_name, payload.good_to_have
                                           , payload.employment_type, payload.industry
                                           , payload.location)
        parsed = parse_jd_response(jd_text)
    except Exception as exc:
        logger.exception("JD generation failed")
        raise HTTPException(status_code=500, detail=str(exc))
    return {"jd": jd_text, "parsed": parsed}
