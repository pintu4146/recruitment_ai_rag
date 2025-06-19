from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Form

from app.core.logger import logger

from app.db.chroma_store import get_collection
from app.llm import assess_resume_with_jd
from app.llm.output_parser import parse_analysis_response
from app.services.embedding import embed_text
from app.services.matching import cosine_score

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
    jd_file: UploadFile | None = File(None),
    jd_text: str | None = Form(None),
    top_k: int = Form(1),
    top_p: float = Form(0.8),
):
    """Upload a single resume and return parsed text."""
    try:
        resume_text = await extract_text_from_upload(file)
    except ValueError as exc:
        logger.exception("Resume upload failed")
        raise HTTPException(status_code=400, detail=str(exc))

    resume_embedding = embed_text(resume_text)
    response = {"metadata": metadata.model_dump(), "text": resume_text}

    jd_source = None
    if jd_file is not None:
        try:
            jd_source = await extract_text_from_upload(jd_file)
        except ValueError as exc:
            logger.exception("JD file upload failed")
            raise HTTPException(status_code=400, detail=str(exc))
    elif jd_text:
        jd_source = jd_text

    if jd_source:
        jd_embedding = embed_text(jd_source)
        score = cosine_score(resume_embedding, jd_embedding)

        collection = get_collection("resumes")
        doc_id = metadata.candidate_name
        collection.add(documents=[resume_text], embeddings=[resume_embedding], ids=[doc_id])
        results = collection.query(query_embeddings=[jd_embedding], n_results=top_k)
        retrieved = "\n".join(" ".join(d) for d in results.get("documents", []))

        try:
            summary = assess_resume_with_jd(jd_source, retrieved or resume_text, top_p=top_p)
            parsed_summary = parse_analysis_response(summary)
        except Exception as exc:
            logger.exception("Resume assessment failed")
            raise HTTPException(status_code=500, detail=str(exc))

        response.update({
            "jd": jd_source,
            "similarity": score,
            "analysis": summary,
            "parsed_analysis": parsed_summary,
        })
    return response
