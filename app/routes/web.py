from fastapi import APIRouter, Request, UploadFile, File, Form, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse

from app.models.resume import ResumeMetadata
from app.utils.file_parser import extract_text_from_upload
from app.services.embedding import embed_text
from app.services.matching import cosine_score
from app.db.chroma_store import get_collection
from app.llm import assess_resume_with_jd
from app.llm.output_parser import parse_analysis_response

router = APIRouter()

templates = Jinja2Templates(directory="app/templates")


@router.get("/resume", response_class=HTMLResponse)
async def resume_form(request: Request):
    return templates.TemplateResponse("form.html", {"request": request})


@router.post("/resume", response_class=HTMLResponse)
async def resume_submit(
    request: Request,
    file: UploadFile = File(...),
    candidate_name: str = Form(...),
    email: str | None = Form(None),
    notes: str | None = Form(None),
    jd_file: UploadFile | None = File(None),
    jd_text: str | None = Form(None),
    top_k: int = Form(1),
    top_p: float = Form(0.8),
):
    try:
        resume_text = await extract_text_from_upload(file)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    metadata = ResumeMetadata(candidate_name=candidate_name, email=email, notes=notes)
    resume_embedding = embed_text(resume_text)
    response = {
        "metadata": metadata.model_dump(),
        "text": resume_text,
    }

    jd_source = None
    if jd_file is not None:
        try:
            jd_source = await extract_text_from_upload(jd_file)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
    elif jd_text:
        jd_source = jd_text

    if jd_source:
        jd_embedding = embed_text(jd_source)
        score = cosine_score(resume_embedding, jd_embedding)

        collection = get_collection("resumes")
        doc_id = candidate_name
        collection.add(documents=[resume_text], embeddings=[resume_embedding], ids=[doc_id])
        results = collection.query(query_embeddings=[jd_embedding], n_results=top_k)
        retrieved = "\n".join(" ".join(d) for d in results.get("documents", []))

        summary = assess_resume_with_jd(jd_source, retrieved or resume_text, top_p=top_p)
        parsed_summary = parse_analysis_response(summary)

        response.update(
            jd=jd_source,
            similarity=score,
            analysis=summary,
            parsed_analysis=parsed_summary,
        )

    return templates.TemplateResponse(
        "results.html", {"request": request, "result": response}
    )
