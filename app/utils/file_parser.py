import io

import fitz
from docx import Document
from fastapi import UploadFile


async def extract_text_from_upload(file: UploadFile) -> str:
    """Extract text from an uploaded PDF or DOCX file."""
    contents = await file.read()
    filename = file.filename.lower()
    if filename.endswith(".pdf"):
        return _extract_pdf(contents)
    if filename.endswith(".docx"):
        return _extract_docx(contents)
    raise ValueError("Unsupported file type. Only PDF and DOCX are allowed")


def _extract_pdf(data: bytes) -> str:
    with fitz.open(stream=data, filetype="pdf") as doc:
        texts = [page.get_text() for page in doc]
    return "\n".join(texts)


def _extract_docx(data: bytes) -> str:
    doc = Document(io.BytesIO(data))
    return "\n".join(p.text for p in doc.paragraphs)
