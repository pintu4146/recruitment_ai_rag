"""Utilities to parse job description responses."""
from typing import List, Dict


def parse_jd_response(text: str) -> Dict[str, List[str]]:
    """Naively parse bullet points from the JD text."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    bullets = [line.lstrip("*- ") for line in lines if line.startswith(("- ", "* "))]
    return {"full_text": text, "bullet_points": bullets}


def parse_analysis_response(text: str) -> Dict[str, List[str]]:
    """Parse resume analysis text into paragraphs and bullet lists."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    paragraphs: List[str] = []
    bullets: List[str] = []
    for line in lines:
        if line.startswith(("- ", "* ")):
            bullets.append(line.lstrip("*- "))
        else:
            paragraphs.append(line)
    return {"paragraphs": paragraphs, "bullets": bullets}
