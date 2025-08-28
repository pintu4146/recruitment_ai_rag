"""Naive skill extraction utilities."""

import re
from typing import List

from app.core.logger import logger

# Simple regex to capture words or phrases that look like skills (e.g. Python, C++)
_SKILL_RE = re.compile(r"[A-Za-z\+\#]{2,}(?:\s+[A-Za-z\+\#]{2,})*")


def extract_skills(text: str) -> List[str]:
    """Extract a list of potential skills from text using regex."""
    logger.debug("Extracting skills from text")
    matches = _SKILL_RE.findall(text)
    unique = {m.strip() for m in matches if m.strip()}
    return sorted(unique, key=str.lower)
