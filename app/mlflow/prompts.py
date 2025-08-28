"""Central registry for prompt templates used in MLflow logging."""
from pathlib import Path

PROMPTS_DIR = Path(__file__).resolve().parents[1] / "prompts"

PROMPT_REGISTRY = {
    "generate_jd": (PROMPTS_DIR / "jd_prompt.txt").read_text(),
    "generate_remark": (PROMPTS_DIR / "remarks_prompt.txt").read_text(),
    "interview_email": (PROMPTS_DIR / "interview_email_prompt.txt").read_text(),
    "rejection_email": (PROMPTS_DIR / "rejection_email_prompt.txt").read_text(),
    "match_resume": (PROMPTS_DIR / "matching_prompt.txt").read_text(),
}

