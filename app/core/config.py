from pathlib import Path
from dotenv import load_dotenv
import os

# Load environment variables from .env.dev at project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ENV_PATH = PROJECT_ROOT / ".env.dev"
load_dotenv(dotenv_path=ENV_PATH)


class Settings:
    """Application settings loaded from environment variables."""

    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    CHROMA_DIR: str = os.getenv("CHROMA_DIR", ".chromadb")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "MiniLM")
    RETRIEVAL_MODEL: str = os.getenv("RETRIEVAL_MODEL", "Gemini")


settings = Settings()
