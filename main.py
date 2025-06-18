from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings  # noqa: F401 - imported for side effects
from app.core.logger import logger
import app.mlflow.setup  # noqa: F401
from app.routes import router as api_router

app = FastAPI(title="Recruitment AI RAG System", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger.info("Application starting with CHROMA_DIR=%s", settings.CHROMA_DIR)

app.include_router(api_router)


@app.get("/", tags=["health"])
def root():
    """Basic health check endpoint."""
    logger.info("Health check accessed")
    return {"message": "Recruitment AI RAG system is running."}


@app.get("/health", tags=["health"])
def healthcheck():
    """Alias health check endpoint."""
    return {"status": "ok"}
