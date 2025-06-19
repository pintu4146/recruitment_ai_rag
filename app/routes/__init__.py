"""Aggregate API and web routers for the application."""

from fastapi import APIRouter

from app.core.logger import logger

from . import jd, resume, web

router = APIRouter()
router.include_router(jd.router, prefix="/jd", tags=["jd"])
router.include_router(resume.router, prefix="/resume", tags=["resume"])
router.include_router(web.router, prefix="/web", tags=["web"])
logger.info("Routers registered")
