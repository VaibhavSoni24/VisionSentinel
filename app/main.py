"""Vision Sentinel FastAPI Application Entrypoint."""

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict

import cv2
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("vision_sentinel")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle startup and shutdown handler."""
    logger.info("Initializing Vision Sentinel...")
    logger.info("OpenCV Runtime Version: %s", cv2.__version__)
    logger.info("Perception Engine: %s", settings.get_operating_mode())
    logger.info("Gemini Reasoning Active: %s", settings.is_gemini_active())
    logger.info("Storage Output Dir: %s", settings.outputs_dir)
    yield
    logger.info("Shutting down Vision Sentinel...")


app = FastAPI(
    title="Vision Sentinel",
    description="Evidence-Driven Restricted-Zone Entry Verification with OpenCV 5",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup directories and static files
PROJECT_ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = PROJECT_ROOT / "app" / "static"
TEMPLATES_DIR = PROJECT_ROOT / "app" / "templates"

STATIC_DIR.mkdir(parents=True, exist_ok=True)
TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


@app.get("/health", tags=["System"])
async def health_check() -> Dict[str, Any]:
    """Health check endpoint exposing runtime health and verified OpenCV version."""
    return {
        "status": "healthy",
        "app": "Vision Sentinel",
        "version": "0.1.0",
        "opencv_version": cv2.__version__,
        "is_opencv_5": cv2.__version__.startswith("5."),
    }


@app.get("/api/status", tags=["System"])
async def system_status() -> Dict[str, Any]:
    """Diagnostic system status exposing operating mode and resource limits."""
    return {
        "app_env": settings.app_env,
        "opencv_version": cv2.__version__,
        "operating_mode": settings.get_operating_mode(),
        "local_detector_available": settings.is_model_available(),
        "gemini_active": settings.is_gemini_active(),
        "gemini_model": settings.gemini_model if settings.is_gemini_active() else None,
        "max_upload_mb": settings.max_upload_mb,
        "max_video_seconds": settings.max_video_seconds,
        "cool_enabled": settings.cool_enabled,
    }


# Include API routes (will be populated in Phase 4)
from app.api.routes import router as api_router  # noqa: E402

app.include_router(api_router)
