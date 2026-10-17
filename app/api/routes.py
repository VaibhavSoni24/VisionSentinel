"""API route definitions for Vision Sentinel."""

from fastapi import APIRouter

from app.config import settings

router = APIRouter(prefix="/api", tags=["Analysis"])


@router.get("/ping")
async def ping():
    return {"message": "pong", "operating_mode": settings.get_operating_mode()}
