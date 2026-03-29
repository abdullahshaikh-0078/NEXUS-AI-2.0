from __future__ import annotations

from fastapi import APIRouter

from rag_service.core.config import get_settings

router = APIRouter(tags=["health"])


@router.get("/health", summary="Service health check")
async def healthcheck() -> dict[str, str]:
    settings = get_settings()
    return {
        "status": "ok",
        "service": settings.app.name,
        "environment": settings.app.env,
        "version": settings.app.version,
    }

