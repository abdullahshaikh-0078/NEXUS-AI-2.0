from __future__ import annotations

from fastapi import APIRouter, Request

router = APIRouter(prefix="/api/v1", tags=["metrics"])


@router.get("/metrics", summary="Runtime service metrics")
async def metrics(request: Request) -> dict[str, object]:
    metrics_registry = getattr(request.app.state, "metrics_registry", None)
    if metrics_registry is None:
        return {"status": "disabled"}
    return metrics_registry.snapshot()
