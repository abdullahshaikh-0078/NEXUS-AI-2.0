from __future__ import annotations

from fastapi import APIRouter, Request

router = APIRouter(tags=["metrics"])


@router.get("/metrics", summary="Runtime service metrics")
async def metrics(request: Request) -> dict[str, object]:
    metrics_registry = getattr(request.app.state, "metrics_registry", None)
    query_service = getattr(request.app.state, "query_service", None)
    admission_controller = getattr(request.app.state, "admission_controller", None)
    if metrics_registry is None:
        return {"status": "disabled"}

    payload = metrics_registry.snapshot()
    if query_service is not None:
        payload["runtime"] = query_service.runtime_snapshot()
    if admission_controller is not None:
        payload.setdefault("runtime", {})["scaling"] = admission_controller.snapshot().__dict__
    return payload
