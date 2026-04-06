from __future__ import annotations

import uvicorn

from rag_service.core.config import get_settings


def run() -> None:
    settings = get_settings()
    reload_enabled = settings.app.env == "development"
    uvicorn.run(
        "rag_service.api.app:create_app",
        factory=True,
        host=settings.app.host,
        port=settings.app.port,
        reload=reload_enabled,
        workers=1 if reload_enabled else settings.scaling.worker_processes,
    )
