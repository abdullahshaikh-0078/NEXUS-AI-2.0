from __future__ import annotations

import uvicorn

from rag_service.core.config import get_settings


def run() -> None:
    settings = get_settings()
    uvicorn.run(
        "rag_service.api.app:create_app",
        factory=True,
        host=settings.app.host,
        port=settings.app.port,
        reload=settings.app.env == "development",
    )

