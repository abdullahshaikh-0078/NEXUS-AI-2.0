from __future__ import annotations

from time import perf_counter

from fastapi import FastAPI
from starlette.requests import Request

from rag_service.api.routes.health import router as health_router
from rag_service.core.config import get_settings
from rag_service.core.lifecycle import lifespan
from rag_service.core.logging import configure_logging, get_logger


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings)

    app = FastAPI(
        title=settings.app.name,
        version=settings.app.version,
        lifespan=lifespan,
    )

    logger = get_logger(__name__)

    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        start_time = perf_counter()
        response = await call_next(request)
        duration_ms = round((perf_counter() - start_time) * 1000, 2)
        logger.info(
            "request_completed",
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            duration_ms=duration_ms,
        )
        return response

    app.include_router(health_router)

    logger.info(
        "application_created",
        app_name=settings.app.name,
        environment=settings.app.env,
        version=settings.app.version,
    )

    return app
