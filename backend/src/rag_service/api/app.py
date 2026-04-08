from __future__ import annotations

from time import perf_counter

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from rag_service.api.routes.health import router as health_router
from rag_service.api.routes.metrics import router as metrics_router
from rag_service.api.routes.query import router as query_router
from rag_service.core.config import get_settings
from rag_service.core.exceptions import RAGServiceError
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
    app.state.settings = settings

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.api.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    logger = get_logger(__name__)

    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        start_time = perf_counter()
        response = await call_next(request)
        duration_ms = round((perf_counter() - start_time) * 1000, 2)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "same-origin"
        response.headers["Cache-Control"] = "no-store"
        logger.info(
            "request_completed",
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            duration_ms=duration_ms,
        )
        return response

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        logger.warning(
            "request_validation_failed",
            path=request.url.path,
            errors=exc.errors(),
        )
        return JSONResponse(
            status_code=422,
            content={"detail": "Invalid request payload.", "errors": exc.errors()},
        )

    @app.exception_handler(RAGServiceError)
    async def rag_exception_handler(request: Request, exc: RAGServiceError):
        logger.warning(
            "handled_service_exception",
            path=request.url.path,
            status_code=exc.status_code,
            detail=exc.detail,
        )
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):
        logger.exception(
            "unhandled_request_exception",
            path=request.url.path,
            error=str(exc),
        )
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error."},
        )

    app.include_router(health_router)
    app.include_router(query_router, prefix=settings.api.prefix)
    app.include_router(metrics_router, prefix=settings.api.prefix)

    logger.info(
        "application_created",
        app_name=settings.app.name,
        environment=settings.app.env,
        version=settings.app.version,
        api_prefix=settings.api.prefix,
    )

    return app
