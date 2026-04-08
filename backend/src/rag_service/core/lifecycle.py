from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from rag_service.core.cache import create_cache_backend
from rag_service.core.config import get_settings
from rag_service.core.logging import get_logger
from rag_service.core.metrics import MetricsRegistry
from rag_service.core.scaling import QueryAdmissionController
from rag_service.services import QueryService

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    cache_backend = create_cache_backend(settings)
    metrics_registry = MetricsRegistry()
    admission_controller = QueryAdmissionController(
        max_concurrent_queries=settings.scaling.max_concurrent_queries,
        acquire_timeout_seconds=settings.scaling.acquire_timeout_seconds,
    )
    app.state.settings = settings
    app.state.cache_backend = cache_backend
    app.state.metrics_registry = metrics_registry
    app.state.admission_controller = admission_controller
    app.state.query_service = QueryService(
        settings=settings,
        cache_backend=cache_backend,
        metrics=metrics_registry,
        admission_controller=admission_controller,
    )

    logger.info("service_startup")
    try:
        yield
    finally:
        await cache_backend.close()
        logger.info("service_shutdown")
