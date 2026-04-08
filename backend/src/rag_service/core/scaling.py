from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass

from rag_service.core.exceptions import BackpressureError


@dataclass
class ScalingSnapshot:
    max_concurrent_queries: int
    in_flight_queries: int
    queued_capacity: int


class QueryAdmissionController:
    def __init__(self, max_concurrent_queries: int, acquire_timeout_seconds: float) -> None:
        self._max_concurrent_queries = max(1, max_concurrent_queries)
        self._acquire_timeout_seconds = max(0.05, acquire_timeout_seconds)
        self._semaphore = asyncio.Semaphore(self._max_concurrent_queries)
        self._in_flight_queries = 0
        self._lock = asyncio.Lock()

    @asynccontextmanager
    async def acquire(self):
        try:
            await asyncio.wait_for(self._semaphore.acquire(), timeout=self._acquire_timeout_seconds)
        except TimeoutError as exc:
            raise BackpressureError() from exc

        async with self._lock:
            self._in_flight_queries += 1

        try:
            yield
        finally:
            async with self._lock:
                self._in_flight_queries = max(0, self._in_flight_queries - 1)
            self._semaphore.release()

    def snapshot(self) -> ScalingSnapshot:
        queued_capacity = max(0, self._max_concurrent_queries - self._in_flight_queries)
        return ScalingSnapshot(
            max_concurrent_queries=self._max_concurrent_queries,
            in_flight_queries=self._in_flight_queries,
            queued_capacity=queued_capacity,
        )
