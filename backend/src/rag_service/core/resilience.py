from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from time import monotonic
from typing import TypeVar

from rag_service.core.exceptions import CircuitBreakerOpenError

T = TypeVar("T")


@dataclass
class CircuitState:
    failure_count: int = 0
    opened_at: float | None = None


class CircuitBreaker:
    def __init__(self, failure_threshold: int, reset_timeout_seconds: float) -> None:
        self._failure_threshold = max(1, failure_threshold)
        self._reset_timeout_seconds = max(1.0, reset_timeout_seconds)
        self._state = CircuitState()
        self._lock = asyncio.Lock()

    async def call(self, operation: Callable[[], Awaitable[T]]) -> T:
        async with self._lock:
            if self._is_open_locked():
                raise CircuitBreakerOpenError()

        try:
            result = await operation()
        except Exception:
            await self.record_failure()
            raise

        await self.record_success()
        return result

    async def record_failure(self) -> None:
        async with self._lock:
            self._state.failure_count += 1
            if self._state.failure_count >= self._failure_threshold:
                self._state.opened_at = monotonic()

    async def record_success(self) -> None:
        async with self._lock:
            self._state.failure_count = 0
            self._state.opened_at = None

    def snapshot(self) -> dict[str, int | bool | None]:
        return {
            "failure_count": self._state.failure_count,
            "open": self.is_open(),
            "reset_timeout_seconds": self._reset_timeout_seconds,
        }

    def is_open(self) -> bool:
        return self._is_open_locked()

    def _is_open_locked(self) -> bool:
        if self._state.opened_at is None:
            return False
        if monotonic() - self._state.opened_at >= self._reset_timeout_seconds:
            self._state.failure_count = 0
            self._state.opened_at = None
            return False
        return True


async def retry_async(
    operation: Callable[[], Awaitable[T]],
    *,
    attempts: int,
    base_delay_seconds: float,
) -> T:
    last_error: Exception | None = None
    for attempt in range(1, max(1, attempts) + 1):
        try:
            return await operation()
        except Exception as exc:
            last_error = exc
            if attempt >= attempts:
                break
            await asyncio.sleep(base_delay_seconds * attempt)

    assert last_error is not None
    raise last_error
