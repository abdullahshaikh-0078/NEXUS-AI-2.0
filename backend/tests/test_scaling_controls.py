from __future__ import annotations

import asyncio

import pytest

from rag_service.core.exceptions import BackpressureError
from rag_service.core.scaling import QueryAdmissionController


@pytest.mark.asyncio
async def test_admission_controller_applies_backpressure() -> None:
    controller = QueryAdmissionController(max_concurrent_queries=1, acquire_timeout_seconds=0.05)

    async with controller.acquire():
        with pytest.raises(BackpressureError):
            async with controller.acquire():
                raise AssertionError("This branch should not run")

    snapshot = controller.snapshot()
    assert snapshot.max_concurrent_queries == 1
    assert snapshot.in_flight_queries == 0
    assert snapshot.queued_capacity == 1
