from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from rag_service.api.app import create_app
from rag_service.core.config import get_settings


@pytest.fixture(autouse=True)
def clear_settings_cache() -> None:
    get_settings.cache_clear()


@pytest.fixture()
def client() -> TestClient:
    return TestClient(create_app())

