from __future__ import annotations

import asyncio
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from rag_service.core.config import Settings
from rag_service.core.logging import get_logger

logger = get_logger(__name__)


class CacheBackend(ABC):
    @abstractmethod
    async def get_json(self, key: str) -> Any | None:
        raise NotImplementedError

    @abstractmethod
    async def set_json(self, key: str, value: Any, ttl_seconds: int | None = None) -> None:
        raise NotImplementedError

    @abstractmethod
    async def close(self) -> None:
        raise NotImplementedError


@dataclass
class CacheEntry:
    value: Any
    expires_at: datetime | None


class InMemoryCache(CacheBackend):
    def __init__(self) -> None:
        self._store: dict[str, CacheEntry] = {}
        self._lock = asyncio.Lock()

    async def get_json(self, key: str) -> Any | None:
        async with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            if entry.expires_at and entry.expires_at <= _utcnow():
                self._store.pop(key, None)
                return None
            return entry.value

    async def set_json(self, key: str, value: Any, ttl_seconds: int | None = None) -> None:
        expires_at = None
        if ttl_seconds is not None:
            expires_at = _utcnow() + timedelta(seconds=ttl_seconds)
        async with self._lock:
            self._store[key] = CacheEntry(value=value, expires_at=expires_at)

    async def close(self) -> None:
        async with self._lock:
            self._store.clear()


class RedisCache(CacheBackend):
    def __init__(self, redis_url: str) -> None:
        try:
            from redis.asyncio import from_url  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover
            raise ImportError("redis is required for the configured cache backend") from exc

        self._client = from_url(redis_url, encoding="utf-8", decode_responses=True)

    async def get_json(self, key: str) -> Any | None:
        payload = await self._client.get(key)
        if payload is None:
            return None
        return json.loads(payload)

    async def set_json(self, key: str, value: Any, ttl_seconds: int | None = None) -> None:
        payload = json.dumps(value)
        await self._client.set(key, payload, ex=ttl_seconds)

    async def close(self) -> None:
        await self._client.aclose()


class NullCache(CacheBackend):
    async def get_json(self, key: str) -> Any | None:  # noqa: ARG002
        return None

    async def set_json(self, key: str, value: Any, ttl_seconds: int | None = None) -> None:  # noqa: ARG002
        return None

    async def close(self) -> None:
        return None


def create_cache_backend(settings: Settings) -> CacheBackend:
    provider = settings.cache.provider.lower()
    if provider == "none":
        return NullCache()
    if provider == "redis":
        try:
            return RedisCache(settings.cache.redis_url)
        except ImportError as exc:
            logger.warning(
                "cache_backend_fallback",
                preferred=provider,
                fallback="memory",
                reason=str(exc),
            )
            return InMemoryCache()
    return InMemoryCache()


class CacheNamespace:
    def __init__(self, backend: CacheBackend, prefix: str, default_ttl_seconds: int) -> None:
        self._backend = backend
        self._prefix = prefix
        self._default_ttl_seconds = default_ttl_seconds

    async def get(self, key: str) -> Any | None:
        return await self._backend.get_json(self._key(key))

    async def set(self, key: str, value: Any, ttl_seconds: int | None = None) -> None:
        ttl = ttl_seconds if ttl_seconds is not None else self._default_ttl_seconds
        await self._backend.set_json(self._key(key), value, ttl_seconds=ttl)

    def _key(self, key: str) -> str:
        return f"{self._prefix}:{key}"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)
