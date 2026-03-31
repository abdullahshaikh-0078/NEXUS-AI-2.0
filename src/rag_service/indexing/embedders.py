from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Sequence
from typing import Protocol

from rag_service.core.config import Settings
from rag_service.core.logging import get_logger

logger = get_logger(__name__)

TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+")


class TextEmbedder(Protocol):
    provider_name: str
    model_name: str
    dimensions: int
    batch_size: int
    normalized: bool

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        ...


class HashEmbeddingEmbedder:
    provider_name = "hash"
    model_name = "hash-bow-v1"
    normalized = True

    def __init__(self, dimensions: int, batch_size: int) -> None:
        self.dimensions = dimensions
        self.batch_size = batch_size

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        tokens = TOKEN_PATTERN.findall(text.lower()) or ["__empty__"]
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            for offset in range(0, 16, 4):
                bucket = int.from_bytes(digest[offset : offset + 4], "little") % self.dimensions
                sign = 1.0 if digest[offset] % 2 == 0 else -1.0
                vector[bucket] += sign

        magnitude = math.sqrt(sum(value * value for value in vector))
        if magnitude == 0:
            return vector
        return [value / magnitude for value in vector]


class SentenceTransformerBGEEmbedder:
    provider_name = "sentence_transformers"
    normalized = True

    def __init__(self, model_name: str, dimensions: int, batch_size: int) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "sentence-transformers is required for the configured embedding provider"
            ) from exc

        self.model_name = model_name
        self.dimensions = dimensions
        self.batch_size = batch_size
        self._model = SentenceTransformer(model_name)

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        embeddings = self._model.encode(
            list(texts),
            batch_size=self.batch_size,
            normalize_embeddings=self.normalized,
            show_progress_bar=False,
        )
        return [[float(value) for value in row] for row in embeddings]


def create_embedder(settings: Settings) -> TextEmbedder:
    provider = settings.indexing.embedding_provider
    if provider == "sentence_transformers":
        try:
            return SentenceTransformerBGEEmbedder(
                model_name=settings.indexing.embedding_model_name,
                dimensions=settings.indexing.embedding_dimensions,
                batch_size=settings.indexing.embedding_batch_size,
            )
        except ImportError as exc:
            fallback = settings.indexing.embedding_fallback_provider
            logger.warning(
                "embedding_provider_fallback",
                preferred=provider,
                fallback=fallback,
                reason=str(exc),
            )
            provider = fallback

    if provider == "hash":
        return HashEmbeddingEmbedder(
            dimensions=settings.indexing.embedding_dimensions,
            batch_size=settings.indexing.embedding_batch_size,
        )

    raise ValueError(f"Unsupported embedding provider: {provider}")
