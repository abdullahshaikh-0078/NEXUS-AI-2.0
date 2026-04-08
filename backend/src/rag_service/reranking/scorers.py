from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence
from typing import Protocol

from rag_service.core.config import Settings
from rag_service.core.logging import get_logger
from rag_service.retrieval.models import RetrievedChunk

logger = get_logger(__name__)
TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+")


class CandidateReranker(Protocol):
    provider_name: str
    model_name: str

    def score(self, query: str, candidates: Sequence[RetrievedChunk]) -> list[float]:
        ...


class HeuristicReranker:
    provider_name = "heuristic"
    model_name = "heuristic-overlap-v1"

    def score(self, query: str, candidates: Sequence[RetrievedChunk]) -> list[float]:
        query_tokens = _tokenize(query)
        query_counts = Counter(query_tokens)
        scores: list[float] = []
        for candidate in candidates:
            candidate_tokens = _tokenize(candidate.text)
            candidate_counts = Counter(candidate_tokens)
            overlap = sum(min(query_counts[token], candidate_counts[token]) for token in query_counts)
            lexical_coverage = overlap / max(len(query_tokens), 1)
            retrieval_signal = 0.15 * _safe_log1p(max(candidate.score, 0.0))
            dense_bonus = 0.05 if candidate.dense_rank is not None else 0.0
            sparse_bonus = 0.05 if candidate.sparse_rank is not None else 0.0
            scores.append(lexical_coverage + retrieval_signal + dense_bonus + sparse_bonus)
        return scores


class CrossEncoderReranker:
    provider_name = "cross_encoder"

    def __init__(self, model_name: str) -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:  # pragma: no cover
            raise ImportError("sentence-transformers is required for cross-encoder reranking") from exc

        self.model_name = model_name
        self._model = CrossEncoder(model_name)

    def score(self, query: str, candidates: Sequence[RetrievedChunk]) -> list[float]:
        pairs = [[query, candidate.text] for candidate in candidates]
        scores = self._model.predict(pairs, show_progress_bar=False)
        return [float(score) for score in scores]


def create_reranker(settings: Settings) -> CandidateReranker:
    provider = settings.reranking.provider
    if provider == "cross_encoder":
        try:
            return CrossEncoderReranker(settings.reranking.model_name)
        except Exception as exc:  # pragma: no cover
            fallback = settings.reranking.fallback_provider
            logger.warning(
                "reranker_provider_fallback",
                preferred=provider,
                fallback=fallback,
                reason=str(exc),
            )
            provider = fallback

    if provider == "heuristic":
        return HeuristicReranker()

    raise ValueError(f"Unsupported reranking provider: {provider}")


def _tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


def _safe_log1p(value: float) -> float:
    return math.log1p(value) if value > -1 else 0.0
