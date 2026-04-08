from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

from rag_service.core.config import Settings
from rag_service.query.models import ProcessedQuery


@dataclass(frozen=True)
class RetrievalPlan:
    dense_top_k: int
    sparse_top_k: int
    candidate_pool_size: int
    fused_top_k: int
    rerank_candidate_limit: int


def build_retrieval_plan(processed_query: ProcessedQuery, settings: Settings) -> RetrievalPlan:
    token_count = len(processed_query.tokens) or len(processed_query.cleaned_query.split())
    if settings.latency.enable_adaptive_retrieval and token_count <= settings.latency.short_query_token_threshold:
        candidate_pool_size = min(
            settings.latency.short_query_candidate_pool_size,
            settings.retrieval.candidate_pool_size,
        )
        fused_top_k = min(settings.latency.short_query_fused_top_k, settings.retrieval.fused_top_k)
        rerank_limit = min(candidate_pool_size, settings.reranking.candidate_limit)
        return RetrievalPlan(
            dense_top_k=min(settings.latency.short_query_dense_top_k, settings.retrieval.dense_top_k),
            sparse_top_k=min(settings.latency.short_query_sparse_top_k, settings.retrieval.sparse_top_k),
            candidate_pool_size=candidate_pool_size,
            fused_top_k=fused_top_k,
            rerank_candidate_limit=max(1, rerank_limit),
        )

    return RetrievalPlan(
        dense_top_k=settings.retrieval.dense_top_k,
        sparse_top_k=settings.retrieval.sparse_top_k,
        candidate_pool_size=settings.retrieval.candidate_pool_size,
        fused_top_k=settings.retrieval.fused_top_k,
        rerank_candidate_limit=settings.reranking.candidate_limit,
    )


def iter_answer_chunks(answer: str, chunk_size: int) -> Iterator[str]:
    if chunk_size <= 0:
        yield answer
        return

    words = answer.split()
    if not words:
        return

    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        if len(candidate) <= chunk_size:
            current = candidate
            continue
        yield current + " "
        current = word
    yield current
