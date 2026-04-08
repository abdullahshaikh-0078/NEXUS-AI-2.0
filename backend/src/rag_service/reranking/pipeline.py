from __future__ import annotations

from rag_service.core.config import Settings
from rag_service.core.logging import get_logger
from rag_service.reranking.models import RerankedChunk, RerankingResult
from rag_service.reranking.scorers import CandidateReranker, create_reranker
from rag_service.retrieval.models import HybridRetrievalResult, RetrievedChunk
from rag_service.retrieval.pipeline import hybrid_retrieve

logger = get_logger(__name__)


def rerank_candidates(
    query: str,
    settings: Settings,
    retrieval: HybridRetrievalResult | None = None,
    reranker: CandidateReranker | None = None,
) -> RerankingResult:
    retrieval_result = retrieval or hybrid_retrieve(query, settings=settings)
    candidates = retrieval_result.fused_hits[: settings.reranking.candidate_limit]
    reranker = reranker or create_reranker(settings)

    rerank_scores = reranker.score(retrieval_result.processed_query.rewritten_query, candidates)
    reranked_hits = _build_reranked_hits(candidates, rerank_scores, settings=settings)

    logger.info(
        "reranking_completed",
        provider=reranker.provider_name,
        model_name=reranker.model_name,
        candidate_count=len(candidates),
        returned_count=len(reranked_hits),
    )

    return RerankingResult(
        retrieval=retrieval_result,
        provider=reranker.provider_name,
        model_name=reranker.model_name,
        candidate_count=len(candidates),
        reranked_hits=reranked_hits,
    )


def _build_reranked_hits(
    candidates: list[RetrievedChunk],
    rerank_scores: list[float],
    settings: Settings,
) -> list[RerankedChunk]:
    if not candidates:
        return []

    paired = list(zip(candidates, rerank_scores, strict=False))
    if settings.reranking.normalize_scores:
        paired = [(candidate, _normalize_score(score, rerank_scores)) for candidate, score in paired]

    ranked = sorted(paired, key=lambda item: item[1], reverse=True)[: settings.reranking.top_k]
    retrieval_rank_lookup = {candidate.chunk_id: index for index, candidate in enumerate(candidates, start=1)}

    return [
        RerankedChunk(
            row_id=candidate.row_id,
            chunk_id=candidate.chunk_id,
            rerank_score=score,
            text=candidate.text,
            metadata=candidate.metadata,
            source=candidate.source,
            retrieval_score=candidate.score,
            retrieval_rank=retrieval_rank_lookup.get(candidate.chunk_id),
            dense_rank=candidate.dense_rank,
            sparse_rank=candidate.sparse_rank,
        )
        for candidate, score in ranked
    ]


def _normalize_score(score: float, all_scores: list[float]) -> float:
    minimum = min(all_scores)
    maximum = max(all_scores)
    if maximum == minimum:
        return 1.0
    return (score - minimum) / (maximum - minimum)
