from __future__ import annotations

from pydantic import BaseModel, Field

from rag_service.retrieval.models import HybridRetrievalResult, RetrievedChunk


class RerankedChunk(BaseModel):
    row_id: int
    chunk_id: str
    rerank_score: float
    text: str
    metadata: object
    source: str
    retrieval_score: float
    retrieval_rank: int | None = None
    dense_rank: int | None = None
    sparse_rank: int | None = None


class RerankingResult(BaseModel):
    retrieval: HybridRetrievalResult
    provider: str
    model_name: str
    candidate_count: int
    reranked_hits: list[RerankedChunk] = Field(default_factory=list)
