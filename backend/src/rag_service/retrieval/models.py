from __future__ import annotations

from pydantic import BaseModel, Field

from rag_service.ingestion.models import ChunkMetadata
from rag_service.indexing.models import SearchHit
from rag_service.query.models import ProcessedQuery


class RetrievedChunk(BaseModel):
    row_id: int
    chunk_id: str
    score: float
    text: str
    metadata: ChunkMetadata
    source: str
    dense_rank: int | None = None
    sparse_rank: int | None = None
    dense_score: float | None = None
    sparse_score: float | None = None


class HybridRetrievalResult(BaseModel):
    processed_query: ProcessedQuery
    dense_hits: list[RetrievedChunk] = Field(default_factory=list)
    sparse_hits: list[RetrievedChunk] = Field(default_factory=list)
    fused_hits: list[RetrievedChunk] = Field(default_factory=list)
    dense_raw_hits: list[SearchHit] = Field(default_factory=list)
    sparse_raw_hits: list[SearchHit] = Field(default_factory=list)
