from __future__ import annotations

from pydantic import BaseModel, Field

from rag_service.reranking.models import RerankingResult


class ContextBlock(BaseModel):
    chunk_id: str
    title: str
    source_path: str
    section_title: str
    rerank_score: float
    token_count: int
    compressed_text: str
    original_text: str


class ContextPackage(BaseModel):
    reranking: RerankingResult
    selected_blocks: list[ContextBlock] = Field(default_factory=list)
    total_tokens: int = 0
    omitted_chunk_ids: list[str] = Field(default_factory=list)
    context_text: str = ""
