from __future__ import annotations

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(min_length=2, max_length=4000)


class QueryCitation(BaseModel):
    id: str
    text: str
    source: str
    score: float
    title: str = ""
    section_title: str = ""
    source_path: str = ""
    reference_text: str = ""


class RetrievedDocument(BaseModel):
    chunk_id: str
    text: str
    source: str
    score: float
    title: str = ""
    section_title: str = ""
    source_path: str = ""


class QueryResponse(BaseModel):
    answer: str
    citations: list[QueryCitation] = Field(default_factory=list)
    latency_ms: float
    confidence_score: float
    confidence_label: str
    references_markdown: str = ""
    stage_latencies_ms: dict[str, float] = Field(default_factory=dict)
    debug_documents: list[RetrievedDocument] = Field(default_factory=list)
    cache_hit: bool = False
