from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field

from rag_service.ingestion.models import ChunkMetadata


def utc_now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


class IndexedChunkRecord(BaseModel):
    row_id: int
    chunk_id: str
    text: str
    metadata: ChunkMetadata


class MappingRecord(BaseModel):
    row_id: int
    chunk_id: str
    document_id: str
    source_path: str
    title: str
    section_title: str


class EmbeddingRecord(BaseModel):
    row_id: int
    chunk_id: str
    values: list[float]


class EmbeddingManifest(BaseModel):
    version: str
    provider: str
    model_name: str
    dimensions: int
    batch_size: int
    normalized: bool = True
    source_chunks_path: str
    embeddings_path: str
    created_at: str = Field(default_factory=utc_now_iso)


class DenseIndexManifest(BaseModel):
    backend: str
    index_type: str
    metric: str
    index_path: str
    vectors_indexed: int
    dimensions: int


class SparseIndexManifest(BaseModel):
    backend: str
    index_path: str
    documents_indexed: int
    average_document_length: float
    k1: float
    b: float


class BuildManifest(BaseModel):
    version: str
    total_chunks: int
    version_dir: str
    source_chunks_path: str
    mapping_path: str
    embeddings: EmbeddingManifest
    dense: DenseIndexManifest
    sparse: SparseIndexManifest
    created_at: str = Field(default_factory=utc_now_iso)


class IndexingResult(BaseModel):
    version: str
    total_chunks: int
    version_dir: Path
    manifest_path: Path


class SearchHit(BaseModel):
    row_id: int
    chunk_id: str
    score: float
