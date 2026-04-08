from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

ChunkStrategy = Literal["fixed", "semantic", "structure_aware"]


class DocumentSection(BaseModel):
    heading: str
    content: str
    level: int = 1


class ParsedDocument(BaseModel):
    document_id: str
    source_path: Path
    source_type: str
    title: str
    raw_text: str
    cleaned_text: str
    sections: list[DocumentSection] = Field(default_factory=list)
    metadata: dict[str, str | int | float] = Field(default_factory=dict)


class ChunkMetadata(BaseModel):
    document_id: str
    source_path: str
    source_type: str
    title: str
    section_title: str
    chunk_index: int
    chunk_strategy: ChunkStrategy
    char_count: int
    word_count: int


class DocumentChunk(BaseModel):
    chunk_id: str
    text: str
    metadata: ChunkMetadata


class IngestionResult(BaseModel):
    documents_processed: int
    chunks_created: int
    output_path: Path

