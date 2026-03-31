from __future__ import annotations

import json
from pathlib import Path

from rag_service.core.config import Settings
from rag_service.core.logging import get_logger
from rag_service.ingestion.chunkers import chunk_document
from rag_service.ingestion.models import ChunkStrategy, DocumentChunk, IngestionResult
from rag_service.ingestion.parsers import parse_document

SUPPORTED_SUFFIXES = {".txt", ".html", ".pdf"}

logger = get_logger(__name__)


def ingest_directory(
    input_dir: Path,
    output_path: Path,
    settings: Settings,
    strategy: ChunkStrategy | None = None,
) -> IngestionResult:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    chunk_strategy = strategy or settings.ingestion.default_strategy
    source_files = sorted(
        path for path in input_dir.rglob("*") if path.suffix.lower() in SUPPORTED_SUFFIXES
    )
    chunks: list[DocumentChunk] = []

    for path in source_files:
        document = parse_document(path)
        document_chunks = chunk_document(
            document=document,
            strategy=chunk_strategy,
            chunk_size=settings.ingestion.chunk_size,
            chunk_overlap=settings.ingestion.chunk_overlap,
            semantic_similarity_threshold=settings.ingestion.semantic_similarity_threshold,
        )
        chunks.extend(document_chunks)
        logger.info(
            "document_ingested",
            document_id=document.document_id,
            source_path=str(path),
            sections=len(document.sections),
            chunks=len(document_chunks),
            strategy=chunk_strategy,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(chunk.model_dump(mode="json")) + "\n")

    logger.info(
        "ingestion_completed",
        input_dir=str(input_dir),
        output_path=str(output_path),
        documents_processed=len(source_files),
        chunks_created=len(chunks),
        strategy=chunk_strategy,
    )

    return IngestionResult(
        documents_processed=len(source_files),
        chunks_created=len(chunks),
        output_path=output_path,
    )
