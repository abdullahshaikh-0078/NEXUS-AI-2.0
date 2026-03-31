from __future__ import annotations

import json
from pathlib import Path

from rag_service.core.config import Settings
from rag_service.core.logging import get_logger
from rag_service.indexing.dense import DenseBackend, create_dense_backend
from rag_service.indexing.embedders import TextEmbedder, create_embedder
from rag_service.indexing.loaders import load_chunks, write_jsonl_models
from rag_service.indexing.models import (
    BuildManifest,
    EmbeddingManifest,
    EmbeddingRecord,
    IndexedChunkRecord,
    IndexingResult,
    MappingRecord,
)
from rag_service.indexing.sparse import SparseBackend, create_sparse_backend

logger = get_logger(__name__)


def build_indexes(
    chunks_file: Path,
    output_dir: Path,
    settings: Settings,
    version: str | None = None,
    embedder: TextEmbedder | None = None,
    dense_backend: DenseBackend | None = None,
    sparse_backend: SparseBackend | None = None,
) -> IndexingResult:
    if not chunks_file.exists():
        raise FileNotFoundError(f"Chunk file does not exist: {chunks_file}")

    chunks = load_chunks(chunks_file)
    indexed_chunks = [
        IndexedChunkRecord(
            row_id=row_id,
            chunk_id=chunk.chunk_id,
            text=chunk.text,
            metadata=chunk.metadata,
        )
        for row_id, chunk in enumerate(chunks)
    ]

    index_version = version or settings.indexing.embedding_version
    version_dir = output_dir / index_version
    version_dir.mkdir(parents=True, exist_ok=True)

    embedder = embedder or create_embedder(settings)
    dense_backend = dense_backend or create_dense_backend(settings)
    sparse_backend = sparse_backend or create_sparse_backend(settings)

    embeddings = _embed_records(indexed_chunks, embedder)
    chunk_ids = [record.chunk_id for record in indexed_chunks]
    texts = [record.text for record in indexed_chunks]

    embeddings_path = version_dir / "embeddings.jsonl"
    mapping_path = version_dir / "id_mapping.json"
    dense_output_path = version_dir / (
        "dense.index" if dense_backend.backend_name == "faiss" else "dense_index.json"
    )
    sparse_output_path = version_dir / (
        "sparse_index" if sparse_backend.backend_name == "whoosh" else "sparse_index.json"
    )
    manifest_path = version_dir / "manifest.json"

    write_jsonl_models(embeddings_path, embeddings)
    mapping_records = [
        MappingRecord(
            row_id=record.row_id,
            chunk_id=record.chunk_id,
            document_id=record.metadata.document_id,
            source_path=record.metadata.source_path,
            title=record.metadata.title,
            section_title=record.metadata.section_title,
        )
        for record in indexed_chunks
    ]
    mapping_path.write_text(
        json.dumps([record.model_dump(mode="json") for record in mapping_records], indent=2),
        encoding="utf-8",
    )

    dense_manifest = dense_backend.build(
        embeddings=[record.values for record in embeddings],
        chunk_ids=chunk_ids,
        output_path=dense_output_path,
        settings=settings,
    )
    sparse_manifest = sparse_backend.build(
        texts=texts,
        chunk_ids=chunk_ids,
        output_path=sparse_output_path,
        settings=settings,
    )
    embedding_manifest = EmbeddingManifest(
        version=index_version,
        provider=embedder.provider_name,
        model_name=embedder.model_name,
        dimensions=embedder.dimensions,
        batch_size=embedder.batch_size,
        normalized=embedder.normalized,
        source_chunks_path=str(chunks_file),
        embeddings_path=str(embeddings_path),
    )

    build_manifest = BuildManifest(
        version=index_version,
        total_chunks=len(indexed_chunks),
        version_dir=str(version_dir),
        source_chunks_path=str(chunks_file),
        mapping_path=str(mapping_path),
        embeddings=embedding_manifest,
        dense=dense_manifest,
        sparse=sparse_manifest,
    )
    manifest_path.write_text(build_manifest.model_dump_json(indent=2), encoding="utf-8")

    logger.info(
        "index_build_completed",
        version=index_version,
        total_chunks=len(indexed_chunks),
        embedding_provider=embedder.provider_name,
        dense_backend=dense_manifest.backend,
        sparse_backend=sparse_manifest.backend,
        output_dir=str(version_dir),
    )

    return IndexingResult(
        version=index_version,
        total_chunks=len(indexed_chunks),
        version_dir=version_dir,
        manifest_path=manifest_path,
    )


def _embed_records(
    records: list[IndexedChunkRecord],
    embedder: TextEmbedder,
) -> list[EmbeddingRecord]:
    embeddings: list[EmbeddingRecord] = []
    for batch_start in range(0, len(records), embedder.batch_size):
        batch = records[batch_start : batch_start + embedder.batch_size]
        vectors = embedder.embed_texts([record.text for record in batch])
        embeddings.extend(
            EmbeddingRecord(row_id=record.row_id, chunk_id=record.chunk_id, values=vector)
            for record, vector in zip(batch, vectors, strict=False)
        )
        logger.info(
            "embedding_batch_completed",
            provider=embedder.provider_name,
            start=batch_start,
            batch_size=len(batch),
        )
    return embeddings
