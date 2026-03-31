from __future__ import annotations

import json
from pathlib import Path

from rag_service.core.config import Settings
from rag_service.indexing.dense import search_dense_index
from rag_service.indexing.embedders import HashEmbeddingEmbedder
from rag_service.indexing.loaders import load_manifest
from rag_service.indexing.pipeline import build_indexes
from rag_service.indexing.sparse import search_sparse_index
from rag_service.ingestion.pipeline import ingest_directory


def test_indexing_pipeline_writes_versioned_artifacts(
    sample_document_dir: Path,
    tmp_path: Path,
) -> None:
    chunks_path = _build_chunk_artifact(sample_document_dir, tmp_path)
    settings = Settings(
        indexing={
            "input_chunk_file": str(chunks_path),
            "output_dir": str(tmp_path / "indexes"),
            "embedding_provider": "hash",
            "embedding_fallback_provider": "hash",
            "embedding_dimensions": 64,
            "embedding_batch_size": 2,
            "embedding_version": "phase3-test",
            "dense_backend": "native",
            "dense_fallback_backend": "native",
            "sparse_backend": "native",
            "sparse_fallback_backend": "native",
        }
    )

    result = build_indexes(chunks_file=chunks_path, output_dir=tmp_path / "indexes", settings=settings)

    manifest = load_manifest(result.manifest_path)
    mapping_records = json.loads(Path(manifest.mapping_path).read_text(encoding="utf-8"))
    embeddings_path = Path(manifest.embeddings.embeddings_path)

    assert result.version == "phase3-test"
    assert result.total_chunks == manifest.total_chunks
    assert embeddings_path.exists()
    assert Path(manifest.dense.index_path).exists()
    assert Path(manifest.sparse.index_path).exists()
    assert len(mapping_records) == result.total_chunks
    assert manifest.embeddings.provider == "hash"
    assert manifest.dense.backend == "native"
    assert manifest.sparse.backend == "native"


def test_native_dense_and_sparse_indexes_support_search(
    sample_document_dir: Path,
    tmp_path: Path,
) -> None:
    chunks_path = _build_chunk_artifact(sample_document_dir, tmp_path)
    settings = Settings(
        indexing={
            "input_chunk_file": str(chunks_path),
            "output_dir": str(tmp_path / "indexes"),
            "embedding_provider": "hash",
            "embedding_fallback_provider": "hash",
            "embedding_dimensions": 64,
            "embedding_batch_size": 2,
            "embedding_version": "phase3-search",
            "dense_backend": "native",
            "dense_fallback_backend": "native",
            "sparse_backend": "native",
            "sparse_fallback_backend": "native",
        }
    )

    result = build_indexes(chunks_file=chunks_path, output_dir=tmp_path / "indexes", settings=settings)
    manifest = load_manifest(result.manifest_path)

    query = "hybrid retrieval metadata"
    query_vector = HashEmbeddingEmbedder(dimensions=64, batch_size=2).embed_texts([query])[0]

    dense_hits = search_dense_index(manifest.dense, query_vector=query_vector, top_k=3)
    sparse_hits = search_sparse_index(manifest.sparse, query=query, top_k=3)
    mapping_records = json.loads(Path(manifest.mapping_path).read_text(encoding="utf-8"))
    chunk_ids = {record["chunk_id"] for record in mapping_records}

    assert dense_hits
    assert sparse_hits
    assert dense_hits[0].chunk_id in chunk_ids
    assert sparse_hits[0].chunk_id in chunk_ids


def _build_chunk_artifact(sample_document_dir: Path, tmp_path: Path) -> Path:
    settings = Settings(
        ingestion={
            "input_dir": str(sample_document_dir),
            "output_dir": str(tmp_path),
            "default_strategy": "structure_aware",
            "chunk_size": 120,
            "chunk_overlap": 30,
            "semantic_similarity_threshold": 0.15,
        }
    )
    chunks_path = tmp_path / "chunks.jsonl"
    ingest_directory(sample_document_dir, chunks_path, settings)
    return chunks_path
