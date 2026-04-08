from __future__ import annotations

from pathlib import Path

import pytest

from rag_service.core.cache import InMemoryCache
from rag_service.core.config import Settings
from rag_service.core.metrics import MetricsRegistry
from rag_service.core.scaling import QueryAdmissionController
from rag_service.indexing.pipeline import build_indexes
from rag_service.ingestion.pipeline import ingest_directory
from rag_service.services.query_service import QueryService


@pytest.mark.asyncio
async def test_query_service_returns_structured_response_and_cache_hits(
    sample_document_dir: Path,
    tmp_path: Path,
) -> None:
    settings = Settings(
        ingestion={
            "input_dir": str(sample_document_dir),
            "output_dir": str(tmp_path),
            "default_strategy": "structure_aware",
            "chunk_size": 120,
            "chunk_overlap": 30,
            "semantic_similarity_threshold": 0.15,
        },
        indexing={
            "input_chunk_file": str(tmp_path / "chunks.jsonl"),
            "output_dir": str(tmp_path / "indexes"),
            "embedding_provider": "hash",
            "embedding_fallback_provider": "hash",
            "embedding_dimensions": 64,
            "embedding_batch_size": 2,
            "embedding_version": "phase10-test",
            "dense_backend": "native",
            "dense_fallback_backend": "native",
            "sparse_backend": "native",
            "sparse_fallback_backend": "native",
        },
        query={"enable_llm_fallback": False},
        retrieval={
            "manifest_path": str(tmp_path / "indexes" / "phase10-test" / "manifest.json"),
            "dense_top_k": 4,
            "sparse_top_k": 4,
            "candidate_pool_size": 5,
            "fused_top_k": 4,
            "rrf_k": 40,
        },
        reranking={
            "provider": "heuristic",
            "fallback_provider": "heuristic",
            "top_k": 4,
            "candidate_limit": 4,
            "normalize_scores": True,
        },
        context={
            "max_context_tokens": 90,
            "max_chunks": 3,
            "min_rerank_score": 0.0,
            "per_chunk_token_limit": 35,
            "deduplicate_by_document": True,
            "deduplicate_by_text": True,
            "compression_strategy": "extractive",
            "include_metadata_headers": True,
        },
        generation={
            "provider": "heuristic",
            "fallback_provider": "heuristic",
            "max_citations": 3,
            "temperature": 0.0,
            "max_output_tokens": 320,
        },
        cache={
            "provider": "memory",
            "query_ttl_seconds": 300,
            "retrieval_ttl_seconds": 300,
            "embedding_ttl_seconds": 300,
        },
        scaling={
            "max_concurrent_queries": 2,
            "acquire_timeout_seconds": 0.5,
        },
    )

    chunks_path = tmp_path / "chunks.jsonl"
    ingest_directory(sample_document_dir, chunks_path, settings)
    build_indexes(chunks_file=chunks_path, output_dir=tmp_path / "indexes", settings=settings)

    metrics = MetricsRegistry()
    service = QueryService(
        settings=settings,
        cache_backend=InMemoryCache(),
        metrics=metrics,
        admission_controller=QueryAdmissionController(2, 0.5),
    )

    first_response = await service.answer("hybrid retrieval metadata")
    second_response = await service.answer("hybrid retrieval metadata")

    assert first_response.answer
    assert first_response.citations
    assert first_response.cache_hit is False
    assert second_response.cache_hit is True
    assert second_response.answer == first_response.answer

    snapshot = metrics.snapshot()
    assert snapshot["total_requests"] == 2
    assert snapshot["cache_hits"] == 1
    assert "retrieval" in snapshot["stage_metrics"]
