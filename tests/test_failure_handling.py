from __future__ import annotations

from pathlib import Path

import pytest

from rag_service.core.cache import InMemoryCache
from rag_service.core.config import Settings
from rag_service.core.metrics import MetricsRegistry
from rag_service.core.scaling import QueryAdmissionController
from rag_service.indexing.models import SearchHit
from rag_service.indexing.pipeline import build_indexes
from rag_service.ingestion.pipeline import ingest_directory
from rag_service.services.query_service import QueryService


@pytest.mark.asyncio
async def test_query_service_degrades_to_sparse_when_dense_search_fails(
    sample_document_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = Settings(
        ingestion={
            "input_dir": str(sample_document_dir),
            "output_dir": str(tmp_path),
            "default_strategy": "structure_aware",
            "chunk_size": 120,
            "chunk_overlap": 30,
        },
        indexing={
            "input_chunk_file": str(tmp_path / "chunks.jsonl"),
            "output_dir": str(tmp_path / "indexes"),
            "embedding_provider": "hash",
            "embedding_fallback_provider": "hash",
            "embedding_dimensions": 64,
            "embedding_batch_size": 2,
            "embedding_version": "phase13-test",
            "dense_backend": "native",
            "dense_fallback_backend": "native",
            "sparse_backend": "native",
            "sparse_fallback_backend": "native",
        },
        query={"enable_llm_fallback": False},
        retrieval={
            "manifest_path": str(tmp_path / "indexes" / "phase13-test" / "manifest.json"),
            "dense_top_k": 4,
            "sparse_top_k": 4,
            "candidate_pool_size": 5,
            "fused_top_k": 4,
        },
        reranking={"provider": "heuristic", "fallback_provider": "heuristic", "top_k": 4, "candidate_limit": 4},
        generation={"provider": "heuristic", "fallback_provider": "heuristic"},
        resilience={"allow_partial_retrieval": True},
        scaling={"max_concurrent_queries": 2, "acquire_timeout_seconds": 0.5},
    )

    chunks_path = tmp_path / "chunks.jsonl"
    ingest_directory(sample_document_dir, chunks_path, settings)
    build_indexes(chunks_file=chunks_path, output_dir=tmp_path / "indexes", settings=settings)

    service = QueryService(
        settings=settings,
        cache_backend=InMemoryCache(),
        metrics=MetricsRegistry(),
        admission_controller=QueryAdmissionController(2, 0.5),
    )

    def fail_dense(*args, **kwargs):
        raise RuntimeError("dense backend offline")

    monkeypatch.setattr("rag_service.services.query_service.search_dense_index", fail_dense)

    response = await service.answer("hybrid retrieval metadata")

    assert response.answer
    assert any("Dense retrieval was unavailable" in warning for warning in response.warnings)


def test_circuit_breaker_snapshots_are_available() -> None:
    from rag_service.core.resilience import CircuitBreaker

    breaker = CircuitBreaker(failure_threshold=2, reset_timeout_seconds=30.0)
    snapshot = breaker.snapshot()
    assert snapshot["open"] is False
    assert snapshot["failure_count"] == 0
