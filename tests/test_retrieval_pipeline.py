from __future__ import annotations

from pathlib import Path

from rag_service.core.config import Settings
from rag_service.indexing.models import SearchHit
from rag_service.indexing.pipeline import build_indexes
from rag_service.ingestion.pipeline import ingest_directory
from rag_service.retrieval.fusion import reciprocal_rank_fusion
from rag_service.retrieval.pipeline import hybrid_retrieve


def test_hybrid_retrieve_returns_dense_sparse_and_fused_hits(
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
            "embedding_version": "phase5-test",
            "dense_backend": "native",
            "dense_fallback_backend": "native",
            "sparse_backend": "native",
            "sparse_fallback_backend": "native",
        },
        retrieval={
            "manifest_path": str(tmp_path / "indexes" / "phase5-test" / "manifest.json"),
            "dense_top_k": 4,
            "sparse_top_k": 4,
            "candidate_pool_size": 5,
            "fused_top_k": 3,
            "rrf_k": 40,
        },
        query={
            "enable_llm_fallback": False,
            "enable_expansion": True,
        },
    )

    chunks_path = tmp_path / "chunks.jsonl"
    ingest_directory(sample_document_dir, chunks_path, settings)
    result = build_indexes(chunks_file=chunks_path, output_dir=tmp_path / "indexes", settings=settings)

    retrieval = hybrid_retrieve(
        "hybrid retrieval metadata",
        settings=settings,
        manifest_path=result.manifest_path,
    )

    assert retrieval.dense_hits
    assert retrieval.sparse_hits
    assert retrieval.fused_hits
    assert retrieval.processed_query.rewrite_strategy == "rule_based"
    assert retrieval.fused_hits[0].chunk_id in {hit.chunk_id for hit in retrieval.dense_hits + retrieval.sparse_hits}
    assert any("retrieval" in hit.text.lower() for hit in retrieval.fused_hits)


def test_reciprocal_rank_fusion_promotes_shared_hits() -> None:
    dense_hits = [
        SearchHit(row_id=0, chunk_id="chunk-a", score=0.90),
        SearchHit(row_id=1, chunk_id="chunk-b", score=0.85),
    ]
    sparse_hits = [
        SearchHit(row_id=2, chunk_id="chunk-c", score=7.20),
        SearchHit(row_id=0, chunk_id="chunk-a", score=6.80),
    ]

    fused_hits = reciprocal_rank_fusion(dense_hits, sparse_hits, rrf_k=60, top_k=3)

    assert fused_hits[0].chunk_id == "chunk-a"
    assert {hit.chunk_id for hit in fused_hits} == {"chunk-a", "chunk-b", "chunk-c"}
