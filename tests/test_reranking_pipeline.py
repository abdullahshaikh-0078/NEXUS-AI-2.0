from __future__ import annotations

from pathlib import Path

from rag_service.core.config import Settings
from rag_service.indexing.pipeline import build_indexes
from rag_service.ingestion.pipeline import ingest_directory
from rag_service.reranking.pipeline import rerank_candidates
from rag_service.reranking.scorers import HeuristicReranker
from rag_service.retrieval.pipeline import hybrid_retrieve


def test_rerank_candidates_returns_top_k_in_score_order(
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
            "embedding_version": "phase6-test",
            "dense_backend": "native",
            "dense_fallback_backend": "native",
            "sparse_backend": "native",
            "sparse_fallback_backend": "native",
        },
        query={
            "enable_llm_fallback": False,
            "enable_expansion": True,
        },
        retrieval={
            "manifest_path": str(tmp_path / "indexes" / "phase6-test" / "manifest.json"),
            "dense_top_k": 4,
            "sparse_top_k": 4,
            "candidate_pool_size": 5,
            "fused_top_k": 4,
            "rrf_k": 40,
        },
        reranking={
            "provider": "heuristic",
            "fallback_provider": "heuristic",
            "top_k": 3,
            "candidate_limit": 4,
            "normalize_scores": True,
        },
    )

    chunks_path = tmp_path / "chunks.jsonl"
    ingest_directory(sample_document_dir, chunks_path, settings)
    build_indexes(chunks_file=chunks_path, output_dir=tmp_path / "indexes", settings=settings)

    retrieval = hybrid_retrieve(
        "hybrid retrieval metadata",
        settings=settings,
    )
    reranked = rerank_candidates(
        "hybrid retrieval metadata",
        settings=settings,
        retrieval=retrieval,
        reranker=HeuristicReranker(),
    )

    assert reranked.provider == "heuristic"
    assert reranked.candidate_count == min(settings.reranking.candidate_limit, len(retrieval.fused_hits))
    assert len(reranked.reranked_hits) == min(settings.reranking.top_k, reranked.candidate_count)
    assert reranked.reranked_hits == sorted(
        reranked.reranked_hits,
        key=lambda item: item.rerank_score,
        reverse=True,
    )
    assert any("retrieval" in hit.text.lower() for hit in reranked.reranked_hits)


def test_rerank_candidates_uses_retrieval_result_without_requerying(
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
            "embedding_version": "phase6-second",
            "dense_backend": "native",
            "dense_fallback_backend": "native",
            "sparse_backend": "native",
            "sparse_fallback_backend": "native",
        },
        query={
            "enable_llm_fallback": False,
        },
        retrieval={
            "manifest_path": str(tmp_path / "indexes" / "phase6-second" / "manifest.json"),
            "dense_top_k": 4,
            "sparse_top_k": 4,
            "candidate_pool_size": 4,
            "fused_top_k": 4,
            "rrf_k": 40,
        },
        reranking={
            "provider": "heuristic",
            "top_k": 2,
            "candidate_limit": 4,
            "normalize_scores": False,
        },
    )

    chunks_path = tmp_path / "chunks.jsonl"
    ingest_directory(sample_document_dir, chunks_path, settings)
    build_indexes(chunks_file=chunks_path, output_dir=tmp_path / "indexes", settings=settings)
    retrieval = hybrid_retrieve("metadata retrieval", settings=settings)

    reranked = rerank_candidates(
        "metadata retrieval",
        settings=settings,
        retrieval=retrieval,
        reranker=HeuristicReranker(),
    )

    assert reranked.retrieval.processed_query.original_query == "metadata retrieval"
    assert len(reranked.reranked_hits) == 2
    assert all(hit.retrieval_rank is not None for hit in reranked.reranked_hits)
