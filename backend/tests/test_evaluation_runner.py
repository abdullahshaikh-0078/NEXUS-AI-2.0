from __future__ import annotations

from pathlib import Path

from rag_service.core.config import Settings
from rag_service.evaluation.dataset import bootstrap_dataset_from_chunks
from rag_service.evaluation.runner import run_experiment_suite
from rag_service.indexing.pipeline import build_indexes
from rag_service.ingestion.pipeline import ingest_directory


def test_run_experiment_suite_generates_benchmarks_and_artifacts(
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
            "embedding_version": "phase16-test",
            "dense_backend": "native",
            "dense_fallback_backend": "native",
            "sparse_backend": "native",
            "sparse_fallback_backend": "native",
        },
        query={"enable_llm_fallback": False},
        retrieval={
            "manifest_path": str(tmp_path / "indexes" / "phase16-test" / "manifest.json"),
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
        evaluation={
            "dataset_path": str(tmp_path / "eval.jsonl"),
            "output_dir": str(tmp_path / "artifacts"),
            "systems": ["bm25", "dense", "hybrid"],
            "baseline_system": "bm25",
            "k_values": [1, 3],
            "primary_k": 3,
            "generate_plots": True,
        },
    )

    chunks_path = tmp_path / "chunks.jsonl"
    ingest_directory(sample_document_dir, chunks_path, settings)
    indexing = build_indexes(chunks_file=chunks_path, output_dir=tmp_path / "indexes", settings=settings)
    bootstrap_dataset_from_chunks(chunks_path, tmp_path / "eval.jsonl", max_documents=2)

    result = run_experiment_suite(settings, manifest_path=indexing.manifest_path)

    assert len(result.benchmarks) == 3
    assert "Recall@3" in result.benchmark_table
    assert all(benchmark.ndcg_at_k['3'] <= 1.0 for benchmark in result.benchmarks)
    assert result.artifacts is not None
    assert Path(result.artifacts.aggregate_json).exists()
    assert Path(result.artifacts.summary_csv).exists()
    assert Path(result.artifacts.per_query_csv).exists()
    assert Path(result.artifacts.benchmark_table_md).exists()
    assert result.artifacts.benchmark_plot_svg is not None
