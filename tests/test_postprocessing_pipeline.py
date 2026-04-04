from __future__ import annotations

from pathlib import Path

from rag_service.core.config import Settings
from rag_service.generation.pipeline import generate_grounded_answer
from rag_service.indexing.pipeline import build_indexes
from rag_service.ingestion.pipeline import ingest_directory
from rag_service.postprocessing.pipeline import postprocess_grounded_answer
from tests.test_generation_pipeline import _fake_context_package


def test_postprocess_grounded_answer_formats_citations_and_confidence() -> None:
    settings = Settings(
        generation={
            "provider": "heuristic",
            "fallback_provider": "heuristic",
            "max_citations": 2,
        },
        postprocessing={
            "high_confidence_threshold": 0.8,
            "medium_confidence_threshold": 0.55,
            "target_citation_count": 2,
            "target_context_blocks": 2,
            "citation_score_weight": 0.55,
            "citation_coverage_weight": 0.25,
            "context_coverage_weight": 0.20,
            "fallback_penalty": 0.12,
        },
    )

    grounded = generate_grounded_answer(
        "what improves retrieval grounding",
        settings=settings,
        context=_fake_context_package(),
    )
    result = postprocess_grounded_answer(
        "what improves retrieval grounding",
        settings=settings,
        grounded_answer=grounded,
    )

    assert result.confidence.score > 0.7
    assert result.confidence.label in {"high", "medium"}
    assert result.citations[0].reference_text.startswith("Doc One")
    assert result.references_markdown.startswith("- [1]")
    assert result.metadata.context_block_count == 2


def test_postprocess_grounded_answer_end_to_end_with_sample_documents(
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
            "embedding_version": "phase9-test",
            "dense_backend": "native",
            "dense_fallback_backend": "native",
            "sparse_backend": "native",
            "sparse_fallback_backend": "native",
        },
        query={"enable_llm_fallback": False},
        retrieval={
            "manifest_path": str(tmp_path / "indexes" / "phase9-test" / "manifest.json"),
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
        postprocessing={
            "high_confidence_threshold": 0.8,
            "medium_confidence_threshold": 0.55,
            "target_citation_count": 3,
            "target_context_blocks": 3,
            "citation_score_weight": 0.55,
            "citation_coverage_weight": 0.25,
            "context_coverage_weight": 0.20,
            "fallback_penalty": 0.12,
        },
    )

    chunks_path = tmp_path / "chunks.jsonl"
    ingest_directory(sample_document_dir, chunks_path, settings)
    build_indexes(chunks_file=chunks_path, output_dir=tmp_path / "indexes", settings=settings)

    result = postprocess_grounded_answer("hybrid retrieval metadata", settings=settings)

    assert result.answer
    assert result.citations
    assert 0.0 <= result.confidence.score <= 1.0
    assert result.references_markdown
    assert result.metadata.total_context_tokens > 0
