from __future__ import annotations

from pathlib import Path

from rag_service.core.config import Settings
from rag_service.generation.pipeline import generate_grounded_answer
from rag_service.indexing.pipeline import build_indexes
from rag_service.ingestion.pipeline import ingest_directory
from helpers import fake_context_package


def test_generate_grounded_answer_injects_citations_from_context() -> None:
    settings = Settings(
        generation={
            "provider": "heuristic",
            "fallback_provider": "heuristic",
            "max_citations": 2,
            "temperature": 0.0,
            "max_output_tokens": 300,
        }
    )

    result = generate_grounded_answer(
        "what improves retrieval grounding",
        settings=settings,
        context=fake_context_package(),
    )

    assert result.provider == "heuristic"
    assert len(result.citations) == 2
    assert "[1]" in result.answer
    assert "[2]" in result.answer
    assert "Question:" in result.prompt.user_prompt


def test_generate_grounded_answer_runs_end_to_end_with_sample_documents(
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
            "embedding_version": "phase8-test",
            "dense_backend": "native",
            "dense_fallback_backend": "native",
            "sparse_backend": "native",
            "sparse_fallback_backend": "native",
        },
        query={"enable_llm_fallback": False},
        retrieval={
            "manifest_path": str(tmp_path / "indexes" / "phase8-test" / "manifest.json"),
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
    )

    chunks_path = tmp_path / "chunks.jsonl"
    ingest_directory(sample_document_dir, chunks_path, settings)
    build_indexes(chunks_file=chunks_path, output_dir=tmp_path / "indexes", settings=settings)

    result = generate_grounded_answer("hybrid retrieval metadata", settings=settings)

    assert result.citations
    assert result.context.selected_blocks
    assert "grounded answer" in result.answer.lower()
    assert any(citation.marker in result.answer for citation in result.citations)
