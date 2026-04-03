from __future__ import annotations

from pathlib import Path

from rag_service.context.models import ContextBlock, ContextPackage
from rag_service.core.config import Settings
from rag_service.generation.pipeline import generate_grounded_answer
from rag_service.indexing.pipeline import build_indexes
from rag_service.ingestion.models import ChunkMetadata
from rag_service.ingestion.pipeline import ingest_directory
from rag_service.query.models import ProcessedQuery
from rag_service.reranking.models import RerankedChunk, RerankingResult
from rag_service.retrieval.models import HybridRetrievalResult


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
        context=_fake_context_package(),
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


def _fake_context_package() -> ContextPackage:
    metadata = ChunkMetadata(
        document_id="doc-1",
        source_path="data/raw/doc1.txt",
        source_type="txt",
        title="Doc One",
        section_title="Overview",
        chunk_index=0,
        chunk_strategy="fixed",
        char_count=120,
        word_count=20,
    )

    retrieval = HybridRetrievalResult(
        processed_query=ProcessedQuery(
            original_query="retrieval grounding",
            cleaned_query="retrieval grounding",
            normalized_query="retrieval grounding",
            tokens=["retrieval", "grounding"],
            rewritten_query="How does retrieval improve grounding?",
            rewrite_strategy="rule_based",
            rewrite_reason="Expanded terse query into a grounded question.",
        )
    )

    reranking = RerankingResult(
        retrieval=retrieval,
        provider="heuristic",
        model_name="heuristic-overlap-v1",
        candidate_count=2,
        reranked_hits=[
            RerankedChunk(
                row_id=0,
                chunk_id="doc-1:0001",
                rerank_score=0.95,
                text="Hybrid retrieval metadata improves grounding and traceability.",
                metadata=metadata,
                source="rrf",
                retrieval_score=0.9,
                retrieval_rank=1,
                dense_rank=1,
                sparse_rank=1,
            ),
            RerankedChunk(
                row_id=1,
                chunk_id="doc-2:0001",
                rerank_score=0.82,
                text="Dense and sparse retrieval together improve recall across private corpora.",
                metadata=metadata.model_copy(
                    update={
                        "document_id": "doc-2",
                        "source_path": "data/raw/doc2.txt",
                        "title": "Doc Two",
                        "chunk_index": 1,
                    }
                ),
                source="rrf",
                retrieval_score=0.8,
                retrieval_rank=2,
                dense_rank=2,
                sparse_rank=2,
            ),
        ],
    )

    return ContextPackage(
        reranking=reranking,
        selected_blocks=[
            ContextBlock(
                chunk_id="doc-1:0001",
                title="Doc One",
                source_path="data/raw/doc1.txt",
                section_title="Overview",
                rerank_score=0.95,
                token_count=12,
                compressed_text="Hybrid retrieval metadata improves grounding and traceability.",
                original_text="Hybrid retrieval metadata improves grounding and traceability.",
            ),
            ContextBlock(
                chunk_id="doc-2:0001",
                title="Doc Two",
                source_path="data/raw/doc2.txt",
                section_title="Overview",
                rerank_score=0.82,
                token_count=13,
                compressed_text="Dense and sparse retrieval together improve recall across private corpora.",
                original_text="Dense and sparse retrieval together improve recall across private corpora.",
            ),
        ],
        total_tokens=25,
        omitted_chunk_ids=[],
        context_text=(
            "[Context 1] title=Doc One; section=Overview; source=data/raw/doc1.txt\n"
            "Hybrid retrieval metadata improves grounding and traceability.\n\n"
            "[Context 2] title=Doc Two; section=Overview; source=data/raw/doc2.txt\n"
            "Dense and sparse retrieval together improve recall across private corpora."
        ),
    )
