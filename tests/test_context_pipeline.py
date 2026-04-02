from __future__ import annotations

from pathlib import Path

from rag_service.context.pipeline import build_context
from rag_service.core.config import Settings
from rag_service.indexing.pipeline import build_indexes
from rag_service.ingestion.pipeline import ingest_directory
from rag_service.reranking.pipeline import rerank_candidates
from rag_service.retrieval.pipeline import hybrid_retrieve


def test_build_context_selects_compressed_blocks_within_budget(
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
            "embedding_version": "phase7-test",
            "dense_backend": "native",
            "dense_fallback_backend": "native",
            "sparse_backend": "native",
            "sparse_fallback_backend": "native",
        },
        query={"enable_llm_fallback": False},
        retrieval={
            "manifest_path": str(tmp_path / "indexes" / "phase7-test" / "manifest.json"),
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
            "max_context_tokens": 70,
            "max_chunks": 2,
            "min_rerank_score": 0.0,
            "per_chunk_token_limit": 30,
            "deduplicate_by_document": True,
            "deduplicate_by_text": True,
            "compression_strategy": "extractive",
            "include_metadata_headers": True,
        },
    )

    chunks_path = tmp_path / "chunks.jsonl"
    ingest_directory(sample_document_dir, chunks_path, settings)
    build_indexes(chunks_file=chunks_path, output_dir=tmp_path / "indexes", settings=settings)
    retrieval = hybrid_retrieve("hybrid retrieval metadata", settings=settings)
    reranking = rerank_candidates("hybrid retrieval metadata", settings=settings, retrieval=retrieval)

    context = build_context("hybrid retrieval metadata", settings=settings, reranking=reranking)

    assert context.selected_blocks
    assert len(context.selected_blocks) <= 2
    assert context.total_tokens <= 70
    assert "[Context 1]" in context.context_text
    assert all(block.token_count <= 30 for block in context.selected_blocks)


def test_build_context_deduplicates_duplicate_documents() -> None:
    settings = Settings(
        context={
            "max_context_tokens": 100,
            "max_chunks": 5,
            "min_rerank_score": 0.0,
            "per_chunk_token_limit": 40,
            "deduplicate_by_document": True,
            "deduplicate_by_text": True,
            "compression_strategy": "extractive",
            "include_metadata_headers": False,
        }
    )

    retrieval = _fake_reranking_result()
    context = build_context("retrieval metadata", settings=settings, reranking=retrieval)

    assert len(context.selected_blocks) == 1
    assert context.omitted_chunk_ids == ["doc-1:0002"]


def _fake_reranking_result():
    from rag_service.ingestion.models import ChunkMetadata
    from rag_service.query.models import ProcessedQuery
    from rag_service.reranking.models import RerankedChunk, RerankingResult
    from rag_service.retrieval.models import HybridRetrievalResult

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
            original_query="retrieval metadata",
            cleaned_query="retrieval metadata",
            normalized_query="retrieval metadata",
            tokens=["retrieval", "metadata"],
            rewritten_query="What information is available about retrieval metadata?",
            rewrite_strategy="rule_based",
            rewrite_reason="Converted terse keyword input into a retrieval-friendly question.",
        )
    )

    return RerankingResult(
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
                chunk_id="doc-1:0002",
                rerank_score=0.82,
                text="Hybrid retrieval metadata supports stronger grounding in search systems.",
                metadata=metadata.model_copy(update={"chunk_index": 1}),
                source="rrf",
                retrieval_score=0.8,
                retrieval_rank=2,
                dense_rank=2,
                sparse_rank=2,
            ),
        ],
    )
