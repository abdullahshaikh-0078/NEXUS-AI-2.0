from __future__ import annotations

from rag_service.context.models import ContextBlock, ContextPackage
from rag_service.ingestion.models import ChunkMetadata
from rag_service.query.models import ProcessedQuery
from rag_service.reranking.models import RerankedChunk, RerankingResult
from rag_service.retrieval.models import HybridRetrievalResult


def fake_context_package() -> ContextPackage:
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
