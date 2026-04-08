from __future__ import annotations

from rag_service.context.models import ContextBlock, ContextPackage
from rag_service.core.config import Settings
from rag_service.core.costing import choose_generation_provider, optimize_context_for_cost
from rag_service.ingestion.models import ChunkMetadata
from rag_service.query.models import ProcessedQuery
from rag_service.reranking.models import RerankingResult
from rag_service.retrieval.models import HybridRetrievalResult


def test_cost_policy_skips_llm_for_high_confidence_context() -> None:
    settings = Settings(
        generation={"provider": "openai", "fallback_provider": "heuristic"},
        cost={
            "skip_llm_for_high_confidence": True,
            "confidence_rerank_threshold": 0.8,
            "minimum_context_blocks_for_skip": 2,
            "max_generation_context_blocks": 2,
        },
    )
    context = _build_context_package(block_scores=[0.92, 0.89, 0.74])

    decision = choose_generation_provider("what is rag", context, settings)
    optimized = optimize_context_for_cost(context, settings)

    assert decision.provider == "heuristic"
    assert decision.reason == "high-confidence-context"
    assert len(optimized.selected_blocks) == 2


def _build_context_package(block_scores: list[float]) -> ContextPackage:
    processed_query = ProcessedQuery(
        original_query="what is rag",
        cleaned_query="what is rag",
        normalized_query="what is rag",
        tokens=["what", "is", "rag"],
        rewritten_query="what is rag",
    )
    retrieval = HybridRetrievalResult(processed_query=processed_query)
    reranking = RerankingResult(retrieval=retrieval, provider="heuristic", model_name="heuristic", candidate_count=len(block_scores))
    blocks = [
        ContextBlock(
            chunk_id=f"chunk-{index}",
            title=f"Doc {index}",
            source_path=f"data/raw/doc-{index}.txt",
            section_title="Overview",
            rerank_score=score,
            token_count=50,
            compressed_text=f"Compressed block {index}",
            original_text=f"Original block {index}",
        )
        for index, score in enumerate(block_scores, start=1)
    ]
    return ContextPackage(
        reranking=reranking,
        selected_blocks=blocks,
        total_tokens=len(blocks) * 50,
        context_text="\n".join(block.compressed_text for block in blocks),
    )
