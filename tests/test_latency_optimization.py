from __future__ import annotations

from rag_service.core.config import Settings
from rag_service.core.optimization import build_retrieval_plan, iter_answer_chunks
from rag_service.query.models import ProcessedQuery


def test_build_retrieval_plan_reduces_candidate_pool_for_short_queries() -> None:
    settings = Settings(
        retrieval={"dense_top_k": 8, "sparse_top_k": 8, "candidate_pool_size": 12, "fused_top_k": 10},
        reranking={"candidate_limit": 10},
        latency={
            "enable_adaptive_retrieval": True,
            "short_query_token_threshold": 4,
            "short_query_dense_top_k": 3,
            "short_query_sparse_top_k": 4,
            "short_query_candidate_pool_size": 5,
            "short_query_fused_top_k": 4,
        },
    )
    processed = ProcessedQuery(
        original_query="rag latency",
        cleaned_query="rag latency",
        normalized_query="rag latency",
        tokens=["rag", "latency"],
        rewritten_query="rag latency",
    )

    plan = build_retrieval_plan(processed, settings)

    assert plan.dense_top_k == 3
    assert plan.sparse_top_k == 4
    assert plan.candidate_pool_size == 5
    assert plan.fused_top_k == 4


def test_iter_answer_chunks_preserves_full_answer() -> None:
    answer = "Grounded answers should stream in stable chunks for frontend rendering."

    chunks = list(iter_answer_chunks(answer, 18))

    assert "".join(chunks).strip() == answer
    assert len(chunks) > 1
