from __future__ import annotations

from rag_service.core.config import Settings
from rag_service.query.pipeline import process_query


class StubLLMRewriter:
    def __init__(self, rewritten_query: str) -> None:
        self.rewritten_query = rewritten_query

    def rewrite(
        self,
        *,
        original_query: str,
        cleaned_query: str,
        normalized_query: str,
        expanded_terms: list[str],
    ) -> str:
        return self.rewritten_query


class FailingLLMRewriter:
    def rewrite(
        self,
        *,
        original_query: str,
        cleaned_query: str,
        normalized_query: str,
        expanded_terms: list[str],
    ) -> str:
        raise RuntimeError("llm unavailable")


def test_process_query_cleans_normalizes_and_expands_terms() -> None:
    settings = Settings(
        query={
            "enable_expansion": True,
            "enable_llm_fallback": False,
            "expansion_terms_per_token": 2,
            "max_expanded_terms": 6,
        }
    )

    result = process_query("  RAG latency + BM25  ", settings=settings)

    assert result.cleaned_query == "RAG latency + BM25"
    assert result.normalized_query == "rag latency bm25"
    assert result.tokens == ["rag", "latency", "bm25"]
    assert "retrieval augmented generation" in result.expanded_terms
    assert "response time" in result.expanded_terms
    assert result.rewrite_strategy == "rule_based"
    assert result.rewritten_query.endswith("?")


def test_process_query_uses_llm_fallback_for_ambiguous_short_queries() -> None:
    settings = Settings(
        query={
            "enable_expansion": True,
            "enable_llm_fallback": True,
            "llm_fallback_min_tokens": 4,
        }
    )
    llm_rewriter = StubLLMRewriter(
        "How should this issue be resolved in the retrieval pipeline?"
    )

    result = process_query("this issue", settings=settings, llm_rewriter=llm_rewriter)

    assert result.rewrite_strategy == "llm_fallback"
    assert result.llm_fallback_used is True
    assert result.rewritten_query == "How should this issue be resolved in the retrieval pipeline?"
    assert result.llm_fallback_error is None


def test_process_query_keeps_rule_based_output_when_llm_fallback_fails() -> None:
    settings = Settings(
        query={
            "enable_expansion": True,
            "enable_llm_fallback": True,
            "llm_fallback_min_tokens": 4,
        }
    )

    result = process_query(
        "bm25 faiss latency",
        settings=settings,
        llm_rewriter=FailingLLMRewriter(),
    )

    assert result.rewrite_strategy == "rule_based"
    assert result.llm_fallback_used is False
    assert result.llm_fallback_error == "llm unavailable"
    assert "What information is available about" in result.rewritten_query


def test_process_query_rejects_empty_content() -> None:
    settings = Settings(query={"enable_llm_fallback": False})

    try:
        process_query("   \n\t  ", settings=settings)
    except ValueError as exc:
        assert "must not be empty" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Expected ValueError for empty query")
