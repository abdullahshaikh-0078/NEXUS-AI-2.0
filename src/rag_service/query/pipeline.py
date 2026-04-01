from __future__ import annotations

from rag_service.core.config import Settings
from rag_service.core.logging import get_logger
from rag_service.query.cleaners import clean_query, normalize_query, tokenize_query
from rag_service.query.expanders import expand_query_terms
from rag_service.query.models import ProcessedQuery
from rag_service.query.rewriters import QueryLLMRewriter, create_llm_rewriter, rule_based_rewrite, should_use_llm_fallback

logger = get_logger(__name__)


def process_query(
    query: str,
    settings: Settings,
    llm_rewriter: QueryLLMRewriter | None = None,
) -> ProcessedQuery:
    cleaned_query = clean_query(query)
    normalized_query = normalize_query(
        cleaned_query,
        preserve_original_case=settings.query.preserve_original_case,
    )
    tokens = tokenize_query(normalized_query)

    expansions = []
    expanded_terms: list[str] = []
    if settings.query.enable_expansion:
        expansions, expanded_terms = expand_query_terms(
            tokens=tokens,
            max_terms=settings.query.max_expanded_terms,
            terms_per_token=settings.query.expansion_terms_per_token,
        )

    rewritten_query = cleaned_query
    rewrite_strategy = "none"
    rewrite_reason = "Preserved cleaned query without rewrite."
    llm_fallback_used = False
    llm_fallback_error: str | None = None

    if settings.query.enable_rule_rewrite:
        decision = rule_based_rewrite(
            cleaned_query=cleaned_query,
            normalized_query=normalized_query,
            tokens=tokens,
            expanded_terms=expanded_terms,
        )
        rewritten_query = decision.rewritten_query
        rewrite_strategy = decision.strategy
        rewrite_reason = decision.reason

    if should_use_llm_fallback(cleaned_query, normalized_query, tokens, settings):
        has_llm_path = llm_rewriter is not None or bool(settings.openai.api_key)
        if has_llm_path:
            try:
                rewriter = llm_rewriter or create_llm_rewriter(settings)
                llm_rewrite = rewriter.rewrite(
                    original_query=query,
                    cleaned_query=cleaned_query,
                    normalized_query=normalized_query,
                    expanded_terms=expanded_terms,
                )
                if llm_rewrite.strip():
                    rewritten_query = llm_rewrite.strip()
                    rewrite_strategy = "llm_fallback"
                    rewrite_reason = "Applied LLM rewrite fallback for ambiguous or terse input."
                    llm_fallback_used = True
            except Exception as exc:  # pragma: no cover - fallback behavior tested via stub
                llm_fallback_error = str(exc)
                logger.warning(
                    "query_llm_fallback_failed",
                    query=cleaned_query,
                    reason=llm_fallback_error,
                )

    return ProcessedQuery(
        original_query=query,
        cleaned_query=cleaned_query,
        normalized_query=normalized_query,
        tokens=tokens,
        expansions=expansions,
        expanded_terms=expanded_terms,
        rewritten_query=rewritten_query,
        rewrite_strategy=rewrite_strategy,
        rewrite_reason=rewrite_reason,
        llm_fallback_used=llm_fallback_used,
        llm_fallback_error=llm_fallback_error,
    )
