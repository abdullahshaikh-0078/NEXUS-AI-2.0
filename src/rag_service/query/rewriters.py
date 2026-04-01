from __future__ import annotations

import re
from typing import Protocol

from rag_service.core.config import Settings

ACRONYM_MAP = {
    "rag": "retrieval augmented generation",
    "llm": "large language model",
    "bm25": "bm25 lexical retrieval",
    "faiss": "faiss vector index",
    "api": "api service",
}
AMBIGUOUS_TOKENS = {"it", "this", "that", "they", "them", "he", "she", "issue", "problem"}
QUESTION_WORDS = {"what", "how", "why", "when", "where", "who", "which", "explain", "compare"}
KEYWORD_DELIMITER_PATTERN = re.compile(r"[,/+|]")
TOKEN_PATTERN = re.compile(r"[a-z0-9_]+")


class QueryLLMRewriter(Protocol):
    def rewrite(
        self,
        *,
        original_query: str,
        cleaned_query: str,
        normalized_query: str,
        expanded_terms: list[str],
    ) -> str:
        ...


class OpenAIQueryRewriter:
    def __init__(self, settings: Settings) -> None:
        if not settings.openai.api_key:
            raise ValueError("OpenAI API key is required for query rewrite fallback")

        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise ImportError("openai package is required for query rewrite fallback") from exc

        self._client = OpenAI(
            api_key=settings.openai.api_key,
            timeout=settings.openai.timeout_seconds,
        )
        self._model = settings.openai.model

    def rewrite(
        self,
        *,
        original_query: str,
        cleaned_query: str,
        normalized_query: str,
        expanded_terms: list[str],
    ) -> str:
        system_prompt = (
            "Rewrite user search queries for enterprise RAG retrieval. "
            "Return one grounded search query sentence only with no preamble."
        )
        user_prompt = (
            f"Original query: {original_query}\n"
            f"Cleaned query: {cleaned_query}\n"
            f"Normalized query: {normalized_query}\n"
            f"Expansion hints: {', '.join(expanded_terms) if expanded_terms else 'none'}"
        )
        response = self._client.chat.completions.create(
            model=self._model,
            temperature=0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        content = response.choices[0].message.content or ""
        return content.strip()


class RewriteDecision:
    def __init__(self, rewritten_query: str, strategy: str, reason: str) -> None:
        self.rewritten_query = rewritten_query
        self.strategy = strategy
        self.reason = reason


def rule_based_rewrite(
    cleaned_query: str,
    normalized_query: str,
    tokens: list[str],
    expanded_terms: list[str],
) -> RewriteDecision:
    expanded_text = _expand_acronyms(cleaned_query)

    if _looks_like_keyword_query(cleaned_query, normalized_query, tokens):
        focus_terms = _focus_terms(tokens, expanded_terms)
        rewritten = f"What information is available about {focus_terms}?"
        return RewriteDecision(
            rewritten_query=rewritten,
            strategy="rule_based",
            reason="Converted terse keyword input into a retrieval-friendly question.",
        )

    if tokens and tokens[0] in QUESTION_WORDS:
        rewritten = expanded_text.rstrip("?.!") + "?"
        return RewriteDecision(
            rewritten_query=_capitalize_first(rewritten),
            strategy="rule_based",
            reason="Normalized question punctuation and expanded domain acronyms.",
        )

    return RewriteDecision(
        rewritten_query=_capitalize_first(expanded_text),
        strategy="rule_based",
        reason="Expanded domain acronyms while preserving the original intent.",
    )


def should_use_llm_fallback(
    cleaned_query: str,
    normalized_query: str,
    tokens: list[str],
    settings: Settings,
) -> bool:
    if not settings.query.enable_llm_fallback:
        return False
    if len(tokens) <= settings.query.llm_fallback_min_tokens:
        return True
    if any(token in AMBIGUOUS_TOKENS for token in tokens):
        return True
    if _looks_like_keyword_query(cleaned_query, normalized_query, tokens):
        return True
    return False


def create_llm_rewriter(settings: Settings) -> QueryLLMRewriter:
    return OpenAIQueryRewriter(settings)


def _expand_acronyms(text: str) -> str:
    result = text
    for acronym, expansion in ACRONYM_MAP.items():
        result = re.sub(
            rf"\b{re.escape(acronym)}\b",
            f"{expansion} ({acronym.upper()})",
            result,
            flags=re.IGNORECASE,
        )
    return result


def _looks_like_keyword_query(cleaned_query: str, normalized_query: str, tokens: list[str]) -> bool:
    if KEYWORD_DELIMITER_PATTERN.search(cleaned_query):
        return True
    if len(tokens) <= 5 and tokens and tokens[0] not in QUESTION_WORDS and "?" not in cleaned_query:
        return True
    if len(tokens) > 1 and len(TOKEN_PATTERN.findall(normalized_query)) == len(tokens) and " " in normalized_query:
        return all(token not in QUESTION_WORDS for token in tokens[:2]) and len(tokens) <= 6
    return False


def _focus_terms(tokens: list[str], expanded_terms: list[str]) -> str:
    seen: dict[str, None] = {}
    for token in tokens:
        display = ACRONYM_MAP.get(token, token)
        seen.setdefault(display, None)
    for term in expanded_terms[:3]:
        seen.setdefault(term, None)
    return ", ".join(seen.keys())


def _capitalize_first(text: str) -> str:
    return text[:1].upper() + text[1:] if text else text
