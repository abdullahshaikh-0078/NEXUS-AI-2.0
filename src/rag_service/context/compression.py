from __future__ import annotations

import re

TOKEN_PATTERN = re.compile(r"\S+")
SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+")
WORD_PATTERN = re.compile(r"[A-Za-z0-9_]+")


def estimate_tokens(text: str) -> int:
    return len(TOKEN_PATTERN.findall(text))


def compress_text(
    query: str,
    text: str,
    token_limit: int,
    strategy: str = "extractive",
) -> str:
    if token_limit <= 0:
        return ""
    if strategy != "extractive":
        return _truncate_to_token_limit(text, token_limit)

    sentences = [sentence.strip() for sentence in SENTENCE_SPLIT_PATTERN.split(text) if sentence.strip()]
    if not sentences:
        return _truncate_to_token_limit(text, token_limit)

    query_terms = set(WORD_PATTERN.findall(query.lower()))
    ranked_sentences = sorted(
        sentences,
        key=lambda sentence: (_sentence_overlap(query_terms, sentence), len(sentence)),
        reverse=True,
    )

    selected: list[str] = []
    total_tokens = 0
    for sentence in ranked_sentences:
        sentence_tokens = estimate_tokens(sentence)
        if sentence_tokens == 0:
            continue
        if total_tokens + sentence_tokens > token_limit:
            continue
        selected.append(sentence)
        total_tokens += sentence_tokens
        if total_tokens >= token_limit:
            break

    if not selected:
        return _truncate_to_token_limit(text, token_limit)
    return " ".join(selected)


def _sentence_overlap(query_terms: set[str], sentence: str) -> int:
    sentence_terms = set(WORD_PATTERN.findall(sentence.lower()))
    return len(query_terms.intersection(sentence_terms))


def _truncate_to_token_limit(text: str, token_limit: int) -> str:
    tokens = TOKEN_PATTERN.findall(text)
    if len(tokens) <= token_limit:
        return text.strip()
    return " ".join(tokens[:token_limit]).strip()
