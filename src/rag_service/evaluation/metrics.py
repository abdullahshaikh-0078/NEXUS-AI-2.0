from __future__ import annotations

import math
import re
from statistics import mean

_WORD_RE = re.compile(r"[a-zA-Z0-9]+")
_STOPWORDS = {
    "a", "an", "the", "and", "or", "to", "of", "in", "on", "for", "with", "is", "are", "was", "were", "be",
    "this", "that", "it", "as", "by", "from", "at", "about", "into", "your", "their", "its", "what",
}


def recall_at_k(relevant_ids: set[str], ranked_ids: list[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    hits = len(relevant_ids.intersection(ranked_ids[:k]))
    return round(hits / len(relevant_ids), 4)


def reciprocal_rank(relevant_ids: set[str], ranked_ids: list[str]) -> float:
    for index, candidate in enumerate(ranked_ids, start=1):
        if candidate in relevant_ids:
            return round(1.0 / index, 4)
    return 0.0


def ndcg_at_k(relevant_ids: set[str], ranked_ids: list[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    dcg = 0.0
    for index, candidate in enumerate(ranked_ids[:k], start=1):
        if candidate in relevant_ids:
            dcg += 1.0 / math.log2(index + 1)
    ideal_hits = min(len(relevant_ids), k)
    if ideal_hits == 0:
        return 0.0
    idcg = sum(1.0 / math.log2(index + 1) for index in range(1, ideal_hits + 1))
    return round(dcg / idcg, 4) if idcg else 0.0


def faithfulness_score(answer: str, evidence_texts: list[str], *, overlap_threshold: float = 0.35) -> float | None:
    sentences = _split_sentences(answer)
    if not sentences:
        return None

    evidence_tokens = [_tokenize(text) for text in evidence_texts if text.strip()]
    if not evidence_tokens:
        return 0.0

    supported = 0
    for sentence in sentences:
        sentence_tokens = _tokenize(sentence)
        if not sentence_tokens:
            continue
        best_overlap = max((_overlap_ratio(sentence_tokens, evidence) for evidence in evidence_tokens), default=0.0)
        if best_overlap >= overlap_threshold:
            supported += 1
    return round(supported / len(sentences), 4)


def hallucination_rate(answer: str, evidence_texts: list[str], *, overlap_threshold: float = 0.35) -> float | None:
    score = faithfulness_score(answer, evidence_texts, overlap_threshold=overlap_threshold)
    if score is None:
        return None
    return round(1.0 - score, 4)


def average(values: list[float | None]) -> float | None:
    filtered = [value for value in values if value is not None]
    if not filtered:
        return None
    return round(mean(filtered), 4)


def _split_sentences(text: str) -> list[str]:
    normalized = " ".join(text.split())
    if not normalized:
        return []
    parts = re.split(r"(?<=[.!?])\s+", normalized)
    return [part.strip() for part in parts if part.strip()]


def _tokenize(text: str) -> set[str]:
    return {
        token.lower()
        for token in _WORD_RE.findall(text)
        if len(token) > 2 and token.lower() not in _STOPWORDS
    }


def _overlap_ratio(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    overlap = len(a.intersection(b))
    return overlap / max(1, len(a))
