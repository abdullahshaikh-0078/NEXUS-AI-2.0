from __future__ import annotations

from collections import OrderedDict

from rag_service.query.models import ExpansionTerm

DOMAIN_SYNONYMS: dict[str, list[str]] = {
    "rag": ["retrieval augmented generation", "grounded generation", "knowledge grounded qa"],
    "llm": ["large language model", "foundation model", "generative model"],
    "bm25": ["lexical retrieval", "sparse retrieval", "term matching"],
    "faiss": ["vector index", "dense retrieval", "similarity search"],
    "api": ["service endpoint", "interface", "backend api"],
    "pdf": ["document", "portable document format", "file"],
    "html": ["web page", "markup document", "content page"],
    "auth": ["authentication", "access control", "identity verification"],
    "latency": ["response time", "performance", "inference delay"],
    "cache": ["memoization", "response cache", "redis cache"],
    "dense": ["vector", "embedding based", "semantic"],
    "sparse": ["lexical", "keyword", "bm25"],
    "hybrid": ["combined retrieval", "dense and sparse", "fusion retrieval"],
    "rerank": ["reranking", "cross encoder", "second stage ranking"],
    "eval": ["evaluation", "benchmarking", "metrics"],
}


def expand_query_terms(
    tokens: list[str],
    max_terms: int,
    terms_per_token: int,
) -> tuple[list[ExpansionTerm], list[str]]:
    expansions: list[ExpansionTerm] = []
    expanded_terms: OrderedDict[str, None] = OrderedDict()

    for token in tokens:
        related_terms = DOMAIN_SYNONYMS.get(token, [])[:terms_per_token]
        if not related_terms:
            continue
        expansions.append(ExpansionTerm(source_token=token, related_terms=related_terms))
        for term in related_terms:
            if term not in expanded_terms and len(expanded_terms) < max_terms:
                expanded_terms[term] = None

    return expansions, list(expanded_terms.keys())
