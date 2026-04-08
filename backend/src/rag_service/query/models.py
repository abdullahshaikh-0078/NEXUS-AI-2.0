from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

RewriteStrategy = Literal["none", "rule_based", "llm_fallback"]


class ExpansionTerm(BaseModel):
    source_token: str
    related_terms: list[str] = Field(default_factory=list)


class ProcessedQuery(BaseModel):
    original_query: str
    cleaned_query: str
    normalized_query: str
    tokens: list[str] = Field(default_factory=list)
    expansions: list[ExpansionTerm] = Field(default_factory=list)
    expanded_terms: list[str] = Field(default_factory=list)
    rewritten_query: str
    rewrite_strategy: RewriteStrategy = "none"
    rewrite_reason: str = ""
    llm_fallback_used: bool = False
    llm_fallback_error: str | None = None
