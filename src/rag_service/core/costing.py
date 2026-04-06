from __future__ import annotations

from dataclasses import dataclass

from rag_service.context.models import ContextPackage
from rag_service.core.config import Settings


@dataclass(frozen=True)
class GenerationDecision:
    provider: str
    reason: str


def choose_generation_provider(query: str, context: ContextPackage, settings: Settings) -> GenerationDecision:
    configured_provider = settings.generation.provider.lower()
    if configured_provider != "openai":
        return GenerationDecision(provider=configured_provider, reason="configured-provider")

    if not settings.cost.skip_llm_for_high_confidence:
        return GenerationDecision(provider=configured_provider, reason="llm-enabled")

    if len(context.selected_blocks) < settings.cost.minimum_context_blocks_for_skip:
        return GenerationDecision(provider=configured_provider, reason="insufficient-context")

    average_score = sum(block.rerank_score for block in context.selected_blocks) / max(1, len(context.selected_blocks))
    if average_score >= settings.cost.confidence_rerank_threshold and len(query.split()) <= 12:
        return GenerationDecision(provider="heuristic", reason="high-confidence-context")

    return GenerationDecision(provider=configured_provider, reason="llm-required")


def optimize_context_for_cost(context: ContextPackage, settings: Settings) -> ContextPackage:
    max_blocks = max(1, settings.cost.max_generation_context_blocks)
    if len(context.selected_blocks) <= max_blocks:
        return context

    trimmed_blocks = context.selected_blocks[:max_blocks]
    return context.model_copy(
        update={
            "selected_blocks": trimmed_blocks,
            "total_tokens": sum(block.token_count for block in trimmed_blocks),
            "omitted_chunk_ids": context.omitted_chunk_ids + [block.chunk_id for block in context.selected_blocks[max_blocks:]],
            "context_text": "\n\n".join(block.compressed_text for block in trimmed_blocks),
        }
    )
