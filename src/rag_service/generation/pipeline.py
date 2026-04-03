from __future__ import annotations

from rag_service.context.models import ContextPackage
from rag_service.context.pipeline import build_context
from rag_service.core.config import Settings
from rag_service.core.logging import get_logger
from rag_service.generation.generators import create_answer_generator
from rag_service.generation.models import Citation, GroundedAnswer
from rag_service.generation.prompts import build_prompt_bundle

logger = get_logger(__name__)


def generate_grounded_answer(
    query: str,
    settings: Settings,
    context: ContextPackage | None = None,
) -> GroundedAnswer:
    context_package = context or build_context(query, settings=settings)
    citations = _build_citations(context_package, max_citations=settings.generation.max_citations)
    prompt = build_prompt_bundle(query=query, context=context_package, citations=citations, settings=settings)

    generator = create_answer_generator(settings)
    used_fallback = False

    try:
        answer = generator.generate(query=query, context=context_package, prompt=prompt, citations=citations)
    except Exception as exc:
        fallback_provider = settings.generation.fallback_provider.lower()
        logger.warning(
            "generation_provider_failed",
            provider=generator.provider_name,
            fallback_provider=fallback_provider,
            error=str(exc),
        )
        fallback = create_answer_generator(settings, provider=fallback_provider)
        answer = fallback.generate(query=query, context=context_package, prompt=prompt, citations=citations)
        generator = fallback
        used_fallback = True

    logger.info(
        "grounded_answer_generated",
        provider=generator.provider_name,
        model_name=generator.model_name,
        citation_count=len(citations),
        used_fallback=used_fallback,
    )

    return GroundedAnswer(
        question=query,
        answer=answer,
        provider=generator.provider_name,
        model_name=generator.model_name,
        used_fallback=used_fallback,
        citations=citations,
        prompt=prompt,
        context=context_package,
    )


def _build_citations(context: ContextPackage, max_citations: int) -> list[Citation]:
    citations: list[Citation] = []
    for index, block in enumerate(context.selected_blocks[:max_citations], start=1):
        citations.append(
            Citation(
                id=block.chunk_id,
                chunk_id=block.chunk_id,
                marker=f"[{index}]",
                title=block.title,
                source_path=block.source_path,
                section_title=block.section_title,
                score=block.rerank_score,
                text=block.compressed_text,
            )
        )
    return citations
