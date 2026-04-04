from __future__ import annotations

from rag_service.core.config import Settings
from rag_service.core.logging import get_logger
from rag_service.generation.models import GroundedAnswer
from rag_service.generation.pipeline import generate_grounded_answer
from rag_service.postprocessing.formatters import format_citations, render_references_markdown
from rag_service.postprocessing.models import AnswerMetadata, PostProcessedAnswer
from rag_service.postprocessing.scoring import assess_confidence

logger = get_logger(__name__)


def postprocess_grounded_answer(
    query: str,
    settings: Settings,
    grounded_answer: GroundedAnswer | None = None,
) -> PostProcessedAnswer:
    answer = grounded_answer or generate_grounded_answer(query, settings=settings)
    formatted_citations = format_citations(answer.citations)
    confidence = assess_confidence(answer, settings=settings)
    references_markdown = render_references_markdown(formatted_citations)

    logger.info(
        "postprocessing_completed",
        confidence_score=confidence.score,
        confidence_label=confidence.label,
        citation_count=len(formatted_citations),
    )

    return PostProcessedAnswer(
        question=answer.question,
        answer=answer.answer,
        answer_markdown=answer.answer,
        confidence=confidence,
        citations=formatted_citations,
        references_markdown=references_markdown,
        metadata=AnswerMetadata(
            provider=answer.provider,
            model_name=answer.model_name,
            used_fallback=answer.used_fallback,
            context_block_count=len(answer.context.selected_blocks),
            total_context_tokens=answer.context.total_tokens,
        ),
        grounded_answer=answer,
    )
