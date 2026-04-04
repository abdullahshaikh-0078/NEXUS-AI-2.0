from __future__ import annotations

from statistics import mean

from rag_service.core.config import Settings
from rag_service.generation.models import GroundedAnswer
from rag_service.postprocessing.models import ConfidenceAssessment


def assess_confidence(answer: GroundedAnswer, settings: Settings) -> ConfidenceAssessment:
    citations = answer.citations
    average_citation_score = mean([citation.score for citation in citations]) if citations else 0.0
    citation_coverage = min(len(citations) / max(settings.postprocessing.target_citation_count, 1), 1.0)
    context_coverage = min(
        len(answer.context.selected_blocks) / max(settings.postprocessing.target_context_blocks, 1),
        1.0,
    )

    score = (
        average_citation_score * settings.postprocessing.citation_score_weight
        + citation_coverage * settings.postprocessing.citation_coverage_weight
        + context_coverage * settings.postprocessing.context_coverage_weight
    )

    if answer.used_fallback:
        score -= settings.postprocessing.fallback_penalty
    if not answer.answer.strip():
        score -= 0.2

    score = max(0.0, min(1.0, round(score, 4)))
    label = _label_for_score(score, settings)
    rationale = _build_rationale(
        average_citation_score=average_citation_score,
        citation_count=len(citations),
        context_block_count=len(answer.context.selected_blocks),
        used_fallback=answer.used_fallback,
    )
    return ConfidenceAssessment(score=score, label=label, rationale=rationale)


def _label_for_score(score: float, settings: Settings) -> str:
    if score >= settings.postprocessing.high_confidence_threshold:
        return "high"
    if score >= settings.postprocessing.medium_confidence_threshold:
        return "medium"
    return "low"


def _build_rationale(
    average_citation_score: float,
    citation_count: int,
    context_block_count: int,
    used_fallback: bool,
) -> list[str]:
    rationale = [
        f"Average citation score: {average_citation_score:.2f}",
        f"Citations used: {citation_count}",
        f"Context blocks selected: {context_block_count}",
    ]
    rationale.append("Generation fallback path used" if used_fallback else "Primary generation path used")
    return rationale
