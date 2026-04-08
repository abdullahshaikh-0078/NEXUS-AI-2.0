from __future__ import annotations

from pathlib import Path

from rag_service.generation.models import Citation
from rag_service.postprocessing.models import FormattedCitation


def format_citations(citations: list[Citation]) -> list[FormattedCitation]:
    return [
        FormattedCitation(
            id=citation.id,
            marker=citation.marker,
            source=Path(citation.source_path or citation.title).name or citation.title or citation.id,
            title=citation.title,
            section_title=citation.section_title,
            source_path=citation.source_path,
            score=round(citation.score, 4),
            text=citation.text,
            reference_text=_build_reference_text(citation),
        )
        for citation in citations
    ]


def render_references_markdown(citations: list[FormattedCitation]) -> str:
    if not citations:
        return ""

    return "\n".join(
        f"- {citation.marker} {citation.reference_text}"
        for citation in citations
    )


def _build_reference_text(citation: Citation) -> str:
    title = citation.title or "Untitled"
    section = citation.section_title or "General"
    source = Path(citation.source_path or title).name
    return f"{title} | section: {section} | source: {source} | score: {citation.score:.2f}"
