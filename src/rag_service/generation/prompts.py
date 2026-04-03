from __future__ import annotations

from rag_service.context.models import ContextPackage
from rag_service.core.config import Settings
from rag_service.generation.models import Citation, PromptBundle


def build_prompt_bundle(
    query: str,
    context: ContextPackage,
    citations: list[Citation],
    settings: Settings,
) -> PromptBundle:
    evidence_lines = _render_evidence_blocks(citations)
    system_prompt = (
        "You are a grounded RAG assistant. "
        "Answer using only the supplied evidence. "
        "Do not invent facts, and include inline citations like [1] that map to the evidence list. "
        "If the evidence is insufficient, say so clearly."
    )
    user_prompt = (
        f"Question:\n{query}\n\n"
        f"Evidence constraints:\n"
        f"- Use at most {settings.generation.max_citations} citations\n"
        f"- Prefer concise, direct answers\n"
        f"- Cite every factual claim\n\n"
        f"Evidence:\n{evidence_lines}\n\n"
        "Write a grounded answer in markdown."
    )
    return PromptBundle(system_prompt=system_prompt, user_prompt=user_prompt)


def _render_evidence_blocks(citations: list[Citation]) -> str:
    if not citations:
        return "[No evidence retrieved]"

    rendered: list[str] = []
    for citation in citations:
        header = (
            f"{citation.marker} title={citation.title or 'untitled'}; "
            f"section={citation.section_title or 'none'}; source={citation.source_path}"
        )
        rendered.append(f"{header}\n{citation.text}")
    return "\n\n".join(rendered)
