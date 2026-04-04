from __future__ import annotations

from pydantic import BaseModel, Field

from rag_service.generation.models import GroundedAnswer


class FormattedCitation(BaseModel):
    id: str
    marker: str
    source: str
    title: str
    section_title: str
    source_path: str
    score: float
    text: str
    reference_text: str


class ConfidenceAssessment(BaseModel):
    score: float
    label: str
    rationale: list[str] = Field(default_factory=list)


class AnswerMetadata(BaseModel):
    provider: str
    model_name: str
    used_fallback: bool
    context_block_count: int
    total_context_tokens: int


class PostProcessedAnswer(BaseModel):
    question: str
    answer: str
    answer_markdown: str
    confidence: ConfidenceAssessment
    citations: list[FormattedCitation] = Field(default_factory=list)
    references_markdown: str = ""
    metadata: AnswerMetadata
    grounded_answer: GroundedAnswer
