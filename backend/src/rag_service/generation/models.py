from __future__ import annotations

from pydantic import BaseModel, Field

from rag_service.context.models import ContextPackage


class Citation(BaseModel):
    id: str
    chunk_id: str
    marker: str
    title: str
    source_path: str
    section_title: str
    score: float
    text: str


class PromptBundle(BaseModel):
    system_prompt: str
    user_prompt: str


class GroundedAnswer(BaseModel):
    question: str
    answer: str
    provider: str
    model_name: str
    used_fallback: bool = False
    citations: list[Citation] = Field(default_factory=list)
    prompt: PromptBundle
    context: ContextPackage
