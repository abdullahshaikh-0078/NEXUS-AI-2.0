from __future__ import annotations

import re
from abc import ABC, abstractmethod

from rag_service.context.models import ContextPackage
from rag_service.core.config import Settings
from rag_service.core.logging import get_logger
from rag_service.generation.models import Citation, PromptBundle

logger = get_logger(__name__)


class AnswerGenerator(ABC):
    provider_name: str
    model_name: str

    @abstractmethod
    def generate(
        self,
        query: str,
        context: ContextPackage,
        prompt: PromptBundle,
        citations: list[Citation],
    ) -> str:
        raise NotImplementedError


class HeuristicAnswerGenerator(AnswerGenerator):
    provider_name = "heuristic"
    model_name = "grounded-synthesis-v1"

    def generate(
        self,
        query: str,
        context: ContextPackage,
        prompt: PromptBundle,
        citations: list[Citation],
    ) -> str:
        if not citations:
            return (
                "I do not have enough retrieved evidence to answer this question confidently. "
                "Try broadening the query or indexing more source material."
            )

        bullets = [self._build_bullet(query=query, citation=citation) for citation in citations]
        intro = f"Here is a grounded answer to `{query}` based on the retrieved context:"
        return intro + "\n\n" + "\n".join(f"- {bullet}" for bullet in bullets)

    def _build_bullet(self, query: str, citation: Citation) -> str:
        sentence = _extract_lead_sentence(citation.text)
        if not sentence:
            sentence = "The retrieved evidence contains relevant support for this topic"
        sentence = sentence.rstrip(". ") + "."
        return f"{sentence} {citation.marker}"


class OpenAIAnswerGenerator(AnswerGenerator):
    provider_name = "openai"

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self.model_name = settings.openai.model

    def generate(
        self,
        query: str,
        context: ContextPackage,
        prompt: PromptBundle,
        citations: list[Citation],
    ) -> str:
        if not self._settings.openai.api_key:
            raise RuntimeError("RAG_OPENAI__API_KEY is not configured.")

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("openai package is not installed.") from exc

        client = OpenAI(
            api_key=self._settings.openai.api_key,
            timeout=self._settings.openai.timeout_seconds,
        )
        response = client.chat.completions.create(
            model=self._settings.openai.model,
            temperature=self._settings.generation.temperature,
            max_tokens=self._settings.generation.max_output_tokens,
            messages=[
                {"role": "system", "content": prompt.system_prompt},
                {"role": "user", "content": prompt.user_prompt},
            ],
        )
        content = response.choices[0].message.content or ""
        content = content.strip()
        if not content:
            raise RuntimeError("OpenAI returned an empty answer.")
        if citations and not _contains_citation_markers(content):
            logger.warning("generation_missing_citations", provider=self.provider_name)
        return content


def create_answer_generator(settings: Settings, provider: str | None = None) -> AnswerGenerator:
    selected_provider = (provider or settings.generation.provider).lower()
    if selected_provider == "openai":
        return OpenAIAnswerGenerator(settings)
    return HeuristicAnswerGenerator()


_CITATION_PATTERN = re.compile(r"\[[0-9]+\]")


def _contains_citation_markers(text: str) -> bool:
    return bool(_CITATION_PATTERN.search(text))


def _extract_lead_sentence(text: str) -> str:
    stripped = " ".join(text.split())
    if not stripped:
        return ""
    match = re.split(r"(?<=[.!?])\s+", stripped, maxsplit=1)
    return match[0].strip()
