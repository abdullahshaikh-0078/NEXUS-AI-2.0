from __future__ import annotations

import re
from collections.abc import Iterable

from rag_service.ingestion.models import ChunkMetadata, ChunkStrategy, DocumentChunk, ParsedDocument

SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def chunk_document(
    document: ParsedDocument,
    strategy: ChunkStrategy,
    chunk_size: int,
    chunk_overlap: int,
    semantic_similarity_threshold: float,
) -> list[DocumentChunk]:
    if strategy == "fixed":
        pieces = _fixed_chunks(document.cleaned_text, chunk_size, chunk_overlap)
        section_titles = [document.title for _ in pieces]
    elif strategy == "semantic":
        pieces = _semantic_chunks(document.cleaned_text, chunk_size, semantic_similarity_threshold)
        section_titles = [document.title for _ in pieces]
    else:
        section_pairs = _structure_aware_chunks(document, chunk_size, chunk_overlap)
        pieces = [text for _, text in section_pairs]
        section_titles = [section_title for section_title, _ in section_pairs]

    chunks: list[DocumentChunk] = []
    for index, (section_title, text) in enumerate(zip(section_titles, pieces, strict=True)):
        normalized_text = text.strip()
        if not normalized_text:
            continue
        chunks.append(
            DocumentChunk(
                chunk_id=f"{document.document_id}:{index:04d}",
                text=normalized_text,
                metadata=ChunkMetadata(
                    document_id=document.document_id,
                    source_path=str(document.source_path),
                    source_type=document.source_type,
                    title=document.title,
                    section_title=section_title,
                    chunk_index=index,
                    chunk_strategy=strategy,
                    char_count=len(normalized_text),
                    word_count=len(normalized_text.split()),
                ),
            )
        )
    return chunks


def _fixed_chunks(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    words = text.split()
    if not words:
        return []

    chunks: list[str] = []
    current: list[str] = []
    current_length = 0
    step_back = max(chunk_overlap, 0)

    for word in words:
        projected_length = current_length + len(word) + (1 if current else 0)
        if current and projected_length > chunk_size:
            chunks.append(" ".join(current))
            overlap_words = _tail_words(current, step_back)
            current = overlap_words[:] if overlap_words else []
            current_length = len(" ".join(current))
        current.append(word)
        current_length = len(" ".join(current))

    if current:
        chunks.append(" ".join(current))
    return chunks


def _semantic_chunks(text: str, chunk_size: int, similarity_threshold: float) -> list[str]:
    sentences = [sentence.strip() for sentence in SENTENCE_SPLIT_RE.split(text) if sentence.strip()]
    if not sentences:
        return []

    chunks: list[str] = []
    current_sentences: list[str] = [sentences[0]]

    for sentence in sentences[1:]:
        current_text = " ".join(current_sentences)
        similarity = _sentence_similarity(current_sentences[-1], sentence)
        projected = f"{current_text} {sentence}".strip()
        if len(projected) > chunk_size or similarity < similarity_threshold:
            chunks.append(current_text)
            current_sentences = [sentence]
            continue
        current_sentences.append(sentence)

    if current_sentences:
        chunks.append(" ".join(current_sentences))
    return chunks


def _structure_aware_chunks(
    document: ParsedDocument,
    chunk_size: int,
    chunk_overlap: int,
) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for section in document.sections:
        section_chunks = _fixed_chunks(section.content, chunk_size, chunk_overlap)
        for chunk in section_chunks or [section.content]:
            if chunk.strip():
                pairs.append((section.heading, chunk))
    return pairs


def _tail_words(words: Iterable[str], max_characters: int) -> list[str]:
    reversed_words = list(words)[::-1]
    selected: list[str] = []
    total = 0
    for word in reversed_words:
        projected = total + len(word) + (1 if selected else 0)
        if projected > max_characters:
            break
        selected.append(word)
        total = projected
    return list(reversed(selected))


def _sentence_similarity(left: str, right: str) -> float:
    left_tokens = {token.lower() for token in re.findall(r"\w+", left)}
    right_tokens = {token.lower() for token in re.findall(r"\w+", right)}
    if not left_tokens or not right_tokens:
        return 0.0
    overlap = len(left_tokens & right_tokens)
    universe = len(left_tokens | right_tokens)
    return overlap / universe if universe else 0.0

