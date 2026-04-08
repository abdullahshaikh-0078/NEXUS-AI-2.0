from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path

from rag_service.evaluation.models import EvaluationSample
from rag_service.indexing.loaders import load_chunks


def load_evaluation_dataset(path: Path) -> list[EvaluationSample]:
    samples: list[EvaluationSample] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            samples.append(_normalize_sample(payload))
    return samples


def write_evaluation_dataset(path: Path, samples: list[EvaluationSample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(_serialize_sample(sample), ensure_ascii=True) + "\n")


def bootstrap_dataset_from_chunks(
    chunks_path: Path,
    output_path: Path,
    *,
    max_documents: int = 5,
) -> list[EvaluationSample]:
    chunks = load_chunks(chunks_path)
    by_document = OrderedDict()
    for chunk in chunks:
        by_document.setdefault(chunk.metadata.document_id, chunk)

    samples: list[EvaluationSample] = []
    for index, chunk in enumerate(list(by_document.values())[:max_documents], start=1):
        title = chunk.metadata.title or Path(chunk.metadata.source_path).stem.replace("_", " ").title()
        section = chunk.metadata.section_title or "overview"
        query = f"What does {title} say about {section.lower()}?"
        expected_answer = _first_sentence(chunk.text)
        samples.append(
            EvaluationSample(
                query_id=f"sample-{index:03d}",
                query=query,
                relevant_ids=[chunk.metadata.document_id],
                target="document",
                expected_answer=expected_answer,
                metadata={
                    "title": title,
                    "section_title": chunk.metadata.section_title,
                    "source_path": chunk.metadata.source_path,
                },
            )
        )

    write_evaluation_dataset(output_path, samples)
    return samples


def _normalize_sample(payload: dict[str, object]) -> EvaluationSample:
    if "relevant_ids" in payload:
        return EvaluationSample.model_validate(payload)
    if "relevant_document_ids" in payload:
        return EvaluationSample(
            query_id=str(payload.get("query_id") or payload.get("id") or payload.get("query")),
            query=str(payload["query"]),
            relevant_ids=[str(value) for value in payload.get("relevant_document_ids", [])],
            target="document",
            expected_answer=str(payload.get("expected_answer") or ""),
            metadata=dict(payload.get("metadata") or {}),
        )
    if "relevant_chunk_ids" in payload:
        return EvaluationSample(
            query_id=str(payload.get("query_id") or payload.get("id") or payload.get("query")),
            query=str(payload["query"]),
            relevant_ids=[str(value) for value in payload.get("relevant_chunk_ids", [])],
            target="chunk",
            expected_answer=str(payload.get("expected_answer") or ""),
            metadata=dict(payload.get("metadata") or {}),
        )
    raise ValueError("Evaluation samples must define relevant_ids, relevant_document_ids, or relevant_chunk_ids.")


def _serialize_sample(sample: EvaluationSample) -> dict[str, object]:
    payload: dict[str, object] = {
        "query_id": sample.query_id,
        "query": sample.query,
        "expected_answer": sample.expected_answer,
        "metadata": sample.metadata,
    }
    if sample.target == "document":
        payload["relevant_document_ids"] = sample.relevant_ids
    else:
        payload["relevant_chunk_ids"] = sample.relevant_ids
    return payload


def _first_sentence(text: str) -> str:
    normalized = " ".join(text.split())
    if not normalized:
        return ""
    for marker in [". ", "? ", "! "]:
        if marker in normalized:
            return normalized.split(marker, 1)[0].strip() + marker.strip()
    return normalized[:180].strip()
