from __future__ import annotations

import json
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

from rag_service.ingestion.models import DocumentChunk
from rag_service.indexing.models import BuildManifest

ModelT = TypeVar("ModelT", bound=BaseModel)


def load_chunks(path: Path) -> list[DocumentChunk]:
    return _load_jsonl_models(path, DocumentChunk)


def load_manifest(path: Path) -> BuildManifest:
    return BuildManifest.model_validate_json(path.read_text(encoding="utf-8"))


def write_jsonl_models(path: Path, records: list[BaseModel]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.model_dump(mode="json")) + "\n")


def _load_jsonl_models(path: Path, model_cls: type[ModelT]) -> list[ModelT]:
    records: list[ModelT] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(model_cls.model_validate_json(line))
    return records
