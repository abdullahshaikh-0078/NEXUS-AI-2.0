from __future__ import annotations

import json
from pathlib import Path

from rag_service.core.config import Settings
from rag_service.ingestion.chunkers import chunk_document
from rag_service.ingestion.parsers import parse_document
from rag_service.ingestion.pipeline import ingest_directory


def test_parse_document_supports_txt_html_and_pdf(sample_document_dir: Path) -> None:
    txt_document = parse_document(sample_document_dir / "platform_overview.txt")
    html_document = parse_document(sample_document_dir / "retrieval_notes.html")
    pdf_document = parse_document(sample_document_dir / "deployment_guide.pdf")

    assert txt_document.source_type == "txt"
    assert html_document.title == "Retrieval Notes"
    assert "Deployment Playbook" in pdf_document.cleaned_text


def test_structure_aware_chunking_preserves_section_metadata(sample_document_dir: Path) -> None:
    document = parse_document(sample_document_dir / "platform_overview.txt")

    chunks = chunk_document(
        document=document,
        strategy="structure_aware",
        chunk_size=100,
        chunk_overlap=20,
        semantic_similarity_threshold=0.18,
    )

    assert len(chunks) >= 2
    assert all(chunk.metadata.section_title for chunk in chunks)
    assert {chunk.metadata.chunk_strategy for chunk in chunks} == {"structure_aware"}


def test_ingestion_pipeline_writes_jsonl_output(sample_document_dir: Path, tmp_path: Path) -> None:
    settings = Settings(
        ingestion={
            "input_dir": str(sample_document_dir),
            "output_dir": str(tmp_path),
            "default_strategy": "semantic",
            "chunk_size": 120,
            "chunk_overlap": 30,
            "semantic_similarity_threshold": 0.15,
        }
    )

    output_path = tmp_path / "chunks.jsonl"
    result = ingest_directory(sample_document_dir, output_path, settings)

    records = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]

    assert result.documents_processed == 3
    assert result.chunks_created == len(records)
    assert output_path.exists()
    assert {record["metadata"]["source_type"] for record in records} == {"txt", "html", "pdf"}
