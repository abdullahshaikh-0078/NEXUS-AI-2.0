from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from reportlab.pdfgen import canvas

BACKEND_DIR = Path(__file__).resolve().parents[1]
SOURCE_DIR = BACKEND_DIR / "src"
TESTS_DIR = BACKEND_DIR / "tests"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

from rag_service.api.app import create_app
from rag_service.core.config import get_settings


@pytest.fixture(autouse=True)
def clear_settings_cache() -> None:
    get_settings.cache_clear()


@pytest.fixture()
def client() -> TestClient:
    return TestClient(create_app())


@pytest.fixture()
def sample_document_dir(tmp_path: Path) -> Path:
    document_dir = tmp_path / "raw_documents"
    document_dir.mkdir()

    (document_dir / "platform_overview.txt").write_text(
        "\n".join(
            [
                "SYSTEM OVERVIEW",
                "The ingestion layer extracts content from private corpora.",
                "It prepares normalized text for indexing and retrieval.",
                "",
                "ADAPTIVE CHUNKING",
                "Fixed chunking provides deterministic windows.",
                "Semantic chunking keeps related ideas together.",
            ]
        ),
        encoding="utf-8",
    )

    (document_dir / "retrieval_notes.html").write_text(
        """
        <html>
          <head><title>Retrieval Notes</title></head>
          <body>
            <h1>Hybrid Retrieval</h1>
            <p>Combine lexical and dense retrieval for higher recall.</p>
            <h2>Metadata</h2>
            <p>Track source, section, and chunk identifiers for every segment.</p>
          </body>
        </html>
        """.strip(),
        encoding="utf-8",
    )

    _build_pdf(
        document_dir / "deployment_guide.pdf",
        [
            "DEPLOYMENT PLAYBOOK",
            "Deployment Playbook for the ingestion service.",
            "Ship offline processing as a reproducible batch job.",
        ],
    )

    return document_dir


def _build_pdf(path: Path, lines: list[str]) -> None:
    pdf = canvas.Canvas(str(path))
    y_position = 800
    for line in lines:
        pdf.drawString(72, y_position, line)
        y_position -= 18
    pdf.save()
