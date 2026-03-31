from __future__ import annotations

import hashlib
import re
from pathlib import Path

from bs4 import BeautifulSoup
from pypdf import PdfReader

from rag_service.ingestion.cleaners import clean_text
from rag_service.ingestion.models import DocumentSection, ParsedDocument

HEADING_RE = re.compile(r"^(#{1,6}\s+.+|[A-Z][A-Z0-9\s:-]{3,})$")


def parse_document(path: Path) -> ParsedDocument:
    suffix = path.suffix.lower()
    if suffix == ".txt":
        raw_text = path.read_text(encoding="utf-8")
        title = path.stem.replace("_", " ").replace("-", " ").title()
    elif suffix == ".html":
        raw_text, title = _parse_html(path)
    elif suffix == ".pdf":
        raw_text, title = _parse_pdf(path)
    else:
        raise ValueError(f"Unsupported document type: {suffix}")

    cleaned_text = clean_text(raw_text)
    sections = _extract_sections(cleaned_text, fallback_title=title)

    return ParsedDocument(
        document_id=_build_document_id(path),
        source_path=path,
        source_type=suffix.lstrip("."),
        title=title,
        raw_text=raw_text,
        cleaned_text=cleaned_text,
        sections=sections,
        metadata={
            "file_name": path.name,
            "suffix": suffix,
            "size_bytes": path.stat().st_size,
            "section_count": len(sections),
        },
    )


def _parse_html(path: Path) -> tuple[str, str]:
    soup = BeautifulSoup(path.read_text(encoding="utf-8"), "html.parser")
    title = (soup.title.string or path.stem).strip() if soup.title else path.stem

    body = soup.body or soup
    lines: list[str] = []
    for element in body.find_all(["h1", "h2", "h3", "p", "li"]):
        text = element.get_text(" ", strip=True)
        if not text:
            continue
        if element.name in {"h1", "h2", "h3"}:
            lines.append(text.upper())
        else:
            lines.append(text)
    return "\n".join(lines), title


def _parse_pdf(path: Path) -> tuple[str, str]:
    reader = PdfReader(str(path))
    extracted_pages = [page.extract_text() or "" for page in reader.pages]
    title = path.stem.replace("_", " ").replace("-", " ").title()
    return "\n".join(extracted_pages), title


def _extract_sections(text: str, fallback_title: str) -> list[DocumentSection]:
    if not text:
        return [DocumentSection(heading=fallback_title, content="", level=1)]

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return [DocumentSection(heading=fallback_title, content=text, level=1)]

    sections: list[DocumentSection] = []
    current_heading = fallback_title
    buffer: list[str] = []

    for line in lines:
        if HEADING_RE.match(line):
            if buffer:
                sections.append(
                    DocumentSection(
                        heading=current_heading,
                        content="\n".join(buffer).strip(),
                        level=1,
                    )
                )
                buffer = []
            current_heading = line.lstrip("# ").strip().title()
            continue
        buffer.append(line)

    if buffer or not sections:
        sections.append(
            DocumentSection(
                heading=current_heading,
                content="\n".join(buffer).strip(),
                level=1,
            )
        )

    return [section for section in sections if section.content]


def _build_document_id(path: Path) -> str:
    digest = hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()
    return digest[:16]

