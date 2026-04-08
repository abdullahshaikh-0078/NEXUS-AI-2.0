from __future__ import annotations

from pathlib import Path

from rag_service.indexing.loaders import load_chunks, load_manifest
from rag_service.indexing.models import BuildManifest
from rag_service.ingestion.models import DocumentChunk


def load_retrieval_artifacts(manifest_path: Path) -> tuple[BuildManifest, dict[str, DocumentChunk]]:
    manifest = load_manifest(manifest_path)
    chunks = load_chunks(Path(manifest.source_chunks_path))
    return manifest, {chunk.chunk_id: chunk for chunk in chunks}
