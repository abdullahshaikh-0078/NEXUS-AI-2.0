from __future__ import annotations

import argparse
from pathlib import Path

from rag_service.core.config import get_settings
from rag_service.core.logging import configure_logging, get_logger
from rag_service.indexing.pipeline import build_indexes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build embeddings and indexes from chunk artifacts.")
    parser.add_argument("--chunks-file", type=Path, help="Path to the JSONL chunk artifact.")
    parser.add_argument("--output-dir", type=Path, help="Directory where index artifacts are stored.")
    parser.add_argument("--version", type=str, help="Embedding and index version label.")
    parser.add_argument(
        "--embedding-provider",
        choices=["sentence_transformers", "hash"],
        help="Override the configured embedding provider.",
    )
    parser.add_argument(
        "--dense-backend",
        choices=["faiss", "native"],
        help="Override the configured dense index backend.",
    )
    parser.add_argument(
        "--sparse-backend",
        choices=["whoosh", "native"],
        help="Override the configured sparse index backend.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = get_settings()

    if args.embedding_provider:
        settings.indexing.embedding_provider = args.embedding_provider
    if args.dense_backend:
        settings.indexing.dense_backend = args.dense_backend
    if args.sparse_backend:
        settings.indexing.sparse_backend = args.sparse_backend

    configure_logging(settings)
    logger = get_logger(__name__)

    chunks_file = args.chunks_file or Path(settings.indexing.input_chunk_file)
    output_dir = args.output_dir or Path(settings.indexing.output_dir)
    result = build_indexes(
        chunks_file=chunks_file,
        output_dir=output_dir,
        settings=settings,
        version=args.version,
    )

    logger.info(
        "indexing_run_finished",
        version=result.version,
        total_chunks=result.total_chunks,
        manifest_path=str(result.manifest_path),
    )


if __name__ == "__main__":
    main()
