from __future__ import annotations

import argparse
from pathlib import Path

from rag_service.core.config import get_settings
from rag_service.ingestion.pipeline import ingest_directory


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the offline document ingestion pipeline.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="Directory containing raw documents.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=None,
        help="JSONL output path.",
    )
    parser.add_argument(
        "--strategy",
        choices=["fixed", "semantic", "structure_aware"],
        default=None,
        help="Chunking strategy override.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    settings = get_settings()

    input_dir = args.input_dir or Path(settings.ingestion.input_dir)
    output_file = args.output_file or Path(settings.ingestion.output_dir) / "chunks.jsonl"

    result = ingest_directory(
        input_dir=input_dir,
        output_path=output_file,
        settings=settings,
        strategy=args.strategy,
    )
    print(
        f"Ingestion completed: documents={result.documents_processed}, "
        f"chunks={result.chunks_created}, output={result.output_path}"
    )


if __name__ == "__main__":
    main()
