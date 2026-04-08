from __future__ import annotations

import argparse
from pathlib import Path

from rag_service.evaluation.dataset import bootstrap_dataset_from_chunks


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a starter evaluation dataset from chunked corpus data.")
    parser.add_argument("--chunks-file", type=Path, required=True, help="Path to the chunks JSONL file.")
    parser.add_argument("--output-file", type=Path, required=True, help="Output JSONL dataset path.")
    parser.add_argument("--max-documents", type=int, default=5, help="Maximum number of document-based queries to create.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    samples = bootstrap_dataset_from_chunks(
        args.chunks_file,
        args.output_file,
        max_documents=args.max_documents,
    )
    print(f"Generated {len(samples)} evaluation samples at {args.output_file}")


if __name__ == "__main__":
    main()
