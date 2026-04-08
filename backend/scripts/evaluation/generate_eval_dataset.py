from __future__ import annotations

import sys
from pathlib import Path as _Path

SCRIPTS_ROOT = _Path(__file__).resolve().parents[1]
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from shared.bootstrap import bootstrap_paths

bootstrap_paths()

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





