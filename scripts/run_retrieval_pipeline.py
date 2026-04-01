from __future__ import annotations

import argparse
import json
from pathlib import Path

from rag_service.core.config import get_settings
from rag_service.retrieval.pipeline import hybrid_retrieve


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the hybrid retrieval pipeline.")
    parser.add_argument("query", type=str, help="Raw user query to retrieve against indexed chunks.")
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=None,
        help="Path to the phase 3 indexing manifest.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    settings = get_settings()

    result = hybrid_retrieve(args.query, settings=settings, manifest_path=args.manifest_path)
    print(json.dumps(result.model_dump(mode="json"), indent=2))


if __name__ == "__main__":
    main()
