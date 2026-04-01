from __future__ import annotations

import argparse
import json

from rag_service.core.config import get_settings
from rag_service.query.pipeline import process_query


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the online query processing pipeline.")
    parser.add_argument("query", type=str, help="Raw user query to clean, expand, and rewrite.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    settings = get_settings()

    result = process_query(args.query, settings=settings)
    print(json.dumps(result.model_dump(mode="json"), indent=2))


if __name__ == "__main__":
    main()
