from __future__ import annotations

import sys
from pathlib import Path as _Path

SCRIPTS_ROOT = _Path(__file__).resolve().parents[1]
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from shared.bootstrap import bootstrap_paths

bootstrap_paths()

import argparse
import json
from pathlib import Path

from rag_service.context.pipeline import build_context
from rag_service.core.config import get_settings
from rag_service.reranking.pipeline import rerank_candidates
from rag_service.retrieval.pipeline import hybrid_retrieve


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the context engineering pipeline.")
    parser.add_argument("query", type=str, help="Raw user query to build grounded context for.")
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

    retrieval = hybrid_retrieve(args.query, settings=settings, manifest_path=args.manifest_path)
    reranking = rerank_candidates(args.query, settings=settings, retrieval=retrieval)
    result = build_context(args.query, settings=settings, reranking=reranking)
    print(json.dumps(result.model_dump(mode="json"), indent=2))


if __name__ == "__main__":
    main()





