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

from rag_service.core.config import get_settings
from rag_service.generation.pipeline import generate_grounded_answer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run grounded generation over the current RAG context pipeline.")
    parser.add_argument("query", type=str, help="Raw user query to answer with grounded citations.")
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

    if args.manifest_path is not None:
        settings = settings.model_copy(
            update={
                "retrieval": settings.retrieval.model_copy(
                    update={"manifest_path": str(args.manifest_path)}
                )
            }
        )

    result = generate_grounded_answer(args.query, settings=settings)
    print(json.dumps(result.model_dump(mode="json"), indent=2))


if __name__ == "__main__":
    main()





