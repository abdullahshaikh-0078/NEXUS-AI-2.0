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

from rag_service.core.config import get_settings
from rag_service.evaluation.runner import run_experiment_suite


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run retrieval and generation evaluation experiments.")
    parser.add_argument("--dataset-path", type=Path, default=None, help="JSONL evaluation dataset path.")
    parser.add_argument("--manifest-path", type=Path, default=None, help="Index manifest path.")
    parser.add_argument(
        "--systems",
        nargs="*",
        default=None,
        choices=["bm25", "dense", "hybrid"],
        help="Retrieval systems to benchmark.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    settings = get_settings()
    result = run_experiment_suite(
        settings,
        dataset_path=args.dataset_path,
        manifest_path=args.manifest_path,
        systems=args.systems,
    )
    print(result.benchmark_table)
    if result.artifacts is not None:
        print(f"Artifacts written to: {result.artifacts.run_dir}")


if __name__ == "__main__":
    main()





