from __future__ import annotations

import sys
from pathlib import Path


def bootstrap_paths() -> None:
    scripts_dir = Path(__file__).resolve().parents[1]
    backend_dir = scripts_dir.parent
    source_dir = backend_dir / "src"
    source_path = str(source_dir)
    if source_path not in sys.path:
        sys.path.insert(0, source_path)
