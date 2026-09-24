from __future__ import annotations

import os
from pathlib import Path


def get_root() -> Path:
    env = os.environ.get("RWCITE_ROOT") or os.environ.get("RWCITE_DATA_ROOT")
    if env:
        return Path(env).resolve()
    # rwcite/runtime/paths.py → parents[2] = RW-Cite root
    return Path(__file__).resolve().parents[2]


RWCITE_ROOT = get_root()


def data_path(*parts: str) -> Path:
    """Resolve a path under RWCITE_DATA_ROOT (default: RWCITE_ROOT)."""
    root = Path(os.environ.get("RWCITE_DATA_ROOT") or get_root()).resolve()
    return root.joinpath(*parts)
