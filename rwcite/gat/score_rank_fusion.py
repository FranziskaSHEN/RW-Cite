"""Paper-defined score-and-rank fusion entry point.

The implementation retains the legacy recipe module as a compatibility layer
for existing experiment artifacts, while exposing publication-facing names to
new callers.
"""

from __future__ import annotations

from rwcite.gat.l0_recipe import (
    DEFAULT_ALPHA,
    DEFAULT_RRF_K as DEFAULT_RANK_OFFSET,
    DEFAULT_RRF_W as DEFAULT_RANK_WEIGHT,
    ensure_l0_default,
    main,
)


ensure_score_rank_fusion = ensure_l0_default

__all__ = [
    "DEFAULT_ALPHA",
    "DEFAULT_RANK_OFFSET",
    "DEFAULT_RANK_WEIGHT",
    "ensure_score_rank_fusion",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
