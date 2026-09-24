"""L0_rrf alias — production default is L0_rrf (cutover 2026-09-12).

Delegates to ``l0_recipe.ensure_l0_default`` for DOMAIN=ewm|sqc|gw.
"""

from __future__ import annotations

import argparse

from rwcite.gat import protocol as P
from rwcite.gat.l0_recipe import (
    DEFAULT_ALPHA,
    DEFAULT_RRF_K,
    DEFAULT_RRF_W,
    ensure_l0_default,
)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Ensure L0_rrf (= production l0_default) recipe"
    )
    ap.add_argument(
        "--domain",
        default="ewm",
        choices=("ewm", "sqc", "gw", "fno", "wsi", "cosmo", "qopt", "sce", "hep", "exo", "radio", "radseg", "driving"),
    )
    ap.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    ap.add_argument("--rrf-k", type=int, default=DEFAULT_RRF_K)
    ap.add_argument("--rrf-w", type=float, default=DEFAULT_RRF_W)
    ap.add_argument("--force-cache", action="store_true")
    ap.add_argument("--force-blend", action="store_true")
    args = ap.parse_args(argv)
    ensure_l0_default(
        root=P.root_path(),
        domain=args.domain,
        alpha=args.alpha,
        rrf_k=args.rrf_k,
        rrf_w=args.rrf_w,
        force_cache=args.force_cache,
        force_blend=args.force_blend,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
