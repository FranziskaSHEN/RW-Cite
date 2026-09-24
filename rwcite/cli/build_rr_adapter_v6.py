#!/usr/bin/env python3
"""Build RR v6 Cite-only train jsonl from L0 pools + full O2 GEXF."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rwcite.adapter.dataset_v6 import build_domain_v6
from rwcite.gat import protocol as P


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--domain", required=True, choices=("ewm", "sqc", "gw"))
    ap.add_argument("--pool-n", type=int, default=50)
    ap.add_argument("--jaccard-thr", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--root", default=None)
    args = ap.parse_args(argv)
    root = P.root_path() if not args.root else Path(args.root)
    meta = build_domain_v6(
        args.domain,
        root=root,
        pool_n=int(args.pool_n),
        jaccard_thr=float(args.jaccard_thr),
        seed=int(args.seed),
    )
    print(json.dumps({"ok": True, "n_train_rows": meta["n_train_rows"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
