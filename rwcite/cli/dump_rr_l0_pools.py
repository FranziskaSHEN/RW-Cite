#!/usr/bin/env python3
"""Dump fused Top-N pools into the citation-adapter artifact directory."""

from __future__ import annotations

import argparse
import json
import sys

from rwcite.adapter.dump_l0_pools import dump_domain_pools
from rwcite.gat import protocol as P


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--domain", required=True, choices=("ewm", "sqc", "gw"))
    ap.add_argument("--pool-n", type=int, default=50)
    ap.add_argument(
        "--splits",
        default="train,test",
        help="comma list: train and/or test",
    )
    ap.add_argument("--force-ce", action="store_true")
    ap.add_argument("--force-gat", action="store_true")
    ap.add_argument("--device", default=None)
    ap.add_argument("--root", default=None)
    args = ap.parse_args(argv)

    root = P.root_path() if not args.root else __import__("pathlib").Path(args.root)
    splits = tuple(s.strip() for s in args.splits.split(",") if s.strip())
    for s in splits:
        if s not in ("train", "test"):
            print(f"ERROR: bad split {s!r}", file=sys.stderr)
            return 2
    report = dump_domain_pools(
        args.domain,
        root=root,
        pool_n=int(args.pool_n),
        splits=splits,
        force_ce=bool(args.force_ce),
        force_gat=bool(args.force_gat),
        device=args.device,
    )
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
