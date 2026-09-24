#!/usr/bin/env python3
"""Merge sharded ce_scores_*.part{i}of{N}.npz into one cache (query order)."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--parts-glob",
        required=True,
        help="Glob of part files, e.g. path/to/ce_scores_train3321_sent_s1.part*of4.npz",
    )
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    p = Path(args.parts_glob)
    parts = sorted(p.parent.glob(p.name))
    if not parts:
        raise SystemExit(f"no parts match {args.parts_glob}")
    scores = []
    qids = []
    meta: dict = {}
    for part in parts:
        z = np.load(part, allow_pickle=True)
        scores.append(z["ce_scores"])
        qids.append(np.asarray(z["query_ids"]))
        if not meta:
            meta = {k: z[k] for k in z.files if k not in ("ce_scores", "query_ids")}
        print(f"loaded {part.name} n={len(z['query_ids'])}", flush=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "ce_scores": np.concatenate(scores, axis=0),
        "query_ids": np.concatenate(qids, axis=0),
        **meta,
    }
    np.savez_compressed(out, **payload)
    print(f"wrote {out} n={payload['query_ids'].shape[0]}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
