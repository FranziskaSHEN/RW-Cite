#!/usr/bin/env python3
"""Quality gate for domain citelink before HN mining.

Fails hard (exit 1) if best_val_loss or standalone n80 hits@10 miss thresholds.
Never loads legacy citelink-v3c weights; only checks a freshly trained model.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", required=True, help="citelink meta.json next to model.npz")
    ap.add_argument(
        "--eval-merged",
        default="",
        help="optional n80 merged.json from standalone citelink eval",
    )
    ap.add_argument("--max-val-loss", type=float, default=0.40)
    ap.add_argument("--min-hits-at-10", type=float, default=2.0)
    ap.add_argument("--out", default="", help="write gate result JSON")
    args = ap.parse_args()

    meta_path = Path(args.meta)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    val = float(meta.get("best_val_loss", 1e9))
    hits = None
    if args.eval_merged:
        ev = json.loads(Path(args.eval_merged).read_text(encoding="utf-8"))
        summary = ev.get("summary") or ev
        hits = float(summary.get("mean_hits_at_10", -1.0))

    pass_val = val <= args.max_val_loss
    pass_hits = True if hits is None else hits >= args.min_hits_at_10
    ok = pass_val and pass_hits
    report = {
        "ok": ok,
        "best_val_loss": val,
        "max_val_loss": args.max_val_loss,
        "pass_val_loss": pass_val,
        "mean_hits_at_10": hits,
        "min_hits_at_10": args.min_hits_at_10 if hits is not None else None,
        "pass_hits_at_10": pass_hits,
        "meta": str(meta_path),
        "eval_merged": args.eval_merged or None,
        "note": "Do not use citelink for HN if ok=false",
    }
    text = json.dumps(report, indent=2)
    print(text, flush=True)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
    if not ok:
        print("GATE FAIL: citelink quality below threshold", file=sys.stderr)
        return 1
    print("GATE PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
