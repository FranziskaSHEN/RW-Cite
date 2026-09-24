#!/usr/bin/env python3
"""Merge RR v6 cite eval shards → pairs sidecar + final JSON (CPU; optional BertScore).

Use when DDP crashed after shards were written, or to fill L2 offline:

  python scripts/merge_rr_v6_cite_shards.py \\
    --out embodied_world_model_retrieval/adapter/rr_v6/data/eval_cite_full370.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rwcite.cli.eval_rr_adapter_cite import (  # noqa: E402
    _atomic_write_json,
    _load_done_sids,
    _shard_dir,
    _shard_path,
    _summarize,
    _write_intermediates,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True, help="final eval JSON path")
    ap.add_argument("--world-size", type=int, default=8)
    ap.add_argument("--skip-bertscore", action="store_true")
    args = ap.parse_args()
    out: Path = args.out
    if not out.is_absolute():
        out = ROOT / out

    merged: dict[str, dict] = {}
    for r in range(int(args.world_size)):
        p = _shard_path(out, r)
        part = _load_done_sids(p)
        print(f"rank{r}: {len(part)} from {p}")
        merged.update(part)
    details = list(merged.values())
    if not details:
        # also accept details_with_cites.json
        raw = out.with_name(out.stem + ".details_with_cites.json")
        if raw.is_file():
            details = list(json.loads(raw.read_text(encoding="utf-8")).get("details") or [])
            print(f"loaded {len(details)} from {raw}")
    if not details:
        raise SystemExit(f"no shard details under {_shard_dir(out)}")

    pairs_path = _write_intermediates(out, details)
    if args.skip_bertscore:
        summary = {
            "label": "v6",
            "n": len(details),
            "kind": "rr_v6_cite_l0_l2",
            "note": "merged shards; BertScore skipped",
            "pairs_sidecar": str(pairs_path),
        }
        # still compute non-L2 means via _summarize but catch bertscore
    summary = _summarize(details, label="v6")
    if args.skip_bertscore:
        summary["mean_cite_bertscore_f1"] = None
        summary["bertscore_error"] = "skipped"
    summary.update(
        {
            "n": len(details),
            "world_size": int(args.world_size),
            "shard_dir": str(_shard_dir(out)),
            "pairs_sidecar": str(pairs_path),
            "merged_from_shards": True,
            "note": "AS v6 L0–L2; merged from shards",
        }
    )
    slim = []
    for d in details:
        dd = dict(d)
        dd.pop("bertscore_pairs", None)
        slim.append(dd)
    _atomic_write_json(out, {"summary": summary, "details": slim})
    print(json.dumps({k: summary[k] for k in summary if k != "details"}, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
