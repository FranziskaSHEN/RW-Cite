#!/usr/bin/env python3
"""Prep side-path cite eval for ewm Release-3.0 test381 (does NOT overwrite 3.1/370).

Writes under <lane>/adapter/rr_v6/data_r30_381/:
  pools_test_l0.jsonl, cite_eval.jsonl, meta.json

Uses frozen windows_test381 + ce_scores_test381_sent_s1 + E4 scores + L0_rrf fuse.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import networkx as nx

from rwcite.adapter.dataset_v6 import build_cite_eval_rows
from rwcite.adapter.dump_l0_pools import fuse_top_pools, write_jsonl
from rwcite.adapter.paths import resolve_adapter_v6
from rwcite.gat import protocol as P
from rwcite.gat.l0_recipe import DEFAULT_ALPHA, DEFAULT_RRF_K, DEFAULT_RRF_W
from rwcite.ranker.reference_recommend import normalize_arxiv_id


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--domain", default="ewm", choices=("ewm",))
    ap.add_argument("--pool-n", type=int, default=50)
    ap.add_argument("--root", default=None)
    args = ap.parse_args()

    root = P.root_path() if not args.root else Path(args.root)
    paths = resolve_adapter_v6(args.domain, root)
    lane = paths.root / paths.lane
    gat_mvp = lane / "ranker" / "gat_mvp"
    eval_dir = lane / "ranker" / "eval"
    out_dir = paths.adapter_dir / "data_r30_381"
    out_dir.mkdir(parents=True, exist_ok=True)

    windows = gat_mvp / "windows_test381.npz"
    ce = gat_mvp / "ce_scores_test381_sent_s1.npz"
    gat = eval_dir / "ewm_gat_mvp_test381_e4_scores_merged.json"
    o2 = paths.gat.full_o2_gexf
    for p in (windows, ce, gat, o2):
        if not p.is_file():
            raise SystemExit(f"missing {p}")

    # Overlap note vs current 3.1 train (adapter was trained on 3.1)
    te30 = {
        normalize_arxiv_id(str(x))
        for x in json.loads(
            (
                lane / "data/_admit_backup_20260912_163729/splits/test_sources.json"
            ).read_text(encoding="utf-8")
        )
    }
    tr31 = {
        normalize_arxiv_id(str(x))
        for x in json.loads((lane / "data/splits/train_sources.json").read_text(encoding="utf-8"))
    }
    leak_ids = sorted(te30 & tr31)

    print(f"fusing L0_rrf Top-{args.pool_n} on test381 …", flush=True)
    rows = fuse_top_pools(
        paths=paths,
        windows_path=windows,
        ce_npz=ce,
        gat_eval_json=gat,
        o2_gexf=o2,
        pool_n=int(args.pool_n),
        alpha=DEFAULT_ALPHA,
        rrf_k=DEFAULT_RRF_K,
        rrf_w=DEFAULT_RRF_W,
    )
    # stamp pool_source for this side path
    for r in rows:
        r["pool_source"] = "l0_rrf_3.0_test381"
    pools_path = out_dir / "pools_test_l0.jsonl"
    write_jsonl(pools_path, rows)

    g = nx.read_gexf(str(o2), node_type=None, relabel=False, version="1.2draft")
    node_by = {normalize_arxiv_id(str(n)): n for n in g.nodes}
    cite_eval = build_cite_eval_rows(
        rows, g, pool_n=int(args.pool_n), node_by=node_by
    )
    for r in cite_eval:
        r["pool_source"] = "l0_rrf_3.0_test381"
        r["split_tag"] = "r30_test381"
    cite_path = out_dir / "cite_eval.jsonl"
    write_jsonl(cite_path, cite_eval)

    meta = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "domain": args.domain,
        "kind": "rr_v6_cite_eval_r30_381",
        "note": (
            "Cross-split: 3.1-trained adapter_rr weights on Release-3.0 test381 "
            "L0 pools. Not the official 3.1/370 gate. "
            f"n_overlap_3.0test_in_3.1train={len(leak_ids)}"
        ),
        "n_pools": len(rows),
        "n_cite_eval": len(cite_eval),
        "leak_3.0test_in_3.1train": leak_ids,
        "windows": str(windows),
        "ce": str(ce),
        "gat_scores": str(gat),
        "pools_test": str(pools_path),
        "cite_eval": str(cite_path),
        "adapter_model": str(paths.model_dir),
        "alpha": DEFAULT_ALPHA,
        "rrf_k": DEFAULT_RRF_K,
        "rrf_w": DEFAULT_RRF_W,
    }
    (out_dir / "meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(meta, indent=2), flush=True)
    print(f"OUT_DIR={out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
