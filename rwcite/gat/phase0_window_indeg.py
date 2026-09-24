"""Phase 0: struct@400 window in-degree distribution (GAT solution §5.3.6 / §10).

Decides whether L4 (query-conditioned readout) has enough surface area:

  window indeg median ≥ 8  → L4 primary
  3–8                    → L4 OK, cocite-primary neighbors
  ≤ 2                    → demote L4 to ablation

Only runs universe + struct coarse (no CE). Writes a JSON report under gat_mvp/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np

from rwcite.gat import NOFUTURE_GEXF_SHA256, protocol as P
from rwcite.ranker.reference_recommend import normalize_arxiv_id
from rwcite.ranker.rr_struct_coarse import rank_struct_coarse
from rwcite.ranker.rr_universe import build_rr_universe
from rwcite.runtime.deps import get_domain_cache, get_retriever_service


def _pct(vals: list[int], p: float) -> float:
    if not vals:
        return float("nan")
    a = np.asarray(vals, dtype=np.float64)
    return float(np.percentile(a, p))


def _decide(median: float) -> dict[str, Any]:
    if median >= 8:
        return {
            "band": "ge_8",
            "l4_role": "primary",
            "note": "L4 has sufficient surface; implement as designed.",
        }
    if median >= 3:
        return {
            "band": "3_to_8",
            "l4_role": "primary_cocite_heavy",
            "note": "L4 OK but neighbor set must be cocite-primary (in-neighbors sparse).",
        }
    return {
        "band": "le_2",
        "l4_role": "demote_to_ablation",
        "note": "L4 surface too small; ship struct-residual + semantic first; L4 as F5 ablation only.",
    }


def _load_unique_queries(jsonl: Path) -> list[dict[str, Any]]:
    """One row per source_id (merge gold_ids)."""
    by: dict[str, dict[str, Any]] = {}
    with jsonl.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("task") not in (None, "", "bundle"):
                continue
            sid = normalize_arxiv_id(str(row.get("source_id") or ""))
            if not sid:
                continue
            gold = {
                normalize_arxiv_id(str(g))
                for g in (row.get("gold_ids") or [])
                if str(g).strip()
            }
            if not gold:
                for g in row.get("gold") or []:
                    if isinstance(g, dict) and g.get("id"):
                        gold.add(normalize_arxiv_id(str(g["id"])))
            if sid not in by:
                by[sid] = {
                    "source_id": sid,
                    "title": row.get("title") or "",
                    "abstract": row.get("abstract") or "",
                    "gold_ids": set(gold),
                }
            else:
                by[sid]["gold_ids"] |= gold
                if not by[sid]["title"] and row.get("title"):
                    by[sid]["title"] = row["title"]
                if not by[sid]["abstract"] and row.get("abstract"):
                    by[sid]["abstract"] = row["abstract"]
    out = []
    for q in by.values():
        q["gold_ids"] = sorted(q["gold_ids"])
        out.append(q)
    out.sort(key=lambda r: r["source_id"])
    return out


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _indeg(graph: nx.DiGraph, pid: str) -> int:
    if pid not in graph:
        return 0
    return int(graph.in_degree(pid))


def run_phase0(
    *,
    domain: str,
    jsonl: Path,
    window: int,
    max_samples: int,
    which: str,
) -> dict[str, Any]:
    # Force linear struct, no CE pollution.
    os.environ.setdefault("RR_STRUCT_COARSE", "linear")
    os.environ["RR_RANKER_WITH_EMB"] = "0"
    os.environ.pop("RR_RANKER_CE_SENT_PATH", None)
    os.environ.pop("RR_RANKER_CE_PATH", None)

    root = P.root_path()
    gexf = root / P.NOFUTURE_GEXF
    gexf_sha = _sha256(gexf) if gexf.is_file() else ""
    if gexf_sha and gexf_sha != NOFUTURE_GEXF_SHA256:
        print(
            f"WARN: nofuture sha {gexf_sha[:16]}… != expected {NOFUTURE_GEXF_SHA256[:16]}…",
            flush=True,
        )

    cache = get_domain_cache().get(domain)
    graph = cache.graph
    assert graph is not None
    retriever = get_retriever_service()

    def search_fn(query: str, limit: int):
        hits = retriever.search(
            scope="domain",
            domain_id=domain,
            query=query,
            offset=0,
            limit=limit,
            sort="relevance",
        )
        return list(hits.get("results") or [])

    queries = _load_unique_queries(jsonl)
    if max_samples > 0:
        queries = queries[:max_samples]

    all_indeg: list[int] = []
    gold_in_win_indeg: list[int] = []
    per_query: list[dict[str, Any]] = []
    t0 = time.perf_counter()

    for i, q in enumerate(queries):
        sid = q["source_id"]
        t_q0 = time.perf_counter()
        pack = build_rr_universe(
            q["title"],
            q["abstract"],
            graph,
            search_fn=search_fn,
            exclude_id=sid,
            multi_query=True,
            prf=True,
        )
        ranked = rank_struct_coarse(
            title=q["title"],
            abstract=q["abstract"],
            universe_pack=pack,
            graph=graph,
            n=window,
            encode=None,
            retriever=None,
            query_id=sid,
        )
        ids = [normalize_arxiv_id(str(c.get("id") or "")) for c in ranked]
        ids = [x for x in ids if x]
        indegs = [_indeg(graph, pid) for pid in ids]
        gold_set = set(q["gold_ids"])
        gold_hit = [pid for pid in ids if pid in gold_set]
        gold_indegs = [_indeg(graph, pid) for pid in gold_hit]

        all_indeg.extend(indegs)
        gold_in_win_indeg.extend(gold_indegs)

        row = {
            "source_id": sid,
            "n_universe": len(pack.get("universe") or []),
            "n_window": len(ids),
            "n_gold": len(gold_set),
            "n_gold_in_window": len(gold_hit),
            "window_indeg_median": float(statistics.median(indegs)) if indegs else None,
            "window_indeg_mean": float(statistics.mean(indegs)) if indegs else None,
            "elapsed_s": float(time.perf_counter() - t_q0),
        }
        per_query.append(row)
        if (i + 1) % 10 == 0 or i == 0:
            print(
                f"[{i+1}/{len(queries)}] {sid} win={len(ids)} "
                f"med={row['window_indeg_median']} gold_in={len(gold_hit)} "
                f"{row['elapsed_s']:.2f}s",
                flush=True,
            )

    elapsed = float(time.perf_counter() - t0)
    med = float(statistics.median(all_indeg)) if all_indeg else float("nan")
    decision = _decide(med)

    def _dist(vals: list[int]) -> dict[str, Any]:
        if not vals:
            return {}
        return {
            "n": len(vals),
            "mean": float(statistics.mean(vals)),
            "median": float(statistics.median(vals)),
            "p25": _pct(vals, 25),
            "p75": _pct(vals, 75),
            "p90": _pct(vals, 90),
            "p95": _pct(vals, 95),
            "p99": _pct(vals, 99),
            "max": int(max(vals)),
            "frac_indeg_0": float(sum(1 for x in vals if x == 0) / len(vals)),
            "frac_indeg_le_2": float(sum(1 for x in vals if x <= 2) / len(vals)),
            "frac_indeg_ge_8": float(sum(1 for x in vals if x >= 8) / len(vals)),
            "frac_indeg_gt_32": float(sum(1 for x in vals if x > 32) / len(vals)),
        }

    # Full-graph baseline for comparison
    full_indeg = [int(graph.in_degree(n)) for n in graph.nodes()]

    report = {
        "phase": "0_window_indeg",
        "domain": domain,
        "which": which,
        "window": window,
        "n_queries": len(queries),
        "gexf": str(gexf),
        "gexf_sha256": gexf_sha,
        "elapsed_s": elapsed,
        "decision": decision,
        "window_candidate_indeg": _dist(all_indeg),
        "gold_in_window_indeg": _dist(gold_in_win_indeg),
        "full_graph_indeg": _dist(full_indeg),
        "per_query_median_of_medians": float(
            statistics.median(
                [r["window_indeg_median"] for r in per_query if r["window_indeg_median"] is not None]
            )
        )
        if per_query
        else None,
        "mean_gold_in_window": float(
            statistics.mean([r["n_gold_in_window"] for r in per_query])
        )
        if per_query
        else None,
        "mean_gold_in_window_frac": float(
            statistics.mean(
                [
                    r["n_gold_in_window"] / r["n_gold"]
                    for r in per_query
                    if r["n_gold"] > 0
                ]
            )
        )
        if per_query
        else None,
        "per_query": per_query,
    }
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="GAT Phase 0: window@W window indeg")
    ap.add_argument("--domain", default="ewm")
    ap.add_argument(
        "--which",
        choices=("test", "train"),
        default="test",
        help="jsonl split to measure (default: test — drives L4 decision)",
    )
    ap.add_argument("--window", type=int, default=P.STRUCT_WINDOW)
    ap.add_argument("--max-samples", type=int, default=0, help="0 = all unique sources")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    root = P.root_path()
    jsonl = root / (P.TEST_JSONL if args.which == "test" else P.TRAIN_JSONL)
    out = args.out or (
        root / P.GAT_MVP_CKPT_DIR / f"phase0_window_indeg_{args.which}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"Phase 0: domain={args.domain} which={args.which} window={args.window} "
        f"jsonl={jsonl}",
        flush=True,
    )
    report = run_phase0(
        domain=args.domain,
        jsonl=jsonl,
        window=args.window,
        max_samples=args.max_samples,
        which=args.which,
    )
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    win = report["window_candidate_indeg"]
    dec = report["decision"]
    print("\n== Phase 0 summary ==", flush=True)
    print(json.dumps({"window_candidate_indeg": win, "decision": dec}, indent=2), flush=True)
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
