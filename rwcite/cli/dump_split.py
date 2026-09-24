#!/usr/bin/env python3
"""Dump frozen domain train/test split from a GEXF (test_admit_v1 protocol).

Eligible pool (defaults on):
  A) sources from retrieval_nodes only
  B) outdeg >= elig_k (default 10)
  D) cite-sentence coverage >= min_sentence_coverage (default 0.95)
  F) gold-count (outdeg) within [gold_quantile_lo, gold_quantile_hi]
     (default 0.10–0.99)
  E) newest frontier_margin_months held corpus-only (default 1)

Then T1: newest ``test_frac`` by arXiv ym → test.

Legacy time_elig10_v2 can be reproduced by turning filters off
(``--frontier-margin-months 0 --min-sentence-coverage 0
--gold-quantile-lo 0 --gold-quantile-hi 1 --no-sources-from-retrieval-nodes``).

Outputs under --out-dir (creates directory).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics as st
from datetime import datetime, timezone
from pathlib import Path

import networkx as nx

from rwcite.graph.domain_paths import DEFAULT_SPLIT_ID

ROOT = Path(__file__).resolve().parents[2]

# Defaults for test_admit_v1 (A/B/D/F/E + T1).
DEFAULT_FRONTIER_MARGIN_MONTHS = 1
DEFAULT_MIN_SENTENCE_COVERAGE = 0.95
DEFAULT_GOLD_QUANTILE_LO = 0.10
DEFAULT_GOLD_QUANTILE_HI = 0.99


def _sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def arxiv_ym(pid: str) -> tuple[int, int] | None:
    """Return (year, month) from arXiv id like YYMM.xxxxx or old archive/NNNNN."""
    s = str(pid).strip()
    if "/" in s:
        return None
    head = s.split(".")[0]
    if len(head) == 4 and head.isdigit():
        yy, mm = int(head[:2]), int(head[2:])
        if not (1 <= mm <= 12):
            return None
        year = 2000 + yy if yy < 80 else 1900 + yy
        return year, mm
    return None


def arxiv_ym_key(pid: str) -> tuple[int, int, str]:
    ym = arxiv_ym(pid)
    if ym is None:
        return (0, 0, str(pid))
    return (ym[0], ym[1], str(pid))


def sentence_coverage(g: nx.DiGraph, nid: str) -> float:
    od = g.out_degree(nid)
    if od <= 0:
        return 0.0
    n_sent = 0
    for _, _, data in g.out_edges(nid, data=True):
        sent = str(data.get("sentence") or data.get("label") or "").strip()
        if sent:
            n_sent += 1
    return n_sent / float(od)


def _quantile(sorted_vals: list[int | float], q: float) -> float:
    """Linear-interpolated quantile; ``sorted_vals`` must be non-empty and sorted."""
    if q <= 0:
        return float(sorted_vals[0])
    if q >= 1:
        return float(sorted_vals[-1])
    idx = (len(sorted_vals) - 1) * q
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return float(sorted_vals[lo])
    frac = idx - lo
    return float(sorted_vals[lo]) * (1.0 - frac) + float(sorted_vals[hi]) * frac


def elig_stats(g: nx.DiGraph, ids: list[str], ret_pos: dict[str, int] | None) -> dict:
    if not ids:
        return {"n": 0}
    od = [g.out_degree(n) for n in ids]
    ind = [g.in_degree(n) for n in ids]
    years = [arxiv_ym(n)[0] for n in ids if arxiv_ym(n)]
    ranks = [ret_pos[n] for n in ids if ret_pos and n in ret_pos]
    out: dict = {
        "n": len(ids),
        "outdeg_mean": round(st.mean(od), 2),
        "outdeg_median": float(st.median(od)),
        "indeg_mean": round(st.mean(ind), 2),
        "indeg_median": float(st.median(ind)),
    }
    if years:
        out["year_mean"] = round(st.mean(years), 2)
        out["year_median"] = float(st.median(years))
    if ranks:
        out["ret_rank_mean"] = round(st.mean(ranks), 1)
        out["ret_rank_median"] = float(st.median(ranks))
        out["in_retrieval"] = len(ranks)
        out["in_retrieval_frac"] = round(len(ranks) / len(ids), 4)
    return out


def dump_time_elig10(
    g: nx.DiGraph,
    out_dir: Path,
    *,
    gexf: Path,
    gexf_hash: str,
    elig_k: int,
    test_frac: float,
    ret_pos: dict[str, int] | None,
    split_id: str = DEFAULT_SPLIT_ID,
    frontier_margin_months: int = DEFAULT_FRONTIER_MARGIN_MONTHS,
    allowed_sources: set[str] | None = None,
    min_sentence_coverage: float = DEFAULT_MIN_SENTENCE_COVERAGE,
    gold_quantile_lo: float = DEFAULT_GOLD_QUANTILE_LO,
    gold_quantile_hi: float = DEFAULT_GOLD_QUANTILE_HI,
) -> dict:
    elig = [n for n in g.nodes() if g.out_degree(n) >= elig_k]
    n_elig_before_allow = len(elig)
    if allowed_sources:
        # Graph enrichment gives cited papers out-edges too, which would quietly
        # promote them to split sources. The domain's sources are its retrieved
        # corpus; enrichment is meant to add signal, not new queries.
        elig = [n for n in elig if str(n) in allowed_sources]
    n_after_allowlist = len(elig)

    n_drop_sentence_coverage = 0
    if min_sentence_coverage > 0:
        kept: list[str] = []
        for n in elig:
            if sentence_coverage(g, n) >= min_sentence_coverage:
                kept.append(n)
            else:
                n_drop_sentence_coverage += 1
        elig = kept

    n_drop_gold_quantile = 0
    gold_outdeg_lo: float | None = None
    gold_outdeg_hi: float | None = None
    if elig and not (gold_quantile_lo <= 0 and gold_quantile_hi >= 1):
        if not (0.0 <= gold_quantile_lo <= gold_quantile_hi <= 1.0):
            raise SystemExit(
                f"invalid gold quantiles: lo={gold_quantile_lo} hi={gold_quantile_hi}"
            )
        ods = sorted(g.out_degree(n) for n in elig)
        gold_outdeg_lo = _quantile(ods, gold_quantile_lo)
        gold_outdeg_hi = _quantile(ods, gold_quantile_hi)
        kept = []
        for n in elig:
            od = g.out_degree(n)
            if gold_outdeg_lo <= od <= gold_outdeg_hi:
                kept.append(n)
            else:
                n_drop_gold_quantile += 1
        elig = kept

    elig_sorted = sorted(elig, key=arxiv_ym_key)
    frontier_sources: list[str] = []
    if frontier_margin_months > 0:
        months = sorted({arxiv_ym(n) for n in elig_sorted if arxiv_ym(n)})
        held = set(months[-frontier_margin_months:])
        frontier_sources = [n for n in elig_sorted if arxiv_ym(n) in held]
        elig_sorted = [n for n in elig_sorted if arxiv_ym(n) not in held]
    if not elig_sorted:
        raise SystemExit(
            "no eligible sources after admit filters "
            f"(before_allow={n_elig_before_allow}, after_allow={n_after_allowlist}, "
            f"drop_sent={n_drop_sentence_coverage}, "
            f"drop_gold_q={n_drop_gold_quantile}, frontier={len(frontier_sources)})"
        )
    n_test = max(1, int(math.ceil(len(elig_sorted) * test_frac)))
    test_sources = elig_sorted[-n_test:]
    train_sources = elig_sorted[:-n_test]
    cut_pid = test_sources[0]
    cut_ym = arxiv_ym(cut_pid)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "train_sources.json").write_text(
        json.dumps(train_sources, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (out_dir / "test_sources.json").write_text(
        json.dumps(test_sources, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (out_dir / "train_nodes.json").write_text(
        json.dumps(train_sources, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (out_dir / "test_nodes.json").write_text(
        json.dumps(test_sources, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    stats = {
        "elig_k": elig_k,
        "test_frac": test_frac,
        "frontier_margin_months": frontier_margin_months,
        "min_sentence_coverage": min_sentence_coverage,
        "gold_quantile_lo": gold_quantile_lo,
        "gold_quantile_hi": gold_quantile_hi,
        "gold_outdeg_lo": gold_outdeg_lo,
        "gold_outdeg_hi": gold_outdeg_hi,
        "n_drop_sentence_coverage": n_drop_sentence_coverage,
        "n_drop_gold_quantile": n_drop_gold_quantile,
        "all_elig": elig_stats(g, elig_sorted, ret_pos),
        "test_sources": elig_stats(g, test_sources, ret_pos),
        "train_sources": elig_stats(g, train_sources, ret_pos),
    }
    if frontier_sources:
        (out_dir / "frontier_sources.json").write_text(
            json.dumps(frontier_sources, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        stats["frontier_sources"] = elig_stats(g, frontier_sources, ret_pos)
    (out_dir / "elig10_stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    rule = f"elig outdeg>={elig_k}; newest {test_frac:.0%} by arxiv_ym → test"
    if min_sentence_coverage > 0:
        rule += f"; sentence_coverage>={min_sentence_coverage:g}"
    if not (gold_quantile_lo <= 0 and gold_quantile_hi >= 1):
        rule += (
            f"; gold_outdeg q[{gold_quantile_lo:g},{gold_quantile_hi:g}]"
            f"→[{gold_outdeg_lo:g},{gold_outdeg_hi:g}]"
            if gold_outdeg_lo is not None and gold_outdeg_hi is not None
            else f"; gold_outdeg q[{gold_quantile_lo:g},{gold_quantile_hi:g}]"
        )
    if frontier_margin_months > 0:
        rule += (
            f"; newest {frontier_margin_months} month(s) corpus-only "
            "(frontier margin)"
        )
    if allowed_sources:
        rule += "; sources-from-retrieval-nodes"
    meta = {
        "split_id": split_id,
        "split_rule": rule,
        "mode": "time_elig10",
        "frontier_margin_months": frontier_margin_months,
        "min_sentence_coverage": min_sentence_coverage,
        "gold_quantile_lo": gold_quantile_lo,
        "gold_quantile_hi": gold_quantile_hi,
        "gold_outdeg_lo": gold_outdeg_lo,
        "gold_outdeg_hi": gold_outdeg_hi,
        "n_drop_sentence_coverage": n_drop_sentence_coverage,
        "n_drop_gold_quantile": n_drop_gold_quantile,
        "n_frontier_sources": len(frontier_sources),
        "n_elig_before_source_allowlist": n_elig_before_allow,
        "n_elig_after_source_allowlist": n_after_allowlist,
        "source_allowlist": bool(allowed_sources),
        "gexf": str(gexf.relative_to(ROOT)) if gexf.is_relative_to(ROOT) else str(gexf),
        "gexf_sha256": gexf_hash,
        "elig_k": elig_k,
        "test_frac": test_frac,
        "n_elig": len(elig_sorted),
        "n_train_sources": len(train_sources),
        "n_test_sources": len(test_sources),
        "cutoff_source": cut_pid,
        "cutoff_ym": {"year": cut_ym[0], "month": cut_ym[1]} if cut_ym else None,
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    (out_dir / "split_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return meta


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--gexf",
        default="superconducting_quantum_computing_retrieval/description/test_graph_rr.gexf",
    )
    ap.add_argument(
        "--out-dir",
        required=True,
        help="e.g. <domain>_retrieval/data/splits/",
    )
    ap.add_argument("--elig-k", type=int, default=10)
    ap.add_argument("--test-frac", type=float, default=0.10)
    ap.add_argument(
        "--split-id",
        default=DEFAULT_SPLIT_ID,
        help=f"recorded in split_meta (default {DEFAULT_SPLIT_ID})",
    )
    ap.add_argument(
        "--frontier-margin-months",
        type=int,
        default=DEFAULT_FRONTIER_MARGIN_MONTHS,
        help="hold the newest N arXiv months out of train and test (corpus-only); "
        "0 disables (legacy)",
    )
    ap.add_argument(
        "--min-sentence-coverage",
        type=float,
        default=DEFAULT_MIN_SENTENCE_COVERAGE,
        help="require non-empty cite-sentence fraction of out-edges >= N; 0 disables",
    )
    ap.add_argument(
        "--gold-quantile-lo",
        type=float,
        default=DEFAULT_GOLD_QUANTILE_LO,
        help="drop sources with outdeg below this quantile of the pool (F); "
        "use 0 with --gold-quantile-hi 1 to disable",
    )
    ap.add_argument(
        "--gold-quantile-hi",
        type=float,
        default=DEFAULT_GOLD_QUANTILE_HI,
        help="drop sources with outdeg above this quantile of the pool (F)",
    )
    ap.add_argument(
        "--retrieval-nodes",
        default="superconducting_quantum_computing_retrieval/retrieval_nodes.json",
        help="retrieval corpus; required when sources-from-retrieval-nodes is on",
    )
    ap.add_argument(
        "--sources-from-retrieval-nodes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="only retrieved corpus papers may be split sources (default: on)",
    )
    args = ap.parse_args()

    gexf = Path(args.gexf)
    if not gexf.is_absolute():
        gexf = ROOT / gexf
    if not gexf.exists():
        raise SystemExit(f"missing gexf: {gexf}")

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir

    ret_pos: dict[str, int] | None = None
    rp = Path(args.retrieval_nodes)
    if not rp.is_absolute():
        rp = ROOT / rp
    if rp.exists():
        keys = list(json.loads(rp.read_text(encoding="utf-8")).keys())
        ret_pos = {k: i for i, k in enumerate(keys)}
    allowed: set[str] | None = None
    if args.sources_from_retrieval_nodes:
        if ret_pos is None:
            raise SystemExit(f"missing retrieval nodes: {rp}")
        allowed = set(ret_pos)

    print(f"loading {gexf} ...")
    g = nx.read_gexf(str(gexf))
    if not g.is_directed():
        g = g.to_directed()
    print(f"hashing {gexf.name} ...")
    ghash = _sha256_file(gexf)

    meta = dump_time_elig10(
        g,
        out_dir,
        gexf=gexf,
        gexf_hash=ghash,
        elig_k=args.elig_k,
        test_frac=args.test_frac,
        ret_pos=ret_pos,
        split_id=args.split_id,
        frontier_margin_months=args.frontier_margin_months,
        allowed_sources=allowed,
        min_sentence_coverage=args.min_sentence_coverage,
        gold_quantile_lo=args.gold_quantile_lo,
        gold_quantile_hi=args.gold_quantile_hi,
    )

    print(f"wrote {out_dir}")
    print(json.dumps(meta, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
