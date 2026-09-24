"""Error analysis for L0 CE-sent×GAT blend vs E4 and vs fair CE.

Answers: where L0 wins/loses; whether remaining miss is out-of-window vs ranking.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from rwcite.gat import BASELINE_FULL381_HITS_AT_10
from rwcite.gat import protocol as P


def _load_jsonl_gold(path: Path) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            qid = str(row.get("source_id") or "")
            # Prefer gold_ids even if empty; never fall through to huge full_gold
            if "gold_ids" in row and row["gold_ids"] is not None:
                g = row["gold_ids"]
            elif "gold" in row and isinstance(row["gold"], list):
                g = row["gold"]
            else:
                g = []
            out[qid] = {str(x) for x in g}
    return out


def _by_qid_gat(path: Path) -> dict[str, dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data.get("per_query") or data.get("details") or []
    return {str(r["source_id"]): r for r in rows if r.get("source_id")}


def _by_qid_ce(path: Path) -> dict[str, dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data.get("details") or data.get("per_query") or []
    return {str(r["source_id"]): r for r in rows if r.get("source_id")}


def _bucket(delta: float, *, eps: float = 0.5) -> str:
    """Per-query hits are ints; tie if equal."""
    if delta > eps:
        return "win"
    if delta < -eps:
        return "lose"
    return "tie"


def analyze(
    *,
    root: Path,
    l0_path: Path,
    e4_path: Path,
    ce_path: Path,
    ce_alpha0_path: Path | None,
    windows_path: Path,
    test_jsonl: Path,
    k: int = 10,
) -> dict[str, Any]:
    ids = json.loads(
        (windows_path.parent / "id_map.json").read_text(encoding="utf-8")
    )["ids"]
    windows = np.load(windows_path, allow_pickle=True)
    # Materialize once — per-element access on compressed npz is extremely slow.
    query_ids = [str(x) for x in windows["query_ids"].tolist()]
    cand_idx = np.asarray(windows["cand_idx"])
    gold_mask = np.asarray(windows["gold_mask"])
    n_cands_arr = np.asarray(windows["n_cands"])
    n_gold_arr_np = np.asarray(windows["n_gold"])
    gold_all = _load_jsonl_gold(test_jsonl)
    l0 = _by_qid_gat(l0_path)
    e4 = _by_qid_gat(e4_path)
    ce = _by_qid_ce(ce_path)
    ce0 = _by_qid_gat(ce_alpha0_path) if ce_alpha0_path and ce_alpha0_path.is_file() else {}

    # window membership per query
    win_cands: dict[str, set[str]] = {}
    win_gold_n: dict[str, int] = {}
    n_gold_arr: dict[str, int] = {}
    for i, qid in enumerate(query_ids):
        nc = int(n_cands_arr[i])
        cset = {
            ids[int(cand_idx[i, j])]
            for j in range(nc)
            if int(cand_idx[i, j]) >= 0
        }
        win_cands[qid] = cset
        win_gold_n[qid] = int(gold_mask[i, :nc].sum())
        n_gold_arr[qid] = int(n_gold_arr_np[i])

    vs_e4 = Counter()
    vs_ce = Counter()
    vs_ce0 = Counter()
    per: list[dict[str, Any]] = []

    sum_l0 = sum_e4 = sum_ce = sum_ce0 = 0.0
    sum_ceiling = 0.0  # min(k, n_gold_in_window)
    sum_oow = 0.0
    sum_gold = 0.0
    # miss decomposition for L0 (relative to all gold)
    miss_oow = 0.0
    miss_inwin_not_top = 0.0
    hit_l0 = 0.0
    # among in-window gold: recovered by L0/E4/CE top-k
    inwin_hit_l0 = inwin_hit_e4 = inwin_hit_ce = 0.0
    inwin_total = 0.0
    # complementarity: gold in L0 topk but not E4, etc.
    only_l0_vs_e4 = only_e4_vs_l0 = both = neither_inwin = 0.0

    qids = sorted(set(l0) & set(e4) & set(win_cands))
    for qid in qids:
        g_all = gold_all.get(qid) or set()
        # prefer jsonl gold; fall back to window n_gold if empty
        if not g_all and n_gold_arr.get(qid, 0) > 0:
            # cannot list ids — skip set ops for this query's OOW, keep hits compare
            g_all = set()
        wc = win_cands[qid]
        g_in = g_all & wc if g_all else set()
        g_out = g_all - wc if g_all else set()

        r0 = l0[qid]
        re = e4[qid]
        rc = ce.get(qid) or {}
        h0 = float(r0.get("hits_at_10") or 0)
        he = float(re.get("hits_at_10") or 0)
        hc = float(rc.get("hits_at_10") or 0)
        h_ce0 = float((ce0.get(qid) or {}).get("hits_at_10") or 0) if ce0 else float("nan")

        sum_l0 += h0
        sum_e4 += he
        sum_ce += hc
        if ce0:
            sum_ce0 += h_ce0
        n_in = win_gold_n[qid]
        sum_ceiling += float(min(k, n_in))
        sum_oow += float(len(g_out)) if g_all else float(max(0, n_gold_arr[qid] - n_in))
        sum_gold += float(len(g_all)) if g_all else float(n_gold_arr[qid])

        vs_e4[_bucket(h0 - he)] += 1
        vs_ce[_bucket(h0 - hc)] += 1
        if ce0:
            vs_ce0[_bucket(h0 - h_ce0)] += 1

        top0 = set((r0.get("top_ids") or [])[:k])
        tope = set((re.get("top_ids") or [])[:k])
        # CE baseline exposes gold_in_top10 rather than full top_ids
        topc_gold = set(rc.get("gold_in_top10") or [])
        if g_all:
            hit_set = top0 & g_all
            hit_l0 += float(len(hit_set))
            miss_oow += float(len(g_out))
            miss_inwin_not_top += float(len(g_in - top0))

            inwin_total += float(len(g_in))
            inwin_hit_l0 += float(len(top0 & g_in))
            inwin_hit_e4 += float(len(tope & g_in))
            inwin_hit_ce += float(len(topc_gold & g_in)) if topc_gold else float(
                len((set(rc.get("gold_in_top10") or [])) & g_in)
            )

            l0_only = (top0 & g_in) - tope
            e4_only = (tope & g_in) - top0
            both_set = (top0 & g_in) & tope
            neither = g_in - top0 - tope
            only_l0_vs_e4 += float(len(l0_only))
            only_e4_vs_l0 += float(len(e4_only))
            both += float(len(both_set))
            neither_inwin += float(len(neither))

        per.append(
            {
                "source_id": qid,
                "hits_l0": h0,
                "hits_e4": he,
                "hits_ce": hc,
                "hits_ce_alpha0": h_ce0 if ce0 else None,
                "delta_l0_minus_e4": h0 - he,
                "delta_l0_minus_ce": h0 - hc,
                "n_gold": int(len(g_all) if g_all else n_gold_arr[qid]),
                "n_gold_in_window": n_in,
                "n_gold_out_of_window": int(len(g_out)) if g_all else int(
                    max(0, n_gold_arr[qid] - n_in)
                ),
                "ceiling_at_10": int(min(k, n_in)),
                "bucket_vs_e4": _bucket(h0 - he),
                "bucket_vs_ce": _bucket(h0 - hc),
            }
        )

    n = max(len(qids), 1)
    # largest L0 gains / losses vs E4
    per_sorted = sorted(per, key=lambda r: r["delta_l0_minus_e4"], reverse=True)
    top_gains = per_sorted[:15]
    top_losses = list(reversed(per_sorted[-15:]))

    report: dict[str, Any] = {
        "n_queries": len(qids),
        "k": k,
        "means": {
            "l0": sum_l0 / n,
            "e4": sum_e4 / n,
            "ce_baseline": sum_ce / n,
            "ce_alpha0_window": (sum_ce0 / n) if ce0 else None,
            "ceiling_inwindow_at_10": sum_ceiling / n,
            "n_gold": sum_gold / n,
            "n_gold_out_of_window": sum_oow / n,
            "bar": BASELINE_FULL381_HITS_AT_10,
        },
        "headroom": {
            "l0_to_ceiling": (sum_ceiling - sum_l0) / n,
            "l0_pct_of_ceiling": 100.0 * sum_l0 / max(sum_ceiling, 1e-9),
            "oow_gold_mean": sum_oow / n,
            "note": (
                "ceiling = mean min(10, n_gold_in_window); OOW golds are unreachable "
                "by any window ranker (struct@400)."
            ),
        },
        "buckets": {
            "l0_vs_e4": dict(vs_e4),
            "l0_vs_ce_baseline": dict(vs_ce),
            "l0_vs_ce_alpha0": dict(vs_ce0) if ce0 else None,
        },
        "miss_decomposition_l0": {
            "hit_in_topk": hit_l0 / n,
            "miss_out_of_window": miss_oow / n,
            "miss_in_window_not_topk": miss_inwin_not_top / n,
            "frac_miss_that_is_oow": miss_oow / max(miss_oow + miss_inwin_not_top, 1e-9),
            "frac_miss_that_is_ranking": miss_inwin_not_top
            / max(miss_oow + miss_inwin_not_top, 1e-9),
            "note": (
                "Per-query averages over gold sets. OOW share high → lift needs shortlist; "
                "ranking share high → still room inside window."
            ),
        },
        "inwindow_gold_recovery": {
            "n_inwindow_gold_mean": inwin_total / n,
            "recovered_by_l0": inwin_hit_l0 / n,
            "recovered_by_e4": inwin_hit_e4 / n,
            "recovered_by_ce_top10_gold": inwin_hit_ce / n,
            "complementarity_l0_vs_e4": {
                "both": both / n,
                "l0_only": only_l0_vs_e4 / n,
                "e4_only": only_e4_vs_l0 / n,
                "neither": neither_inwin / n,
            },
        },
        "paths": {
            "l0": str(l0_path),
            "e4": str(e4_path),
            "ce": str(ce_path),
            "ce_alpha0": str(ce_alpha0_path) if ce_alpha0_path else None,
            "windows": str(windows_path),
            "test_jsonl": str(test_jsonl),
        },
        "top_l0_gains_vs_e4": top_gains,
        "top_l0_losses_vs_e4": top_losses,
        "recommendation": None,  # filled below
    }

    oow_frac = report["miss_decomposition_l0"]["frac_miss_that_is_oow"]
    rank_frac = report["miss_decomposition_l0"]["frac_miss_that_is_ranking"]
    l0_only = report["inwindow_gold_recovery"]["complementarity_l0_vs_e4"]["l0_only"]
    if oow_frac >= 0.55:
        rec = (
            "Dominant miss is out-of-window → next ROI is shortlist/universe quality "
            "(not more GAT capacity). Keep L0 as default fusion."
        )
    elif rank_frac >= 0.55:
        rec = (
            "Dominant miss is in-window ranking → query-side / stronger semantic "
            "interaction may help; avoid expand_hop / L1-style capacity digs."
        )
    else:
        rec = (
            "Miss split is mixed OOW + ranking → dual track: protect L0 blend, "
            "probe shortlist OR query encoding — not GAT width."
        )
    if l0_only >= 0.15:
        rec += f" L0 uniquely recovers ~{l0_only:.2f} in-window golds/query vs E4 (keep blend)."
    report["recommendation"] = rec
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="L0 error analysis vs E4 / CE")
    ap.add_argument("--l0", default=None, help="L0 default merged json (stage1 α=0.4)")
    ap.add_argument("--e4", default=None)
    ap.add_argument("--ce", default=None, help="fair CE baseline (2.0 HN full381 by default)")
    ap.add_argument(
        "--ce-alpha0",
        default=None,
        help="window-pure CE α=0 blend json (default: s1 a0p0)",
    )
    ap.add_argument("--windows", default=None)
    ap.add_argument("--test-jsonl", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    root = P.root_path()
    eval_dir = root / "embodied_world_model_retrieval/ranker/eval"
    out_dir = root / P.GAT_MVP_CKPT_DIR

    l0_path = Path(args.l0) if args.l0 else eval_dir / "ewm_gat_l0_default_test381_merged.json"
    if not l0_path.is_file():
        l0_path = eval_dir / "ewm_gat_ce_sent_blend_test381_a0p75_merged.json"
    e4_path = Path(args.e4) if args.e4 else eval_dir / "ewm_gat_mvp_test381_e4_fair_merged.json"
    if not e4_path.is_file():
        e4_path = eval_dir / "ewm_gat_mvp_test381_e4_merged.json"
    ce_path = Path(args.ce) if args.ce else root / P.BASELINE_FULL381_MERGED
    ce0_path = (
        Path(args.ce_alpha0)
        if args.ce_alpha0
        else eval_dir / "ewm_gat_ce_sent_s1_blend_test381_a0p0_merged.json"
    )
    if not args.ce_alpha0 and not ce0_path.is_file():
        # fallback HN-era α0 only if s1 artifact missing
        ce0_path = eval_dir / "ewm_gat_ce_sent_blend_test381_a0p0_merged.json"
    windows_path = Path(args.windows) if args.windows else out_dir / "windows_test381.npz"
    test_jsonl = (
        Path(args.test_jsonl)
        if args.test_jsonl
        else root / "embodied_world_model_retrieval/data/reference_recommend/test.jsonl"
    )
    out_path = Path(args.out) if args.out else out_dir / "l0_error_analysis.json"

    report = analyze(
        root=root,
        l0_path=l0_path,
        e4_path=e4_path,
        ce_path=ce_path,
        ce_alpha0_path=ce0_path if ce0_path.is_file() else None,
        windows_path=windows_path,
        test_jsonl=test_jsonl,
    )
    # drop bulky per-query list from default file? keep gains/losses only (already)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    # slim console summary
    slim = {
        "means": report["means"],
        "headroom": report["headroom"],
        "buckets": report["buckets"],
        "miss_decomposition_l0": report["miss_decomposition_l0"],
        "inwindow_gold_recovery": report["inwindow_gold_recovery"],
        "recommendation": report["recommendation"],
        "out": str(out_path),
    }
    print(json.dumps(slim, indent=2), flush=True)
    print(f"wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
