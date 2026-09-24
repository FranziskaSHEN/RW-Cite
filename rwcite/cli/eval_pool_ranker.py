#!/usr/bin/env python3
"""Evaluate RR pool ranker: U, hits@k (gate), pool_recall (appendix).

Supports --shard-id/--num-shards for multi-GPU, and --merge-shards to combine.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]

from rwcite.runtime.deps import get_domain_cache, get_retriever_service  # noqa: E402
from rwcite.ranker.reference_recommend import normalize_arxiv_id  # noqa: E402
from rwcite.ranker.rr_pool_build import build_ranked_pool  # noqa: E402


def _load_rows(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("task") in (None, "", "bundle"):
                rows.append(r)
    return rows


def _source_ym(source_id: str) -> str:
    """arXiv id YYMM -> 'YYYY-MM'; empty when the id is not YYMM.NNNNN."""
    s = str(source_id or "")
    if len(s) >= 4 and s[:4].isdigit():
        return f"{2000 + int(s[:2])}-{s[2:4]}"
    return ""


def _by_month_hits(details: list[dict]) -> dict:
    """Per-month hits@10 / n_gold: the frontier-coverage read-out.

    The newest month of a `newest 10%` split sits at the corpus frontier, where the
    test sources' own references are not yet collected; that shows up here as a
    simultaneous drop in n_gold and hits@10.
    """
    groups: dict[str, list[dict]] = {}
    for d in details:
        ym = _source_ym(d.get("source_id") or "")
        if ym:
            groups.setdefault(ym, []).append(d)
    if len(groups) < 2:
        return {}
    return {
        ym: {
            "n": len(rows),
            "mean_hits_at_10": float(np.mean([float(r["hits_at_10"]) for r in rows])),
            "mean_n_gold": float(np.mean([float(r["n_gold"]) for r in rows])),
            "mean_universe_recall": float(
                np.mean([float(r["universe_recall"]) for r in rows])
            ),
            "mean_pool_over_u": float(np.mean([float(r["pool_over_u"]) for r in rows])),
        }
        for ym, rows in sorted(groups.items())
    }


def _bootstrap_gate(
    h10_vals: list[float],
    *,
    gate_thr: int,
    sub_n: int = 80,
    n_boot: int = 5000,
    seed: int = 12345,
) -> dict:
    """Resampling spread of the n-sub gate metric.

    The published gate reads one fixed 80-query subsample, whose standard error is
    large enough (~0.3 on EWM) to flip the verdict on its own. Report the spread so
    a pass is never claimed off a lucky draw.
    """
    n = len(h10_vals)
    if n < 2:
        return {}
    arr = np.asarray(h10_vals, dtype=np.float64)
    sd = float(arr.std(ddof=0))
    sub_n = min(int(sub_n), n)
    rng = np.random.default_rng(seed)
    # Without replacement: matches how the gate subsamples the fixed test set.
    means = np.array(
        [arr[rng.choice(n, size=sub_n, replace=False)].mean() for _ in range(int(n_boot))]
    )
    # Finite-population correction for the without-replacement subsample.
    fpc = ((n - sub_n) / (n - 1)) ** 0.5 if n > 1 else 0.0
    return {
        "n_sub": sub_n,
        "full_mean_hits_at_10": float(arr.mean()),
        "sd_hits_at_10": sd,
        "se_sub_mean": float(sd / sub_n**0.5 * fpc),
        "sub_mean_p2_5": float(np.percentile(means, 2.5)),
        "sub_mean_p50": float(np.percentile(means, 50.0)),
        "sub_mean_p97_5": float(np.percentile(means, 97.5)),
        "prob_sub_mean_ge_gate": float((means >= gate_thr).mean()),
        "n_boot": int(n_boot),
        "seed": int(seed),
    }


def _timing_agg(details: list[dict]) -> dict:
    """Aggregate per-sample ``timings`` dicts into mean/p50/p95/sum seconds."""
    keys: list[str] = []
    seen: set[str] = set()
    rows: list[dict[str, float]] = []
    for d in details:
        t = d.get("timings")
        if not isinstance(t, dict) or not t:
            continue
        rows.append({k: float(v) for k, v in t.items() if isinstance(v, (int, float))})
        for k in t:
            if k not in seen and isinstance(t.get(k), (int, float)):
                seen.add(k)
                keys.append(k)
    if not rows:
        return {}
    out: dict[str, dict[str, float]] = {}
    for k in keys:
        vals = np.asarray([r[k] for r in rows if k in r], dtype=np.float64)
        if vals.size == 0:
            continue
        out[k] = {
            "n": int(vals.size),
            "mean_s": float(vals.mean()),
            "p50_s": float(np.percentile(vals, 50)),
            "p95_s": float(np.percentile(vals, 95)),
            "sum_s": float(vals.sum()),
            "max_s": float(vals.max()),
        }
    # Preferred stage order for logs / reports
    preferred = [
        "universe_s",
        "query_encode_s",
        "cand_embed_s",
        "struct_shortlist_s",
        "citelink_cheap_s",
        "cand_text_s",
        "ce_score_s",
        "blend_rank_s",
        "rank_total_s",
        "r1a_rerank_s",
        "total_s",
    ]
    ordered: dict[str, dict[str, float]] = {}
    for k in preferred:
        if k in out:
            ordered[k] = out[k]
    for k, v in out.items():
        if k not in ordered:
            ordered[k] = v
    return ordered


def _summary_from_details(
    details: list[dict],
    *,
    pool_n: int,
    gate_thr: int,
    extra: dict | None = None,
) -> dict:
    u_vals = [float(d["universe_recall"]) for d in details]
    p_vals = [float(d["pool_recall@30"]) for d in details]
    ratios = [float(d["pool_over_u"]) for d in details]
    h10_vals = [float(d["hits_at_10"]) for d in details]
    h30_vals = [float(d["hits_at_30"]) for d in details]
    h50_vals = [
        float(d["hits_at_50"]) for d in details if "hits_at_50" in d
    ]
    mean_u = float(np.mean(u_vals)) if u_vals else 0.0
    mean_h10 = float(np.mean(h10_vals)) if h10_vals else 0.0
    mean_h30 = float(np.mean(h30_vals)) if h30_vals else 0.0
    mean_h50 = float(np.mean(h50_vals)) if h50_vals else None
    mean_p = float(np.mean(p_vals)) if p_vals else 0.0
    summary = {
        "n_samples": len(details),
        "n": pool_n,
        "hits_gate_threshold": gate_thr,
        "mean_universe_recall": mean_u,
        "mean_hits_at_10": mean_h10,
        "mean_hits_at_30": mean_h30,
        "mean_hits_at_50": mean_h50,
        "frac_hits_at_10_ge_gate": float(np.mean([h >= gate_thr for h in h10_vals]))
        if h10_vals
        else 0.0,
        "frac_hits_at_30_ge_gate": float(np.mean([h >= gate_thr for h in h30_vals]))
        if h30_vals
        else 0.0,
        "mean_pool_recall": mean_p,
        "mean_pool_over_u": float(np.mean(ratios)) if ratios else 0.0,
        "frac_universe_ge_0_80": float(np.mean([v >= 0.8 for v in u_vals]))
        if u_vals
        else 0.0,
        "frac_pool_ge_0_80": float(np.mean([v >= 0.8 for v in p_vals]))
        if p_vals
        else 0.0,
        "gate_universe_mean_ge_0_80": bool(mean_u >= 0.80) if u_vals else False,
        "gate_hits_at_10_mean_ge_8": bool(mean_h10 >= gate_thr) if h10_vals else False,
        "gate_hits_at_30_mean_ge_8": bool(mean_h30 >= gate_thr) if h30_vals else False,
        "gate_pool_mean_ge_0_80": bool(mean_p >= 0.80) if p_vals else False,
        "gate_pool_mean_ge_0_80_is_appendix": True,
    }
    boot = _bootstrap_gate(h10_vals, gate_thr=gate_thr)
    if boot:
        summary["bootstrap_hits_at_10"] = boot
    by_month = _by_month_hits(details)
    if by_month:
        summary["by_month_hits_at_10"] = by_month
    timing_stats = _timing_agg(details)
    if timing_stats:
        summary["timings"] = timing_stats
    if extra:
        summary.update(extra)
    return summary


def _merge_shards(
    *,
    out_dir: Path,
    tag: str,
    num_shards: int,
    out_path: Path,
    gate_thr: int,
) -> None:
    details: list[dict] = []
    pool_n = 30
    for i in range(num_shards):
        sp = out_dir / f"{tag}_shard{i}of{num_shards}.json"
        if not sp.is_file():
            raise SystemExit(f"missing shard: {sp}")
        data = json.loads(sp.read_text(encoding="utf-8"))
        details.extend(data.get("details") or [])
        pool_n = int((data.get("summary") or {}).get("n") or pool_n)
        print(f"merge {sp.name}: +{len(data.get('details') or [])}", flush=True)
    summary = _summary_from_details(
        details,
        pool_n=pool_n,
        gate_thr=gate_thr,
        extra={"merged_from": [f"{tag}_shard{i}of{num_shards}.json" for i in range(num_shards)]},
    )
    print(json.dumps(summary, indent=2), flush=True)
    if summary.get("timings"):
        print("== inference timings (seconds / query) ==", flush=True)
        for k, v in summary["timings"].items():
            print(
                f"  {k}: mean={v['mean_s']:.3f} p50={v['p50_s']:.3f} "
                f"p95={v['p95_s']:.3f} max={v['max_s']:.3f} sum={v['sum_s']:.1f}",
                flush=True,
            )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps({"summary": summary, "details": details}, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="")
    ap.add_argument("--test-jsonl", default="datasets/reference_recommend/test.jsonl")
    ap.add_argument("--ranker-mlp", default="models/rr-pool-ranker-v1/mlp.npz")
    ap.add_argument(
        "--ranker-ltr",
        default="",
        help="LightGBM LTR model.txt; if set, preferred over MLP (after CE)",
    )
    ap.add_argument(
        "--ranker-citelink",
        default="",
        help="Cite-link MLP model.npz; preferred over LTR/MLP (after CE)",
    )
    ap.add_argument(
        "--ranker-fusion",
        default="",
        help="Fusion LightGBM model.txt; needs --ranker-citelink; preferred over raw citelink",
    )
    ap.add_argument(
        "--ranker-sent",
        default="",
        help="Cite-sentence emb bank npz (v3e); preferred over fusion/citelink",
    )
    ap.add_argument(
        "--ranker-ce-sent",
        default="",
        help="v4 Cite-Sentence CE checkpoint dir; preferred over sent/fusion/CE",
    )
    ap.add_argument(
        "--ranker-ce",
        default="",
        help="CE checkpoint dir; if set, preferred over MLP (after ce-sent)",
    )
    ap.add_argument("--ce-prefilter", type=int, default=500)
    ap.add_argument("--ce-full-u", action="store_true", help="CE score full universe")
    ap.add_argument("--max-samples", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n", type=int, default=30, help="pool size for ranking (default 30)")
    ap.add_argument("--hits-gate", type=int, default=8, help="hits@k gate threshold")
    ap.add_argument("--paper-id", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--merge-shards", action="store_true")
    ap.add_argument("--tag", default="ltr_eval")
    ap.add_argument("--out-dir", default="datasets/rr_pool_ranker")
    args = ap.parse_args()

    if args.merge_shards:
        out = Path(args.out) if args.out else Path(args.out_dir) / f"{args.tag}_merged.json"
        _merge_shards(
            out_dir=Path(args.out_dir),
            tag=args.tag,
            num_shards=args.num_shards,
            out_path=out,
            gate_thr=int(args.hits_gate),
        )
        return

    if args.num_shards < 1 or args.shard_id < 0 or args.shard_id >= args.num_shards:
        raise SystemExit(f"bad shard: id={args.shard_id} num={args.num_shards}")

    if args.ranker_ce_sent:
        os.environ["RR_RANKER_CE_SENT_PATH"] = str(
            Path(args.ranker_ce_sent).resolve()
        )
    if args.ranker_ce:
        os.environ["RR_RANKER_CE_PATH"] = str(Path(args.ranker_ce).resolve())
        os.environ["RR_CE_PREFILTER"] = str(args.ce_prefilter)
        if args.ce_full_u:
            os.environ["RR_CE_FULL_U"] = "1"
    if args.ranker_sent:
        os.environ["RR_RANKER_SENT_PATH"] = str(Path(args.ranker_sent).resolve())
    if args.ranker_citelink:
        os.environ["RR_RANKER_CITELINK_PATH"] = str(
            Path(args.ranker_citelink).resolve()
        )
    if args.ranker_fusion:
        os.environ["RR_RANKER_FUSION_PATH"] = str(
            Path(args.ranker_fusion).resolve()
        )
        # fusion needs citelink; default to v3c if unset
        if not args.ranker_citelink:
            default_cl = Path("models/rr-pool-ranker-citelink-v3c/model.npz")
            if default_cl.is_file():
                os.environ["RR_RANKER_CITELINK_PATH"] = str(default_cl.resolve())
    if args.ranker_ltr:
        os.environ["RR_RANKER_LTR_PATH"] = str(Path(args.ranker_ltr).resolve())
    elif (
        Path(args.ranker_mlp).is_file()
        and not args.ranker_ce
        and not args.ranker_citelink
    ):
        os.environ["RR_RANKER_MLP_PATH"] = str(Path(args.ranker_mlp).resolve())
    os.environ.setdefault("RR_RANKER_FULL_EMB", "1")
    os.environ.setdefault("RR_RANKER_WITH_EMB", "1")

    graph = get_domain_cache().get(args.domain).graph
    retriever = get_retriever_service()

    def search_fn(query: str, limit: int):
        hits = retriever.search(
            scope="domain",
            domain_id=args.domain,
            query=query,
            offset=0,
            limit=limit,
            sort="relevance",
        )
        return list(hits.get("results") or [])

    rows = _load_rows(Path(args.test_jsonl))
    if args.paper_id:
        pid = normalize_arxiv_id(args.paper_id)
        rows = [
            r
            for r in rows
            if normalize_arxiv_id(str(r.get("source_id") or "")) == pid
        ]
    else:
        rng = random.Random(args.seed)
        rng.shuffle(rows)
        rows = rows[: args.max_samples]
    if args.num_shards > 1:
        rows = [r for i, r in enumerate(rows) if i % args.num_shards == args.shard_id]
        print(
            f"shard {args.shard_id}/{args.num_shards} n={len(rows)} "
            f"(of max_samples={args.max_samples})",
            flush=True,
        )

    pool_n = max(int(args.n), 30)
    gate_thr = int(args.hits_gate)
    details = []
    for s in rows:
        title = s.get("title") or ""
        abstract = s.get("abstract") or ""
        excl = normalize_arxiv_id(str(s.get("source_id") or ""))
        gold = [normalize_arxiv_id(g["id"]) for g in s.get("gold") or []]
        gold_set = set(gold)
        top, meta = build_ranked_pool(
            title,
            abstract,
            graph,
            search_fn=search_fn,
            exclude_id=excl,
            n=pool_n,
            retriever=retriever,
            domain_id=args.domain,
        )
        ranked_ids = [c["id"] for c in top]
        u_ids = set(meta.get("universe_ids") or [])
        u_rec = len(gold_set & u_ids) / len(gold_set) if gold_set else 0.0
        top50 = set(ranked_ids[:50])
        top30 = set(ranked_ids[:30])
        top10 = set(ranked_ids[:10])
        hits50 = len(gold_set & top50)
        hits30 = len(gold_set & top30)
        hits10 = len(gold_set & top10)
        p_rec = hits30 / len(gold_set) if gold_set else 0.0
        ratio = (p_rec / u_rec) if u_rec > 1e-9 else 0.0
        timings = meta.get("timings") if isinstance(meta.get("timings"), dict) else {}
        details.append(
            {
                "source_id": excl,
                "n_gold": len(gold_set),
                "universe_recall": u_rec,
                "pool_recall@30": p_rec,
                "pool_over_u": ratio,
                "hits_at_10": hits10,
                "hits_at_30": hits30,
                "hits_at_50": hits50,
                "universe_size": meta.get("universe_size"),
                "ranker": meta.get("ranker"),
                "gold_in_top10": sorted(gold_set & top10),
                "gold_in_top30": sorted(gold_set & top30),
                "gold_in_top50": sorted(gold_set & top50),
                "timings": timings,
            }
        )
        t_bits = ""
        if timings:
            # Compact per-query latency line for full-test logs
            parts = []
            for k in (
                "universe_s",
                "struct_shortlist_s",
                "cand_text_s",
                "ce_score_s",
                "rank_total_s",
                "total_s",
            ):
                if k in timings:
                    parts.append(f"{k.replace('_s','')}={float(timings[k]):.3f}s")
            if parts:
                t_bits = " | " + " ".join(parts)
        print(
            f"{excl}: U={u_rec:.2f} hits@10={hits10} hits@30={hits30} "
            f"hits@50={hits50} pool@30={p_rec:.2f} pool/U={ratio:.2f} "
            f"|U|={meta.get('universe_size')} ranker={meta.get('ranker')}"
            f"{t_bits}",
            flush=True,
        )

    summary = _summary_from_details(
        details,
        pool_n=pool_n,
        gate_thr=gate_thr,
        extra={
            "ranker_mlp": args.ranker_mlp,
            "ranker_ltr": args.ranker_ltr,
            "shard_id": args.shard_id,
            "num_shards": args.num_shards,
        },
    )
    print(json.dumps(summary, indent=2), flush=True)
    if summary.get("timings"):
        print("== inference timings (seconds / query) ==", flush=True)
        for k, v in summary["timings"].items():
            print(
                f"  {k}: mean={v['mean_s']:.3f} p50={v['p50_s']:.3f} "
                f"p95={v['p95_s']:.3f} max={v['max_s']:.3f} sum={v['sum_s']:.1f}",
                flush=True,
            )
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(
            json.dumps({"summary": summary, "details": details}, indent=2),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
