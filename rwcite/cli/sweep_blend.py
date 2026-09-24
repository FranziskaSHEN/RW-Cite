#!/usr/bin/env python3
"""Offline sweep of CE⊕citelink blend β from a score dump.

Formula (matches rr_ranker_ce_sent.rank_universe_ce_sent):
  final = (1-β)·z(CE) + β·z(CL) + struct_blend·struct_rank_score
among CE-scored candidates; non-CE padded below (usually empty when score_m=window).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def _zscore(arr: np.ndarray) -> np.ndarray:
    if arr.size == 0:
        return arr
    mu = float(arr.mean())
    sd = float(arr.std())
    if sd < 1e-8:
        return np.zeros_like(arr)
    return (arr - mu) / sd


def _rank_with_blend(
    cands: list[dict],
    *,
    blend: float,
    struct_blend: float,
    pool_n: int,
) -> list[str]:
    n = len(cands)
    if n == 0:
        return []
    ce_scores = np.full(n, -1e9, dtype=np.float32)
    cl_scores = np.zeros(n, dtype=np.float32)
    ce_mask = np.zeros(n, dtype=bool)
    for i, c in enumerate(cands):
        cl_scores[i] = float(c.get("citelink_score") or 0.0)
        if c.get("ce_scored") and "ce_score" in c:
            ce_scores[i] = float(c["ce_score"])
            ce_mask[i] = True
    struct = np.array(
        [
            1.0 - float(int(c.get("struct_idx", i))) / float(max(n, 1))
            for i, c in enumerate(cands)
        ],
        dtype=np.float32,
    )
    final = np.zeros(n, dtype=np.float32)
    if ce_mask.any():
        zce = _zscore(ce_scores[ce_mask])
        zcl = _zscore(cl_scores[ce_mask]) if blend > 0 else np.zeros_like(zce)
        final[ce_mask] = (
            (1.0 - blend) * zce + blend * zcl + struct_blend * struct[ce_mask]
        )
        floor = float(final[ce_mask].min()) - 1.0
    else:
        floor = 0.0
    if (~ce_mask).any():
        # cheap pad: prefer higher citelink / earlier struct
        cheap = cl_scores[~ce_mask]
        if float(np.std(cheap)) < 1e-8:
            cheap = struct[~ce_mask]
        z_cheap = _zscore(cheap.astype(np.float32))
        final[~ce_mask] = floor - 1.0 + 0.01 * z_cheap
    order = np.argsort(-final)
    return [cands[int(i)]["id"] for i in order[:pool_n]]


def _hits(ranked: list[str], gold: set[str], k: int) -> int:
    return len(gold & set(ranked[:k]))


def _agg(details_hits: list[dict], *, gate: int = 8) -> dict:
    h10 = [float(d["hits_at_10"]) for d in details_hits]
    h30 = [float(d["hits_at_30"]) for d in details_hits]
    h50 = [float(d["hits_at_50"]) for d in details_hits]
    return {
        "n_samples": len(details_hits),
        "mean_hits_at_10": float(np.mean(h10)) if h10 else 0.0,
        "mean_hits_at_30": float(np.mean(h30)) if h30 else 0.0,
        "mean_hits_at_50": float(np.mean(h50)) if h50 else 0.0,
        "frac_hits_at_10_ge_gate": float(np.mean([h >= gate for h in h10]))
        if h10
        else 0.0,
        "frac_hits_at_30_ge_gate": float(np.mean([h >= gate for h in h30]))
        if h30
        else 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dump",
        default="",
    )
    ap.add_argument(
        "--out",
        default="",
    )
    ap.add_argument("--pool-n", type=int, default=50)
    ap.add_argument("--struct-blend", type=float, default=0.05)
    ap.add_argument(
        "--betas",
        default="0,0.05,0.10,0.15,0.20,0.25,0.30,0.40,0.50",
    )
    ap.add_argument("--gate", type=int, default=8)
    ap.add_argument("--tol", type=float, default=0.05, help="@10 noise tolerance vs β=0")
    args = ap.parse_args()

    dump = json.loads(Path(args.dump).read_text(encoding="utf-8"))
    details = dump.get("details") or []
    if not details:
        raise SystemExit(f"empty dump: {args.dump}")

    betas = [float(x) for x in args.betas.split(",") if x.strip()]
    sweeps: list[dict] = []
    per_beta_h10: dict[float, list[float]] = {}

    for beta in betas:
        rows = []
        for d in details:
            gold = {normalize for normalize in (d.get("gold") or [])}
            # ensure normalized (dump already stores normalized ids)
            gold = {str(g) for g in gold if g}
            ranked = _rank_with_blend(
                d.get("cands") or [],
                blend=float(beta),
                struct_blend=float(args.struct_blend),
                pool_n=int(args.pool_n),
            )
            row = {
                "source_id": d.get("source_id"),
                "n_gold": d.get("n_gold"),
                "hits_at_10": _hits(ranked, gold, 10),
                "hits_at_30": _hits(ranked, gold, 30),
                "hits_at_50": _hits(ranked, gold, 50),
            }
            rows.append(row)
        agg = _agg(rows, gate=int(args.gate))
        per_beta_h10[beta] = [float(r["hits_at_10"]) for r in rows]
        sweeps.append({"blend": beta, **agg})

    base = next((s for s in sweeps if abs(s["blend"] - 0.0) < 1e-12), sweeps[0])
    base_h10 = float(base["mean_hits_at_10"])
    base_per = per_beta_h10.get(float(base["blend"])) or []

    for s in sweeps:
        s["delta_hits_at_10_vs_b0"] = float(s["mean_hits_at_10"] - base_h10)
        s["delta_hits_at_30_vs_b0"] = float(
            s["mean_hits_at_30"] - float(base["mean_hits_at_30"])
        )
        s["delta_hits_at_50_vs_b0"] = float(
            s["mean_hits_at_50"] - float(base["mean_hits_at_50"])
        )
        cur = per_beta_h10[float(s["blend"])]
        if base_per and len(cur) == len(base_per):
            help_n = sum(1 for a, b in zip(cur, base_per) if a > b)
            hurt_n = sum(1 for a, b in zip(cur, base_per) if a < b)
            same_n = sum(1 for a, b in zip(cur, base_per) if a == b)
            s["help_hurt_same_at_10"] = {
                "help": help_n,
                "hurt": hurt_n,
                "same": same_n,
            }

    best = max(sweeps, key=lambda s: (s["mean_hits_at_10"], -s["blend"]))
    best_h10 = float(best["mean_hits_at_10"])
    deploy_blend = float(best["blend"])
    keep_pure = best_h10 <= base_h10 + float(args.tol)
    if keep_pure:
        deploy_blend = 0.0
        decision = (
            f"keep BLEND=0 (best@10={best_h10:.4f} at β={best['blend']} "
            f"≤ β0={base_h10:.4f}+tol={args.tol})"
        )
    else:
        decision = (
            f"use BLEND={deploy_blend} (best@10={best_h10:.4f}, "
            f"Δvs0={best_h10 - base_h10:+.4f})"
        )

    report = {
        "protocol": "offline_ce_citelink_blend_sweep",
        "dump": str(args.dump),
        "domain": dump.get("domain"),
        "ce_path": dump.get("ce_path"),
        "citelink_path": dump.get("citelink_path"),
        "n_samples": len(details),
        "pool_n": int(args.pool_n),
        "struct_blend": float(args.struct_blend),
        "tol_hits_at_10": float(args.tol),
        "reference_online": {
            "hn_pure_ce_hits_at_10": 6.5,
            "hn_blend_0_25_hits_at_10": 6.21,
            "note": "offline β=0 / 0.25 should match within noise",
        },
        "sweeps": sweeps,
        "best_blend_by_hits_at_10": float(best["blend"]),
        "best_hits_at_10": best_h10,
        "deploy_blend": deploy_blend,
        "keep_pure_ce": keep_pure,
        "decision": decision,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps({k: report[k] for k in (
        "n_samples",
        "best_blend_by_hits_at_10",
        "best_hits_at_10",
        "deploy_blend",
        "keep_pure_ce",
        "decision",
    )}, indent=2), flush=True)
    print("--- sweeps ---", flush=True)
    for s in sweeps:
        print(
            f"β={s['blend']:.2f}: @10={s['mean_hits_at_10']:.4f} "
            f"(Δ{s['delta_hits_at_10_vs_b0']:+.3f}) "
            f"@30={s['mean_hits_at_30']:.4f} @50={s['mean_hits_at_50']:.4f} "
            f"frac≥8={s['frac_hits_at_10_ge_gate']:.3f}",
            flush=True,
        )
    print(f"wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
