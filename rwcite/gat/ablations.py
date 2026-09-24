"""Fair E/F ablations: independently trained ckpts (续冲 R3)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rwcite.gat import BASELINE_FULL381_HITS_AT_10
from rwcite.gat import protocol as P
from rwcite.gat.evaluate import CEILING_HITS_AT_10, evaluate
from rwcite.gat.train import GatTrainPack


def eval_e2_cosine(*, out_dir: Path, windows_path: Path) -> dict[str, Any]:
    pack = GatTrainPack(out_dir)
    windows = np.load(windows_path, allow_pickle=True)
    id_to_i = pack.id_to_i
    n = int(windows["query_ids"].shape[0])
    hits10 = hits30 = 0.0
    per = []
    X = pack.X
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    for i in range(n):
        qid = str(windows["query_ids"][i])
        qg = id_to_i.get(qid)
        nc = int(windows["n_cands"][i])
        gold = windows["gold_mask"][i, :nc].astype(np.float32)
        if qg is None or nc < 1:
            per.append({"source_id": qid, "hits_at_10": 0})
            continue
        cidx = windows["cand_idx"][i, :nc].astype(np.int64)
        sc = Xn[cidx] @ Xn[qg]
        top10 = sc.argsort()[::-1][:10]
        top30 = sc.argsort()[::-1][:30]
        h10 = float(gold[top10].sum())
        h30 = float(gold[top30].sum())
        hits10 += h10
        hits30 += h30
        per.append({"source_id": qid, "hits_at_10": h10, "hits_at_30": h30})
    summary = {
        "n": n,
        "mean_hits_at_10": hits10 / max(n, 1),
        "mean_hits_at_30": hits30 / max(n, 1),
        "pct_of_ceiling_at_10": 100.0 * (hits10 / max(n, 1)) / CEILING_HITS_AT_10,
        "ablation": "E2",
        "fair_eval": True,
        "bar_hits_at_10": BASELINE_FULL381_HITS_AT_10,
    }
    return {"summary": summary, "per_query": per}


def _ckpt_for(out_dir: Path, name: str, fallback: Path) -> Path:
    cand = out_dir / f"ckpt_{name.lower()}.pt"
    return cand if cand.is_file() else fallback


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="GAT fair ablations (independent ckpts)")
    ap.add_argument("--runs", default="E1,E2,E3,E4,E5,F5")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--windows", default=None)
    ap.add_argument("--ckpt-e4", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--probe", action="store_true", help="legacy flag-flip (not for route)")
    ap.add_argument("--expand-hop", type=int, default=0)
    args = ap.parse_args(argv)

    root = P.root_path()
    out_dir = Path(args.out_dir) if args.out_dir else root / P.GAT_MVP_CKPT_DIR
    windows_path = Path(args.windows) if args.windows else out_dir / "windows_test381.npz"
    e4_ckpt = Path(args.ckpt_e4) if args.ckpt_e4 else _ckpt_for(
        out_dir, "e4", out_dir / "ckpt.pt"
    )
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    eval_dir = root / "embodied_world_model_retrieval/ranker/eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    route: dict[str, Any] = {
        "runs": {},
        "bar": BASELINE_FULL381_HITS_AT_10,
        "fair": not args.probe,
        "protocol": "independent_ckpts" if not args.probe else "flagflip_probe",
    }

    for name in [x.strip() for x in args.runs.split(",") if x.strip()]:
        print(f"== ablation {name} fair={not args.probe} ==", flush=True)
        if name.upper() == "E2":
            result = eval_e2_cosine(out_dir=out_dir, windows_path=windows_path)
        elif name.upper() == "E1":
            result = {
                "summary": {
                    "ablation": "E1",
                    "mean_hits_at_10": BASELINE_FULL381_HITS_AT_10,
                    "note": "2.0 CE baseline (not re-run)",
                    "fair_eval": True,
                }
            }
        else:
            ckpt = _ckpt_for(out_dir, name, e4_ckpt)
            if not args.probe and name.upper() != "E4" and ckpt == e4_ckpt:
                print(
                    f"WARN: missing ckpt_{name.lower()}.pt — refusing fair eval "
                    f"(would fall back to E4 weights). Skip {name}.",
                    flush=True,
                )
                route["runs"][name.upper()] = {
                    "skipped": True,
                    "reason": f"missing ckpt_{name.lower()}.pt",
                }
                continue
            result = evaluate(
                out_dir=out_dir,
                windows_path=windows_path,
                ckpt_path=ckpt,
                ablation=name,
                device=device,
                fair=not args.probe,
                expand_hop=args.expand_hop,
            )
        tag = f"ewm_gat_mvp_test381_{name.lower()}"
        if not args.probe:
            tag = f"{tag}_fair"
        path = eval_dir / f"{tag}_merged.json"
        path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        s = result["summary"]
        route["runs"][name.upper()] = {
            "mean_hits_at_10": s.get("mean_hits_at_10"),
            "mean_hits_at_30": s.get("mean_hits_at_30"),
            "pct_of_ceiling_at_10": s.get("pct_of_ceiling_at_10"),
            "path": str(path),
            "fair_eval": s.get("fair_eval", not args.probe),
        }
        print(json.dumps(route["runs"][name.upper()], indent=2), flush=True)

    e4 = (route["runs"].get("E4") or {}).get("mean_hits_at_10")
    e1 = BASELINE_FULL381_HITS_AT_10
    f5 = (route["runs"].get("F5") or {}).get("mean_hits_at_10")
    e3 = (route["runs"].get("E3") or {}).get("mean_hits_at_10")
    e2 = (route["runs"].get("E2") or {}).get("mean_hits_at_10")
    e5 = (route["runs"].get("E5") or {}).get("mean_hits_at_10")

    decision = "insufficient_runs"
    note = ""
    if args.probe:
        decision = "probe_only_not_architecture_verdict"
        note = "Flag-flip probes must not drive drop-GNN final decision."
    elif e4 is not None and not any(
        (route["runs"].get(x) or {}).get("skipped") for x in ("F5", "E3", "E5")
    ):
        if e4 > e1 and f5 is not None and f5 < e4 - 0.1:
            decision = "promote_gat_l4_holds"
            note = "Fair: E4>E1 and independent F5 hurts → L4 hypothesis holds"
        elif e3 is not None and abs(e3 - e4) < 0.15:
            decision = "keep_struct_residual_drop_GNN"
            note = "Fair: independent E3≈E4 → shallow struct residual enough"
        elif (e2 is not None and abs(e4 - e2) < 0.15) or (
            e5 is not None and abs(e4 - e5) < 0.15
        ):
            decision = "plan_b_or_simplify_gcn"
            note = "Fair: E4≈E2/E5 → graph/attention adds little"
        elif e4 <= e1:
            decision = "keep_2.0_CE"
            note = "Fair: E4≤E1 after B-clean"
        else:
            decision = "mixed_continue_r4"
            note = "Fair: E4 above bar but F5 inconclusive → R4 enrich subgraph / press residual"
    elif e4 is not None:
        decision = "partial_runs_continue"
        note = "Some independent ckpts missing; do not finalize drop-GNN"

    route["decision"] = decision
    route["note"] = note
    route_path = out_dir / (
        "route_decision_probe.json" if args.probe else "route_decision.json"
    )
    route_path.write_text(json.dumps(route, indent=2), encoding="utf-8")
    print(json.dumps(route, indent=2), flush=True)
    print(f"wrote {route_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
