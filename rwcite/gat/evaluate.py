"""GAT MVP evaluate on frozen test windows + paired compare hooks."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rwcite.gat import BASELINE_FULL381_HITS_AT_10
from rwcite.gat import protocol as P
from rwcite.gat.compare import compare
from rwcite.gat.domain_layout import resolve_gat_domain
from rwcite.gat.model import GatMvpConfig, GatMvpModel, build_ablation_config
from rwcite.gat.ranking_metrics import mean_ir_fields, query_ir_metrics
from rwcite.gat.train import GatTrainPack, _load_ce_score_map, _zscore_1d, score_query_chunked


CEILING_HITS_AT_10 = 10.0
CEILING_HITS_AT_30 = 19.1


def evaluate(
    *,
    out_dir: Path,
    windows_path: Path,
    ckpt_path: Path,
    ablation: str,
    device: torch.device,
    top_n: int = 30,
    fair: bool = True,
    expand_hop: int = 0,
    l4_eval_k: int = 32,
    dump_scores: bool = False,
    ce_cache: Path | None = None,
    start_query: int = 0,
    max_queries: int = 0,
    gat_chunk: int = 0,
) -> dict[str, Any]:
    windows = np.load(windows_path, allow_pickle=True)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg_dict = ckpt.get("cfg") or {}
    if fair:
        # Independently trained ckpt: use its own cfg (no flag-flip on foreign weights).
        cfg = GatMvpConfig(
            **{k: v for k, v in cfg_dict.items() if k in GatMvpConfig.__dataclass_fields__}
        )
        if ablation and ablation.upper() == "E3" and cfg_dict.get("struct_only"):
            cfg.struct_only = True
        model = GatMvpModel(cfg)
        model.load_state_dict(ckpt["model"], strict=True)
    else:
        # Legacy probe: E4 weights + flag flip (residual attribution only).
        base_cfg = GatMvpConfig()
        model = GatMvpModel(base_cfg)
        model.load_state_dict(ckpt["model"], strict=True)
        if ablation:
            model.cfg = build_ablation_config(ablation)
    model.to(device)
    model.eval()

    pack = GatTrainPack(
        out_dir,
        use_foldout=True,
        exact_target_b=True,
        expand_hop=expand_hop,
        l4_train_k=l4_eval_k,
        l4_eval_k=l4_eval_k,
    )
    id_to_i = pack.id_to_i

    ce_by_qid = None
    if getattr(model.cfg, "use_ce_feat", False):
        ce_path = ce_cache or (out_dir / "ce_scores_test381_sent_s1.npz")
        if not Path(ce_path).is_file():
            raise FileNotFoundError(f"cefeat eval needs CE cache: {ce_path}")
        ce_by_qid = _load_ce_score_map(Path(ce_path), windows)

    n_all = int(windows["query_ids"].shape[0])
    start = max(0, int(start_query))
    if max_queries and max_queries > 0:
        end = min(n_all, start + int(max_queries))
    else:
        end = n_all
    indices = list(range(start, end))
    n = len(indices)
    per: list[dict[str, Any]] = []
    gold_in = 0.0
    t0 = time.perf_counter()
    rng = np.random.default_rng(0)
    chunk = int(gat_chunk) if gat_chunk and gat_chunk > 0 else int(P.GAT_CHUNK)

    with torch.no_grad():
        for bi, i in enumerate(indices):
            qid = str(windows["query_ids"][i])
            qg = id_to_i.get(qid)
            nc = int(windows["n_cands"][i])
            gold = windows["gold_mask"][i, :nc].astype(np.bool_)
            gold_in += float(gold.sum())
            if qg is None or nc < 1:
                per.append({"source_id": qid, "hits_at_10": 0, "hits_at_30": 0, "skip": True})
                continue
            ce_arr = None
            if ce_by_qid is not None:
                raw = ce_by_qid.get(qid)
                if raw is None:
                    ce_arr = np.zeros(nc, dtype=np.float32)
                else:
                    ce_arr = np.asarray(raw[:nc], dtype=np.float32)
            out = score_query_chunked(
                model,
                pack,
                query_global=qg,
                query_id=qid,
                cand_globals=windows["cand_idx"][i, :nc],
                n_cands=nc,
                phi_np=windows["phi_struct"][i, :nc],
                device=device,
                rng=rng,
                chunk_size=chunk,
                ce_arr=ce_arr,
                use_checkpoint=False,
            )
            sc = out["scores"].detach().cpu().numpy()
            ir = query_ir_metrics(sc, gold.astype(np.float32))
            order = sc.argsort()[::-1][:top_n]
            cand_globals = windows["cand_idx"][i, :nc]
            top_ids = [
                pack.ids[int(cand_globals[j])] for j in order if int(cand_globals[j]) >= 0
            ]
            row: dict[str, Any] = {
                "source_id": qid,
                "n_gold": int(windows["n_gold"][i]),
                "n_gold_in_window": int(gold.sum()),
                "top_ids": top_ids,
            }
            row.update(ir)
            if dump_scores:
                row["scores"] = [float(x) for x in sc.tolist()]
                row["cand_ids"] = [
                    pack.ids[int(cand_globals[j])]
                    for j in range(nc)
                    if int(cand_globals[j]) >= 0
                ]
            per.append(row)
            if (bi + 1) % 20 == 0:
                means_so_far = mean_ir_fields(per)
                print(
                    f"eval {bi+1}/{n} mean@10={means_so_far['mean_hits_at_10']:.3f} "
                    f"nDCG@10={means_so_far['mean_ndcg_at_10']:.3f}",
                    flush=True,
                )

    means = mean_ir_fields(per)
    mean10 = means["mean_hits_at_10"]
    mean30 = means["mean_hits_at_30"]
    summary = {
        "n": n,
        "mean_hits_at_10": mean10,
        "mean_hits_at_30": mean30,
        "mean_ndcg_at_10": means["mean_ndcg_at_10"],
        "mean_mrr_at_10": means["mean_mrr_at_10"],
        "mean_recall_at_30": means["mean_recall_at_30"],
        "mean_recall_at_100": means["mean_recall_at_100"],
        "mean_recall_at_400": means["mean_recall_at_400"],
        "recall_note": (
            "R@k = |Top-k ∩ G_window| / |G_window|; "
            "R@400 window-capped (pool=400); R@1000 not defined"
        ),
        "mean_gold_in_window": gold_in / max(n, 1),
        "pct_of_ceiling_at_10": 100.0 * mean10 / CEILING_HITS_AT_10,
        "pct_of_ceiling_at_30": 100.0 * mean30 / CEILING_HITS_AT_30,
        "ceiling_hits_at_10": CEILING_HITS_AT_10,
        "bar_hits_at_10": BASELINE_FULL381_HITS_AT_10,
        "ablation": ablation or "E4",
        "fair_eval": fair,
        "elapsed_s": float(time.perf_counter() - t0),
        "ckpt": str(ckpt_path),
        "windows": str(windows_path),
    }
    return {"summary": summary, "per_query": per}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="GAT MVP evaluate")
    ap.add_argument("--domain", default="ewm")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--windows", default=None)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--ablation", default="E4")
    ap.add_argument("--out-tag", default=P.GAT_MVP_EVAL_TAG)
    ap.add_argument("--device", default=None)
    ap.add_argument("--max-samples", type=int, default=381)
    ap.add_argument("--start-query", type=int, default=0, help="Slice start for sharded eval")
    ap.add_argument(
        "--max-queries",
        type=int,
        default=0,
        help="Max queries from start (0=all remaining); overrides max-samples when >0",
    )
    ap.add_argument("--window", type=int, default=P.STRUCT_WINDOW)
    ap.add_argument("--probe", action="store_true", help="flag-flip probe (not fair)")
    ap.add_argument("--expand-hop", type=int, default=0)
    ap.add_argument("--l4-eval-k", type=int, default=32)
    ap.add_argument(
        "--dump-scores",
        action="store_true",
        help="Include per-candidate scores in per_query (for CE blend)",
    )
    ap.add_argument("--compare-name", default=None, help="override compare_*.json name")
    ap.add_argument("--ce-cache", default=None, help="CE-sent scores for cefeat ckpt")
    ap.add_argument(
        "--gat-chunk",
        type=int,
        default=0,
        help=f"cand chunk size (0→GAT_CHUNK={P.GAT_CHUNK})",
    )
    args = ap.parse_args(argv)

    root = P.root_path()
    dpaths = resolve_gat_domain(args.domain, root)
    out_dir = Path(args.out_dir) if args.out_dir else root / dpaths.gat_mvp_dir
    if not out_dir.is_absolute():
        out_dir = root / out_dir
    n_test = 381
    meta = dpaths.split_dir / "split_meta.json"
    if meta.is_file():
        n_test = int(json.loads(meta.read_text(encoding="utf-8")).get("n_test_sources") or n_test)
    default_win = out_dir / f"windows_test{n_test}.npz"
    if not default_win.is_file() and args.domain == "ewm":
        legacy = out_dir / "windows_test381.npz"
        if legacy.is_file():
            default_win = legacy
    windows_path = Path(args.windows) if args.windows else default_win
    if not windows_path.is_absolute():
        windows_path = root / windows_path
    default_ckpt = out_dir / "ckpt_e4.pt"
    if not default_ckpt.is_file():
        default_ckpt = out_dir / "ckpt.pt"
    ckpt_path = Path(args.ckpt) if args.ckpt else default_ckpt
    if not ckpt_path.is_absolute():
        ckpt_path = root / ckpt_path
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    mq = args.max_queries if args.max_queries > 0 else args.max_samples
    result = evaluate(
        out_dir=out_dir,
        windows_path=windows_path,
        ckpt_path=ckpt_path,
        ablation=args.ablation,
        device=device,
        fair=not args.probe,
        expand_hop=args.expand_hop,
        l4_eval_k=args.l4_eval_k,
        dump_scores=args.dump_scores,
        ce_cache=Path(args.ce_cache) if args.ce_cache else None,
        start_query=args.start_query,
        max_queries=mq,
        gat_chunk=args.gat_chunk,
    )
    eval_dir = root / dpaths.eval_dir
    eval_dir.mkdir(parents=True, exist_ok=True)
    tag = args.out_tag
    if args.out_tag == P.GAT_MVP_EVAL_TAG:
        tag = f"{args.domain}_gat_mvp_test{n_test}"
    if args.ablation:
        tag = f"{tag}_{args.ablation.lower()}"
    out_path = eval_dir / f"{tag}_merged.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["summary"], indent=2), flush=True)
    print(f"wrote {out_path}", flush=True)

    if args.ablation.upper() in ("E4", "") and args.domain == "ewm":
        cmp = compare(
            root / P.BASELINE_FULL381_MERGED,
            out_path,
            bar=BASELINE_FULL381_HITS_AT_10,
        )
        cmp_name = args.compare_name or "compare_e4.json"
        cmp_path = out_dir / cmp_name
        cmp_path.write_text(json.dumps(cmp, indent=2), encoding="utf-8")
        if args.out_tag == P.GAT_MVP_EVAL_TAG and not args.dump_scores:
            canon = eval_dir / f"{P.GAT_MVP_EVAL_TAG}_merged.json"
            canon.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(cmp, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
