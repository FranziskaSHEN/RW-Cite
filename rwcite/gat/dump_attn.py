"""Dump GAT L4 neighbor attention (β) for heatmaps.

Writes compact npz under gat_mvp/:
  beta [Q,C,K], mask [Q,C,K], nbr_globals [Q,C,K], scores [Q,C],
  gold_mask, cand_idx, query_ids, n_cands, n_gold, max_beta, attn_entropy
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rwcite.gat import protocol as P
from rwcite.gat.model import GatMvpConfig, GatMvpModel
from rwcite.gat.train import GatTrainPack, _load_ce_score_map, score_query_chunked


def dump_attn(
    *,
    out_dir: Path,
    windows_path: Path,
    ckpt_path: Path,
    dump_path: Path,
    device: torch.device,
    l4_eval_k: int = 32,
    max_queries: int = 0,
    start_query: int = 0,
    gat_chunk: int = 0,
    ce_cache: Path | None = None,
) -> dict[str, Any]:
    windows = np.load(windows_path, allow_pickle=True)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg_dict = ckpt.get("cfg") or {}
    cfg = GatMvpConfig(
        **{k: v for k, v in cfg_dict.items() if k in GatMvpConfig.__dataclass_fields__}
    )
    model = GatMvpModel(cfg)
    model.load_state_dict(ckpt["model"], strict=True)
    model.to(device)
    model.eval()

    pack = GatTrainPack(
        out_dir,
        use_foldout=True,
        exact_target_b=True,
        expand_hop=0,
        l4_train_k=l4_eval_k,
        l4_eval_k=l4_eval_k,
    )
    id_to_i = pack.id_to_i

    ce_by_qid = None
    if getattr(model.cfg, "use_ce_feat", False):
        ce_path = ce_cache or (out_dir / "ce_scores_test381_sent_s1.npz")
        if not Path(ce_path).is_file():
            raise FileNotFoundError(f"cefeat dump needs CE cache: {ce_path}")
        ce_by_qid = _load_ce_score_map(Path(ce_path), windows)

    n_all = int(windows["query_ids"].shape[0])
    start = max(0, int(start_query))
    end = min(n_all, start + int(max_queries)) if max_queries and max_queries > 0 else n_all
    indices = list(range(start, end))
    n = len(indices)
    # pad C to window width (usually 400)
    c_max = int(max(int(windows["n_cands"][i]) for i in indices)) if indices else 0
    k = int(l4_eval_k)

    beta = np.zeros((n, c_max, k), dtype=np.float32)
    mask = np.zeros((n, c_max, k), dtype=np.bool_)
    nbr_g = np.full((n, c_max, k), -1, dtype=np.int64)
    scores = np.full((n, c_max), np.nan, dtype=np.float32)
    gold = np.zeros((n, c_max), dtype=np.bool_)
    cand_idx = np.full((n, c_max), -1, dtype=np.int64)
    n_cands = np.zeros(n, dtype=np.int32)
    n_gold = np.zeros(n, dtype=np.int32)
    query_ids = np.empty(n, dtype=object)
    max_beta = np.zeros(n, dtype=np.float32)
    attn_ent = np.zeros(n, dtype=np.float32)

    rng = np.random.default_rng(0)
    chunk = int(gat_chunk) if gat_chunk and gat_chunk > 0 else int(P.GAT_CHUNK)
    t0 = time.perf_counter()

    with torch.no_grad():
        for bi, i in enumerate(indices):
            qid = str(windows["query_ids"][i])
            query_ids[bi] = qid
            qg = id_to_i.get(qid)
            nc = int(windows["n_cands"][i])
            n_cands[bi] = nc
            n_gold[bi] = int(windows["n_gold"][i])
            gmask = windows["gold_mask"][i, :nc].astype(np.bool_)
            gold[bi, :nc] = gmask
            cands = windows["cand_idx"][i, :nc]
            cand_idx[bi, :nc] = cands
            if qg is None or nc < 1:
                continue
            ce_arr = None
            if ce_by_qid is not None:
                raw = ce_by_qid.get(qid)
                ce_arr = (
                    np.zeros(nc, dtype=np.float32)
                    if raw is None
                    else np.asarray(raw[:nc], dtype=np.float32)
                )
            out = score_query_chunked(
                model,
                pack,
                query_global=int(qg),
                query_id=qid,
                cand_globals=cands,
                n_cands=nc,
                phi_np=windows["phi_struct"][i, :nc],
                device=device,
                rng=rng,
                chunk_size=chunk,
                ce_arr=ce_arr,
                return_attn=True,
            )
            sc = out["scores"].detach().cpu().numpy().astype(np.float32)
            scores[bi, :nc] = sc
            b = out["l4_beta"].detach().cpu().numpy().astype(np.float32)
            m = out["l4_mask"].detach().cpu().numpy().astype(np.bool_)
            ng = out["l4_nbr_globals"].detach().cpu().numpy().astype(np.int64)
            kk = min(k, b.shape[1])
            beta[bi, :nc, :kk] = b[:, :kk]
            mask[bi, :nc, :kk] = m[:, :kk]
            nbr_g[bi, :nc, :kk] = ng[:, :kk]
            if "max_beta" in out:
                max_beta[bi] = float(out["max_beta"].detach().cpu())
            if "attn_entropy" in out:
                attn_ent[bi] = float(out["attn_entropy"].detach().cpu())
            if (bi + 1) % 20 == 0:
                print(f"attn dump {bi+1}/{n}", flush=True)

    dump_path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "ckpt": str(ckpt_path),
        "windows": str(windows_path),
        "l4_eval_k": k,
        "n_queries": n,
        "c_max": c_max,
        "start_query": start,
        "elapsed_s": float(time.perf_counter() - t0),
        "ablation_note": "L4 β only (not edge-channel α)",
    }
    np.savez_compressed(
        dump_path,
        beta=beta,
        mask=mask,
        nbr_globals=nbr_g,
        scores=scores,
        gold_mask=gold,
        cand_idx=cand_idx,
        query_ids=query_ids,
        n_cands=n_cands,
        n_gold=n_gold,
        max_beta=max_beta,
        attn_entropy=attn_ent,
        meta=json.dumps(meta),
    )
    print(json.dumps(meta, indent=2), flush=True)
    print(f"wrote {dump_path}", flush=True)
    return meta


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Dump GAT L4 attention β")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--windows", default=None)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--dump", default=None, help="output npz path")
    ap.add_argument("--device", default=None)
    ap.add_argument("--l4-eval-k", type=int, default=32)
    ap.add_argument("--start-query", type=int, default=0)
    ap.add_argument("--max-queries", type=int, default=0)
    ap.add_argument("--gat-chunk", type=int, default=0)
    ap.add_argument("--ce-cache", default=None)
    args = ap.parse_args(argv)

    root = P.root_path()
    out_dir = Path(args.out_dir) if args.out_dir else root / P.GAT_MVP_CKPT_DIR
    windows_path = Path(args.windows) if args.windows else out_dir / "windows_test381.npz"
    default_ckpt = out_dir / "ckpt_e4.pt"
    if not default_ckpt.is_file():
        default_ckpt = out_dir / "ckpt.pt"
    ckpt_path = Path(args.ckpt) if args.ckpt else default_ckpt
    dump_path = (
        Path(args.dump) if args.dump else out_dir / "attn_l4_test381.npz"
    )
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    dump_attn(
        out_dir=out_dir,
        windows_path=windows_path,
        ckpt_path=ckpt_path,
        dump_path=dump_path,
        device=device,
        l4_eval_k=args.l4_eval_k,
        max_queries=args.max_queries,
        start_query=args.start_query,
        gat_chunk=args.gat_chunk,
        ce_cache=Path(args.ce_cache) if args.ce_cache else None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
