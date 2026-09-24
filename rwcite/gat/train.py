"""Listwise GAT training over structural windows with fold and edge masks."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from rwcite.gat.domain_layout import resolve_gat_domain
from rwcite.gat import protocol as P
from rwcite.gat.fold_mask import drop_target_edges
from rwcite.gat.model import GatMvpConfig, GatMvpModel, build_ablation_config
from rwcite.gat.preprocess import load_csr


def _ddp_world_size() -> int:
    import os

    return int(os.environ.get("WORLD_SIZE") or "1")


def _ddp_rank() -> int:
    import os

    return int(os.environ.get("RANK") or os.environ.get("LOCAL_RANK") or "0")


def _ddp_local_rank() -> int:
    import os

    return int(os.environ.get("LOCAL_RANK") or "0")


def _ddp_is_main() -> bool:
    return _ddp_rank() == 0


def _maybe_init_ddp() -> bool:
    """Init process group when launched under torchrun (WORLD_SIZE>1)."""
    if _ddp_world_size() <= 1:
        return False
    import os
    from datetime import timedelta

    import torch.distributed as dist

    if not dist.is_initialized():
        # Hold/eval can exceed default 10min NCCL watchdog; keep collectives aligned instead,
        # but also raise timeout so a slow hold shard cannot abort the job.
        timeout_min = int(os.environ.get("RWCITE_DDP_TIMEOUT_MIN", "120") or "120")
        dist.init_process_group(backend="nccl", timeout=timedelta(minutes=timeout_min))
    return True


def _ddp_allreduce_mean_metrics(metrics: dict[str, float], device: torch.device) -> dict[str, float]:
    """Weighted-mean hits/loss across ranks using local n as weight."""
    import torch.distributed as dist

    n = float(metrics.get("n") or 0.0)
    loss = float(metrics.get("loss") or 0.0)
    hits = float(metrics.get("mean_hits_at_10") or 0.0)
    buf = torch.tensor([n, loss * n, hits * n], dtype=torch.float64, device=device)
    dist.all_reduce(buf, op=dist.ReduceOp.SUM)
    n_sum = float(buf[0].item())
    if n_sum <= 0:
        out = dict(metrics)
        out["n"] = 0
        return out
    out = dict(metrics)
    out["n"] = int(n_sum)
    out["loss"] = float(buf[1].item() / n_sum)
    out["mean_hits_at_10"] = float(buf[2].item() / n_sum)
    return out


def _broadcast_hold_indices(
    hold_list: list[int], *, device: torch.device, seed: int, n: int, n_hold: int
) -> list[int]:
    """Rank0 owns hold_list; broadcast length + indices to all ranks."""
    import torch.distributed as dist

    if not dist.is_initialized():
        return hold_list
    if _ddp_is_main():
        buf = torch.tensor([n_hold, *hold_list], dtype=torch.long, device=device)
    else:
        buf = torch.zeros(1 + n_hold, dtype=torch.long, device=device)
    # first broadcast n_hold then full buffer
    meta = torch.tensor([n_hold], dtype=torch.long, device=device)
    if _ddp_is_main():
        meta[0] = len(hold_list)
    dist.broadcast(meta, src=0)
    n_hold = int(meta[0].item())
    buf = torch.zeros(n_hold, dtype=torch.long, device=device)
    if _ddp_is_main():
        buf[:] = torch.tensor(hold_list, dtype=torch.long, device=device)
    dist.broadcast(buf, src=0)
    return [int(x) for x in buf.cpu().tolist()]


def _load_id_map(path: Path) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return list(data["ids"])


def _zscore_1d(arr: np.ndarray) -> np.ndarray:
    if arr.size == 0:
        return arr.astype(np.float32)
    mu = float(np.nanmean(arr))
    sd = float(np.nanstd(arr))
    if not np.isfinite(sd) or sd < 1e-6:
        return np.zeros_like(arr, dtype=np.float32)
    out = (arr - mu) / sd
    return np.nan_to_num(out, nan=0.0).astype(np.float32)


def _load_ce_score_map(ce_npz: Path, windows: Any) -> dict[str, np.ndarray]:
    """Map query_id → raw CE scores aligned to window cand slots."""
    data = np.load(ce_npz, allow_pickle=True)
    ce_q = [str(x) for x in data["query_ids"].tolist()]
    ce_sc = data["ce_scores"]
    by_qid = {qid: ce_sc[i] for i, qid in enumerate(ce_q)}
    # sanity: cover windows query ids
    miss = 0
    for qid in windows["query_ids"]:
        if str(qid) not in by_qid:
            miss += 1
    if miss:
        print(f"WARN ce cache missing {miss} window queries", flush=True)
    return by_qid


def _edges_induced(mat, node_set: set[int]) -> tuple[np.ndarray, np.ndarray]:
    """Edges with both ends in node_set; message src=neighbor → dst=node (row)."""
    srcs: list[np.ndarray] = []
    dsts: list[np.ndarray] = []
    for i in node_set:
        s, e = mat.indptr[i], mat.indptr[i + 1]
        if s == e:
            continue
        nbrs = mat.indices[s:e]
        keep = [int(j) for j in nbrs if int(j) in node_set]
        if not keep:
            continue
        srcs.append(np.asarray(keep, dtype=np.int64))
        dsts.append(np.full(len(keep), i, dtype=np.int64))
    if not srcs:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    return np.concatenate(srcs), np.concatenate(dsts)


class WindowDataset(Dataset):
    def __init__(self, windows_path: Path, *, indices: list[int] | None = None) -> None:
        self.z = np.load(windows_path, allow_pickle=True)
        self.n = int(self.z["query_ids"].shape[0])
        self.indices = indices if indices is not None else list(range(self.n))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> int:
        return self.indices[i]


class GatTrainPack:
    def __init__(
        self,
        out_dir: Path,
        *,
        use_foldout: bool = True,
        exact_target_b: bool = True,
        expand_hop: int = 0,
        hop_budget: int = 64,
        l4_train_k: int = 32,
        l4_eval_k: int = 32,
    ) -> None:
        self.out_dir = out_dir
        self.ids = _load_id_map(out_dir / "id_map.json")
        self.id_to_i = {pid: i for i, pid in enumerate(self.ids)}
        whitened = out_dir / "node_scibert_whitened.npy"
        raw = out_dir / "node_scibert.npy"
        self.X = np.load(whitened if whitened.is_file() else raw).astype(np.float32)
        self.adj_in_full = load_csr(out_dir / "adj_in.npz")
        self.adj_out_full = load_csr(out_dir / "adj_out.npz")
        self.adj_cocite_full = load_csr(out_dir / "adj_cocite.npz")
        nbr = np.load(out_dir / "neighbors_k32.npz")
        self.nbr_k = nbr["neighbors_k32"]
        self.nbr_pool = nbr["neighbors_pool64"]
        self.use_foldout = use_foldout
        self.exact_target_b = exact_target_b
        self.expand_hop = expand_hop
        self.hop_budget = hop_budget
        self.l4_train_k = int(l4_train_k)
        self.l4_eval_k = int(l4_eval_k)

        self.fold_of: dict[str, int] = {}
        self.fold_adjs: dict[int, dict[str, Any]] = {}
        fold_meta = out_dir / "fold_masks.json"
        if use_foldout and fold_meta.is_file():
            meta = json.loads(fold_meta.read_text(encoding="utf-8"))
            self.fold_of = {str(k): int(v) for k, v in meta["fold_of"].items()}
            k = int(meta.get("k") or 4)
            for f in range(k):
                pin = out_dir / f"adj_in_fold{f}.npz"
                if not pin.is_file():
                    print(f"WARN: missing {pin}; fold-out disabled for fold {f}", flush=True)
                    continue
                self.fold_adjs[f] = {
                    "in": load_csr(pin),
                    "out": load_csr(out_dir / f"adj_out_fold{f}.npz"),
                    "cocite": load_csr(out_dir / f"adj_cocite_fold{f}.npz"),
                }
            print(
                f"fold-out loaded: {len(self.fold_adjs)} folds, "
                f"{len(self.fold_of)} train sources mapped",
                flush=True,
            )
        elif use_foldout:
            print("WARN: fold_masks.json missing — fold-out off; exact B still applied", flush=True)

    def _adjs_for_query(self, query_id: str) -> dict[str, Any]:
        if self.use_foldout and query_id in self.fold_of:
            f = self.fold_of[query_id]
            if f in self.fold_adjs:
                return self.fold_adjs[f]
        return {
            "in": self.adj_in_full,
            "out": self.adj_out_full,
            "cocite": self.adj_cocite_full,
        }

    def build_subgraph(
        self,
        *,
        query_global: int,
        query_id: str,
        cand_globals: np.ndarray,
        n_cands: int,
        train: bool,
        rng: np.random.Generator,
        apply_exact_b: bool | None = None,
    ) -> dict[str, Any]:
        apply_b = self.exact_target_b if apply_exact_b is None else apply_exact_b
        adjs = self._adjs_for_query(query_id)
        cands = [int(x) for x in cand_globals[:n_cands] if int(x) >= 0]
        k_l4 = self.l4_train_k if train else self.l4_eval_k
        k_l4 = max(1, min(int(k_l4), int(self.nbr_pool.shape[1])))
        nbr_mat = np.full((len(cands), k_l4), -1, dtype=np.int64)
        node_set: set[int] = set(cands)
        node_set.add(int(query_global))
        for ci, c in enumerate(cands):
            pool = self.nbr_pool[c]
            pool = pool[pool >= 0]
            if train and pool.size > k_l4:
                pick = rng.choice(pool, size=k_l4, replace=False)
            else:
                # eval (or small pool): deterministic top-k from nbr_k then pool
                base = self.nbr_k[c]
                base = base[base >= 0]
                if base.size >= k_l4:
                    pick = base[:k_l4]
                else:
                    # pad from pool preserving order
                    seen = set(int(x) for x in base)
                    extra = [int(x) for x in pool if int(x) not in seen]
                    pick = np.asarray(
                        list(base) + extra[: max(0, k_l4 - len(base))], dtype=np.int64
                    )
            for ki, g in enumerate(pick[:k_l4]):
                g = int(g)
                nbr_mat[ci, ki] = g
                node_set.add(g)

        # optional 1-hop expansion (R4): top citers of candidates
        if self.expand_hop > 0:
            extra: list[int] = []
            ain = adjs["in"]
            for c in cands:
                s, e = ain.indptr[c], ain.indptr[c + 1]
                citers = [int(x) for x in ain.indices[s:e] if int(x) != query_global]
                extra.extend(citers[: self.hop_budget])
            for g in extra[: self.expand_hop * len(cands)]:
                node_set.add(g)

        nodes = sorted(node_set)
        loc = {g: i for i, g in enumerate(nodes)}
        text = self.X[nodes]
        q_local = loc[int(query_global)]
        cand_locals = {loc[c] for c in cands}

        def remap(src: np.ndarray, dst: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
            if src.size == 0:
                z = torch.zeros(0, dtype=torch.long)
                return z, z
            return (
                torch.tensor([loc[int(x)] for x in src], dtype=torch.long),
                torch.tensor([loc[int(x)] for x in dst], dtype=torch.long),
            )

        ein_s, ein_d = _edges_induced(adjs["in"], node_set)
        eout_s, eout_d = _edges_induced(adjs["out"], node_set)
        eco_s, eco_d = _edges_induced(adjs["cocite"], node_set)

        # remap to local then exact B on in-channel
        ein_s_l = np.asarray([loc[int(x)] for x in ein_s], dtype=np.int64) if ein_s.size else ein_s
        ein_d_l = np.asarray([loc[int(x)] for x in ein_d], dtype=np.int64) if ein_d.size else ein_d
        b_dropped = 0
        if apply_b and ein_s_l.size:
            before = int(ein_s_l.size)
            ein_s_l, ein_d_l = drop_target_edges(
                ein_s_l, ein_d_l, query_local=q_local, cand_locals=cand_locals
            )
            b_dropped = before - int(ein_s_l.size)

        # assert: no q→cand remains on in
        leak_left = 0
        if ein_s_l.size:
            for a, b in zip(ein_s_l, ein_d_l):
                if int(a) == q_local and int(b) in cand_locals:
                    leak_left += 1

        C = len(cands)
        neighbors = np.full((C, k_l4), -1, dtype=np.int64)
        nmask = np.zeros((C, k_l4), dtype=bool)
        for ci in range(C):
            for ki in range(k_l4):
                g = int(nbr_mat[ci, ki])
                if g < 0:
                    continue
                neighbors[ci, ki] = loc[g]
                nmask[ci, ki] = True

        def _t(src: np.ndarray, dst: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
            if src.size == 0:
                z = torch.zeros(0, dtype=torch.long)
                return z, z
            return torch.tensor(src, dtype=torch.long), torch.tensor(dst, dtype=torch.long)

        return {
            "text": text,
            "edges": {
                "in": _t(ein_s_l, ein_d_l),
                "out": remap(eout_s, eout_d),
                "cocite": remap(eco_s, eco_d),
            },
            "cand_idx": torch.tensor([loc[c] for c in cands], dtype=torch.long),
            "query_idx": torch.tensor(q_local, dtype=torch.long),
            "neighbors": torch.tensor(neighbors, dtype=torch.long),
            "neighbor_mask": torch.tensor(nmask, dtype=torch.bool),
            "nbr_globals": nbr_mat.copy(),
            "n_nodes": len(nodes),
            "b_dropped": b_dropped,
            "b_leak_left": leak_left,
            "fold": self.fold_of.get(query_id),
        }


def score_query_chunked(
    model: GatMvpModel,
    pack: GatTrainPack,
    *,
    query_global: int,
    query_id: str,
    cand_globals: np.ndarray,
    n_cands: int,
    phi_np: np.ndarray,
    device: torch.device,
    rng: np.random.Generator,
    chunk_size: int = 0,
    ce_arr: np.ndarray | None = None,
    use_checkpoint: bool = False,
    return_attn: bool = False,
) -> dict[str, Any]:
    """Score all window cands; optionally chunk candidates to bound peak VRAM.

    When ``chunk_size`` > 0 and ``n_cands`` exceeds it, each chunk builds its own
    induced subgraph (L4 neighbors of that chunk only). Scores are concatenated
    in cand order — used for large fair-U windows (K≳1.5k).

    With ``return_attn=True``, also returns L4 ``l4_beta`` / ``l4_mask`` /
    ``l4_nbr_globals`` (global neighbor ids, A-mask applied).
    """
    del use_checkpoint  # reserved; model has no checkpoint hook yet
    nc = int(n_cands)
    cands = np.asarray(cand_globals[:nc], dtype=np.int64)
    phi = np.asarray(phi_np[:nc], dtype=np.float32)
    cs = int(chunk_size) if chunk_size and chunk_size > 0 else nc
    cs = max(1, cs)

    def _one(sl: slice) -> dict[str, torch.Tensor]:
        sub_c = cands[sl]
        sub_phi = phi[sl]
        sub = pack.build_subgraph(
            query_global=int(query_global),
            query_id=str(query_id),
            cand_globals=sub_c,
            n_cands=int(sub_c.shape[0]),
            train=False,
            rng=rng,
            apply_exact_b=True,
        )
        text = torch.tensor(sub["text"], device=device)
        edges = {k: (v[0].to(device), v[1].to(device)) for k, v in sub["edges"].items()}
        phi_t = torch.tensor(sub_phi, device=device)
        ce_t = None
        if ce_arr is not None:
            raw = np.asarray(ce_arr[sl], dtype=np.float32)
            ce_t = torch.tensor(_zscore_1d(raw), device=device)
        out = model(
            text,
            edges,
            sub["cand_idx"].to(device),
            sub["query_idx"].to(device),
            phi_t,
            sub["neighbors"].to(device),
            sub["neighbor_mask"].to(device),
            mask_target_edge=True,
            edge_dropout=0.0,
            ce_feat=ce_t,
            return_attn=return_attn,
        )
        if return_attn:
            nbr_g = torch.tensor(sub["nbr_globals"], device=device, dtype=torch.long)
            qg = int(query_global)
            mask = out["l4_mask"]
            hit = nbr_g == qg
            nbr_g = nbr_g.masked_fill(hit, -1)
            mask = mask & ~hit
            out = dict(out)
            out["l4_nbr_globals"] = nbr_g
            out["l4_mask"] = mask
        return out

    if nc <= cs:
        out0 = _one(slice(0, nc))
        if return_attn:
            return out0
        return {"scores": out0["scores"]}
    parts_s: list[torch.Tensor] = []
    parts_b: list[torch.Tensor] = []
    parts_m: list[torch.Tensor] = []
    parts_n: list[torch.Tensor] = []
    for start in range(0, nc, cs):
        o = _one(slice(start, min(nc, start + cs)))
        parts_s.append(o["scores"])
        if return_attn:
            parts_b.append(o["l4_beta"])
            parts_m.append(o["l4_mask"])
            parts_n.append(o["l4_nbr_globals"])
    merged: dict[str, Any] = {"scores": torch.cat(parts_s, dim=0)}
    if return_attn:
        kmax = max(int(t.shape[-1]) for t in parts_b)

        def _pad(t: torch.Tensor, fill: float | bool) -> torch.Tensor:
            if t.shape[-1] == kmax:
                return t
            pad = t.new_full((*t.shape[:-1], kmax - t.shape[-1]), fill)
            return torch.cat([t, pad], dim=-1)

        merged["l4_beta"] = torch.cat([_pad(t, 0.0) for t in parts_b], dim=0)
        merged["l4_mask"] = torch.cat([_pad(t, False) for t in parts_m], dim=0)
        merged["l4_nbr_globals"] = torch.cat([_pad(t, -1) for t in parts_n], dim=0)
    return merged


def listwise_loss(scores: torch.Tensor, gold: torch.Tensor) -> torch.Tensor:
    log_prob = F.log_softmax(scores, dim=0)
    pos = gold > 0
    if not pos.any():
        return scores.sum() * 0.0
    return -(log_prob[pos]).mean()


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = scores[labels > 0]
    neg = scores[labels <= 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    correct = 0.0
    for p in pos:
        correct += float((neg < p).sum()) + 0.5 * float((neg == p).sum())
    return correct / (len(pos) * len(neg))


def leak_selfcheck(
    model: GatMvpModel,
    pack: GatTrainPack,
    windows: Any,
    *,
    device: torch.device,
    n_samples: int = 64,
    seed: int = 0,
) -> dict[str, Any]:
    """A-mask + B on/off AUC gap; query-in-N_K; assert no residual q→cand on in."""
    rng = np.random.default_rng(seed)
    n = int(windows["query_ids"].shape[0])
    idxs = list(rng.choice(n, size=min(n_samples, n), replace=False))
    id_to_i = pack.id_to_i
    aucs_b: list[float] = []
    aucs_no_b: list[float] = []
    self_hit_fracs: list[float] = []
    b_leaks = 0
    b_dropped_total = 0
    model.eval()
    with torch.no_grad():
        for i in idxs:
            qid = str(windows["query_ids"][i])
            qg = id_to_i.get(qid, -1)
            if qg < 0:
                continue
            nc = int(windows["n_cands"][i])
            if nc < 2:
                continue
            gold = windows["gold_mask"][i, :nc].astype(np.float32)
            if gold.sum() < 1:
                continue

            def _score(apply_b: bool) -> tuple[np.ndarray, dict[str, Any]]:
                sub = pack.build_subgraph(
                    query_global=qg,
                    query_id=qid,
                    cand_globals=windows["cand_idx"][i],
                    n_cands=nc,
                    train=False,
                    rng=rng,
                    apply_exact_b=apply_b,
                )
                text = torch.tensor(sub["text"], device=device)
                edges = {
                    k: (v[0].to(device), v[1].to(device)) for k, v in sub["edges"].items()
                }
                phi = torch.tensor(windows["phi_struct"][i, :nc], device=device)
                out = model(
                    text,
                    edges,
                    sub["cand_idx"].to(device),
                    sub["query_idx"].to(device),
                    phi,
                    sub["neighbors"].to(device),
                    sub["neighbor_mask"].to(device),
                    mask_target_edge=True,
                    edge_dropout=0.0,
                )
                return out["scores"].detach().cpu().numpy(), sub

            sc_b, sub_b = _score(True)
            sc_nb, _ = _score(False)
            aucs_b.append(roc_auc(sc_b, gold))
            aucs_no_b.append(roc_auc(sc_nb, gold))
            b_leaks += int(sub_b["b_leak_left"])
            b_dropped_total += int(sub_b["b_dropped"])

            qloc = int(sub_b["query_idx"].item())
            self_pos = (sub_b["neighbors"] == qloc) & sub_b["neighbor_mask"]
            self_hit_fracs.append(float(self_pos.any(dim=1).float().mean()))

    arr_b = np.asarray([a for a in aucs_b if not math.isnan(a)], dtype=np.float64)
    arr_nb = np.asarray([a for a in aucs_no_b if not math.isnan(a)], dtype=np.float64)
    mean_auc_b = float(arr_b.mean()) if len(arr_b) else float("nan")
    mean_auc_nb = float(arr_nb.mean()) if len(arr_nb) else float("nan")
    b_gap = (
        float(mean_auc_nb - mean_auc_b)
        if not (math.isnan(mean_auc_b) or math.isnan(mean_auc_nb))
        else float("nan")
    )
    mean_self_hit = float(np.mean(self_hit_fracs)) if self_hit_fracs else 0.0
    ok = (
        (math.isnan(mean_auc_b) or mean_auc_b <= 0.99)
        and mean_self_hit < 0.85
        and b_leaks == 0
    )
    return {
        "n_samples": len(aucs_b),
        "mean_auc_with_B": mean_auc_b,
        "mean_auc_without_B": mean_auc_nb,
        "auc_gap_B_off_minus_on": b_gap,
        "mean_query_in_nbr_frac": mean_self_hit,
        "b_leak_left_total": b_leaks,
        "b_dropped_total": b_dropped_total,
        "leak_gate_ok": ok,
        "note": (
            "B must leave zero q→cand on in-edges; AUC≤0.99 with B; "
            "large AUC gap when B off suggests residual target-edge leakage."
        ),
    }


def train_loop(args: argparse.Namespace) -> dict[str, Any]:
    use_ddp = _maybe_init_ddp()
    ws = _ddp_world_size()
    rank = _ddp_rank()
    local_rank = _ddp_local_rank()
    root = P.root_path()
    dpaths = resolve_gat_domain(getattr(args, "domain", "ewm") or "ewm", root)
    out_dir = Path(args.out_dir) if args.out_dir else root / dpaths.gat_mvp_dir
    if not out_dir.is_absolute():
        out_dir = root / out_dir
    ckpt_name = args.ckpt_name or "ckpt_e4.pt"
    if use_ddp:
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device(
            args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

    cfg = build_ablation_config(args.ablation) if args.ablation else GatMvpConfig()
    cfg.struct_residual_scale = float(args.struct_residual_scale)
    cfg.use_ce_feat = bool(args.use_ce_feat)
    # struct-only true E3: still build full model but train only struct
    model = GatMvpModel(cfg)
    struct_path = dpaths.struct_model
    if struct_path.is_file():
        model.load_struct_init(struct_path)
    if args.init_ckpt:
        init_path = Path(args.init_ckpt)
        if not init_path.is_file():
            raise FileNotFoundError(f"init ckpt missing: {init_path}")
        init = torch.load(init_path, map_location="cpu", weights_only=False)
        if cfg.use_ce_feat:
            model.load_partial_from_e4(init["model"])
            if _ddp_is_main():
                print(f"loaded partial E4→cefeat from {init_path}", flush=True)
        else:
            model.load_state_dict(init["model"], strict=False)
            if _ddp_is_main():
                print(f"loaded init ckpt {init_path}", flush=True)
    model.to(device)

    accumulate_k = max(1, int(getattr(args, "accumulate_k", 1) or 1))
    ddp_lr_scale = bool(getattr(args, "ddp_lr_scale", False))
    lr = float(args.lr)
    if use_ddp and ddp_lr_scale:
        lr = lr * (ws**0.5)
    if use_ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            # All params used each forward; True adds graph walk + false-positive risk.
            find_unused_parameters=False,
        )
    raw_model = model.module if use_ddp else model

    pack = GatTrainPack(
        out_dir,
        use_foldout=not args.no_foldout,
        exact_target_b=not args.no_exact_b,
        expand_hop=args.expand_hop,
        hop_budget=args.hop_budget,
        l4_train_k=args.l4_train_k,
        l4_eval_k=args.l4_eval_k,
    )
    n_train = 3423
    meta = dpaths.split_dir / "split_meta.json"
    if meta.is_file():
        n_train = int(
            json.loads(meta.read_text(encoding="utf-8")).get("n_train_sources") or n_train
        )
    default_win = out_dir / f"windows_train{n_train}.npz"
    if not default_win.is_file() and dpaths.domain == "ewm":
        legacy = out_dir / "windows_train3423.npz"
        if legacy.is_file():
            default_win = legacy
            n_train = 3423
    win_path = Path(args.windows) if args.windows else default_win
    if not Path(win_path).is_absolute():
        win_path = root / win_path
    windows = np.load(win_path, allow_pickle=True)
    n = int(windows["query_ids"].shape[0])
    id_to_i = pack.id_to_i
    ce_by_qid: dict[str, np.ndarray] | None = None
    if args.use_ce_feat:
        default_ce = out_dir / f"ce_scores_train{n_train}_sent.npz"
        if not default_ce.is_file() and dpaths.domain == "ewm":
            legacy_ce = out_dir / "ce_scores_train3423_sent.npz"
            if legacy_ce.is_file():
                default_ce = legacy_ce
        ce_path = Path(args.ce_cache) if args.ce_cache else default_ce
        if not ce_path.is_file():
            raise FileNotFoundError(f"CE cache required for --use-ce-feat: {ce_path}")
        ce_by_qid = _load_ce_score_map(ce_path, windows)
        print(f"ce feat cache {ce_path} n={len(ce_by_qid)}", flush=True)

    rng = np.random.default_rng(args.seed)
    if use_ddp:
        import torch.distributed as dist

        if _ddp_is_main():
            all_idx = list(range(n))
            rng.shuffle(all_idx)
            buf = torch.tensor(all_idx, dtype=torch.long, device=device)
        else:
            buf = torch.zeros(n, dtype=torch.long, device=device)
        dist.broadcast(buf, src=0)
        all_idx = [int(x) for x in buf.cpu().tolist()]
    else:
        all_idx = list(range(n))
        rng.shuffle(all_idx)
    n_hold = max(1, int(0.1 * n))
    hold_list = all_idx[:n_hold]
    hold_idx = set(hold_list)
    train_idx = all_idx[n_hold:]
    if args.max_queries > 0:
        train_idx = train_idx[: args.max_queries]
        hold_list = list(hold_idx)[: max(1, args.max_queries // 10)]
        hold_idx = set(hold_list)
    # Query-shard: each rank walks its shard once per epoch (global covers train once).
    # Drop remainder so every rank has identical step count → equal DDP allreduces.
    if use_ddp and ws > 1:
        n_usable = (len(train_idx) // ws) * ws
        if n_usable < len(train_idx) and _ddp_is_main():
            print(
                f"ddp drop {len(train_idx) - n_usable} train queries so len%{ws}==0 "
                f"(was {len(train_idx)})",
                flush=True,
            )
        train_idx = train_idx[:n_usable]
    train_idx_rank = train_idx[rank::ws] if use_ddp else list(train_idx)
    hold_list_sorted = sorted(hold_idx)
    hold_idx_rank = hold_list_sorted[rank::ws] if use_ddp else list(hold_list_sorted)
    if _ddp_is_main():
        print(
            f"ddp={use_ddp} world_size={ws} rank={rank} n_train={len(train_idx)} "
            f"n_train_rank={len(train_idx_rank)} n_hold={len(hold_idx)} "
            f"n_hold_rank={len(hold_idx_rank)} "
            f"accumulate_k={accumulate_k} lr={lr} ddp_lr_scale={ddp_lr_scale}",
            flush=True,
        )

    struct_params = [raw_model.struct_w, raw_model.struct_b]
    if args.freeze_gnn:
        for p in raw_model.parameters():
            p.requires_grad_(False)
        for p in raw_model.mlp.parameters():
            p.requires_grad_(True)
        if raw_model.ce_ln is not None:
            for p in raw_model.ce_ln.parameters():
                p.requires_grad_(True)
        if args.freeze_struct:
            opt = torch.optim.AdamW(
                [p for p in raw_model.parameters() if p.requires_grad],
                lr=lr,
                weight_decay=args.weight_decay,
            )
        else:
            raw_model.struct_w.requires_grad_(True)
            raw_model.struct_b.requires_grad_(True)
            mlp_params = [
                p
                for n, p in raw_model.named_parameters()
                if p.requires_grad and not n.startswith("struct_")
            ]
            opt = torch.optim.AdamW(
                [
                    {"params": mlp_params, "lr": lr},
                    {"params": struct_params, "lr": lr * 0.1},
                ],
                weight_decay=args.weight_decay,
            )
        if _ddp_is_main():
            print(
                f"freeze_gnn=1 trainable={sum(p.numel() for p in raw_model.parameters() if p.requires_grad)}",
                flush=True,
            )
    elif args.freeze_struct:
        raw_model.struct_w.requires_grad_(False)
        raw_model.struct_b.requires_grad_(False)
        other = [p for p in raw_model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(other, lr=lr, weight_decay=args.weight_decay)
    elif args.struct_only_train:
        for p in raw_model.parameters():
            p.requires_grad_(False)
        raw_model.struct_w.requires_grad_(True)
        raw_model.struct_b.requires_grad_(True)
        raw_model.cfg.struct_only = True
        opt = torch.optim.AdamW(struct_params, lr=lr, weight_decay=args.weight_decay)
    else:
        opt = torch.optim.AdamW(
            [
                {
                    "params": [
                        p
                        for n, p in raw_model.named_parameters()
                        if not n.startswith("struct_")
                    ],
                    "lr": lr,
                },
                {"params": struct_params, "lr": lr * 0.1},
            ],
            weight_decay=args.weight_decay,
        )

    ckpt_name = args.ckpt_name or "ckpt.pt"
    best_path = out_dir / ckpt_name
    report: dict[str, Any] = {
        "n_train_queries": len(train_idx),
        "n_train_queries_rank": len(train_idx_rank),
        "n_hold_queries": len(hold_idx),
        "n_params": raw_model.count_params(),
        "ablation": args.ablation or "E4",
        "device": str(device),
        "world_size": ws,
        "accumulate_K": accumulate_k,
        "global_queries_per_opt_step": ws * accumulate_k,
        "ddp_lr_scale": ddp_lr_scale,
        "effective_lr": lr,
        "foldout": not args.no_foldout,
        "exact_b": not args.no_exact_b,
        "freeze_struct": bool(args.freeze_struct),
        "freeze_gnn": bool(args.freeze_gnn),
        "use_ce_feat": bool(args.use_ce_feat),
        "struct_residual_scale": float(args.struct_residual_scale),
        "l4_train_k": int(args.l4_train_k),
        "l4_eval_k": int(args.l4_eval_k),
        "struct_only_train": bool(args.struct_only_train),
        "full_window_hold": True,
        "windows": str(win_path),
        "epochs": [],
        "ckpt": str(best_path),
    }
    if _ddp_is_main():
        print(
            f"params={raw_model.count_params()} train_q={len(train_idx)} "
            f"train_rank={len(train_idx_rank)} hold={len(hold_idx)} "
            f"foldout={not args.no_foldout} exact_b={not args.no_exact_b} "
            f"freeze_struct={args.freeze_struct} ws={ws} K={accumulate_k}",
            flush=True,
        )

    best_hold = -1e9
    best_epoch = 0
    t0 = time.perf_counter()
    opt_steps_box = [0]

    def _ddp_train_sync_skip() -> None:
        """Keep DDP allreduce counts aligned when a query is skipped mid-epoch."""
        if not use_ddp:
            return
        loss = None
        for p in raw_model.parameters():
            if not p.requires_grad:
                continue
            term = p.reshape(-1)[0] * 0.0
            loss = term if loss is None else loss + term
        if loss is None:
            return
        (loss / accumulate_k).backward()
        opt.zero_grad(set_to_none=True)

    def run_epoch(indices: list[int], *, train: bool) -> dict[str, float]:
        model.train(train)
        total_loss = 0.0
        total_hits = 0.0
        n_ok = 0
        g_bars: list[float] = []
        max_betas: list[float] = []
        attn_ents: list[float] = []
        n_eff_ge4: list[float] = []
        max_beta_near1: list[float] = []
        b_dropped = 0
        b_leaks = 0
        for bi, i in enumerate(indices):
            qid = str(windows["query_ids"][i])
            qg = id_to_i.get(qid)
            if qg is None:
                if train:
                    _ddp_train_sync_skip()
                continue
            nc = int(windows["n_cands"][i])
            gold = windows["gold_mask"][i, :nc]
            if nc < 2 or gold.sum() < 1:
                if train:
                    _ddp_train_sync_skip()
                continue

            # train: optional neg subsample; hold/eval: full window
            if train and not args.full_window_train:
                pos_ix = np.where(gold)[0]
                neg_ix = np.where(~gold)[0]
                neg_keep = min(len(neg_ix), max(4, args.neg_per_pos * len(pos_ix)))
                if len(neg_ix) > neg_keep:
                    neg_ix = rng.choice(neg_ix, size=neg_keep, replace=False)
                keep = np.concatenate([pos_ix, neg_ix])
                keep.sort()
            else:
                keep = np.arange(nc)

            sub = pack.build_subgraph(
                query_global=qg,
                query_id=qid,
                cand_globals=windows["cand_idx"][i],
                n_cands=nc,
                train=train,
                rng=rng,
            )
            b_dropped += int(sub["b_dropped"])
            b_leaks += int(sub["b_leak_left"])
            text = torch.tensor(sub["text"], device=device)
            edges = {k: (v[0].to(device), v[1].to(device)) for k, v in sub["edges"].items()}
            cand_idx = sub["cand_idx"].to(device)[keep]
            phi = torch.tensor(windows["phi_struct"][i, :nc][keep], device=device)
            neighbors = sub["neighbors"].to(device)[keep]
            nmask = sub["neighbor_mask"].to(device)[keep]
            gold_t = torch.tensor(gold[keep].astype(np.float32), device=device)
            ce_t = None
            if ce_by_qid is not None:
                raw = ce_by_qid.get(qid)
                if raw is None:
                    ce_arr = np.zeros(nc, dtype=np.float32)
                else:
                    ce_arr = np.asarray(raw[:nc], dtype=np.float32)
                ce_t = torch.tensor(_zscore_1d(ce_arr)[keep], device=device)

            out = model(
                text,
                edges,
                cand_idx,
                sub["query_idx"].to(device),
                phi,
                neighbors,
                nmask,
                mask_target_edge=True,
                ce_feat=ce_t,
            )
            loss = listwise_loss(out["scores"], gold_t)
            if train:
                loss = loss / accumulate_k
                loss.backward()
                if (n_ok + 1) % accumulate_k == 0:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in raw_model.parameters() if p.requires_grad], 5.0
                    )
                    opt.step()
                    opt.zero_grad(set_to_none=True)
                    opt_steps_box[0] += 1
                total_loss += float(loss.detach().cpu()) * accumulate_k
            else:
                total_loss += float(loss.detach().cpu())
            topk = min(10, out["scores"].numel())
            # map back to keep-local indices
            pred = set(
                out["scores"].detach().cpu().numpy().argsort()[::-1][:topk].tolist()
            )
            gold_set = set(np.where(gold[keep])[0].tolist())
            total_hits += float(len(pred & gold_set))
            n_ok += 1
            g_bars.append(float(out["g_bar"].cpu()))
            max_betas.append(float(out["max_beta"].detach().cpu()))
            attn_ents.append(float(out["attn_entropy"].detach().cpu()))
            n_eff_ge4.append(float(out["n_eff_ge4_frac"].detach().cpu()))
            max_beta_near1.append(float(out["max_beta_near1_frac"].detach().cpu()))
            if (bi + 1) % 50 == 0:
                print(
                    f"  {'train' if train else 'hold'} {bi+1}/{len(indices)} "
                    f"loss={total_loss/max(n_ok,1):.4f} hits@10≈{total_hits/max(n_ok,1):.3f}",
                    flush=True,
                )
        near1 = float(np.mean(max_beta_near1)) if max_beta_near1 else 0.0
        if near1 > 0.35 and not train:
            print(
                f"  WARN L4: max_beta>0.99 frac={near1:.3f} (collapse risk)",
                flush=True,
            )
        return {
            "loss": total_loss / max(n_ok, 1),
            "mean_hits_at_10": total_hits / max(n_ok, 1),
            "g_bar_mean": float(np.mean(g_bars)) if g_bars else None,
            "max_beta_mean": float(np.mean(max_betas)) if max_betas else None,
            "attn_entropy_mean": float(np.mean(attn_ents)) if attn_ents else None,
            "n_eff_ge4_frac": float(np.mean(n_eff_ge4)) if n_eff_ge4 else None,
            "max_beta_near1_frac": near1 if max_beta_near1 else None,
            "n": n_ok,
            "b_dropped": b_dropped,
            "b_leak_left": b_leaks,
        }

    epochs_no_improve = 0
    for epoch in range(1, args.epochs + 1):
        tr = run_epoch(train_idx_rank, train=True)
        # Flush leftover grads if last microbatch incomplete
        if any(p.grad is not None for p in raw_model.parameters() if p.requires_grad):
            torch.nn.utils.clip_grad_norm_(
                [p for p in raw_model.parameters() if p.requires_grad], 5.0
            )
            opt.step()
            opt.zero_grad(set_to_none=True)
            opt_steps_box[0] += 1
        if use_ddp:
            import torch.distributed as dist

            tr = _ddp_allreduce_mean_metrics(tr, device)
            dist.barrier()
            # Shard hold across ranks (all ranks busy) then reduce — avoids rank0-only
            # hold while peers sit on NCCL broadcast and hit the watchdog timeout.
            hold = run_epoch(hold_idx_rank, train=False)
            hold = _ddp_allreduce_mean_metrics(hold, device)
            dist.barrier()
        else:
            hold = run_epoch(hold_list_sorted, train=False)
        row = {"epoch": epoch, "train": tr, "hold": hold}
        if _ddp_is_main():
            report["epochs"].append(row)
            print(
                f"epoch {epoch}: train loss={tr['loss']:.4f} @10={tr['mean_hits_at_10']:.3f} "
                f"hold @10={hold['mean_hits_at_10']:.3f} ḡ={tr.get('g_bar_mean')} "
                f"b_drop={tr['b_dropped']} leak_left={tr['b_leak_left']}",
                flush=True,
            )
        improved = False
        if _ddp_is_main() and hold["mean_hits_at_10"] > best_hold:
            best_hold = hold["mean_hits_at_10"]
            best_epoch = epoch
            epochs_no_improve = 0
            improved = True
            torch.save(
                {
                    "model": raw_model.state_dict(),
                    "cfg": raw_model.cfg.__dict__,
                    "epoch": epoch,
                    "hold_hits_at_10": best_hold,
                    "ablation": args.ablation or "E4",
                    "freeze_struct": bool(args.freeze_struct),
                    "freeze_gnn": bool(args.freeze_gnn),
                    "use_ce_feat": bool(args.use_ce_feat),
                    "struct_residual_scale": float(args.struct_residual_scale),
                    "l4_train_k": int(args.l4_train_k),
                    "l4_eval_k": int(args.l4_eval_k),
                    "foldout": not args.no_foldout,
                    "exact_b": not args.no_exact_b,
                    "world_size": ws,
                    "accumulate_K": accumulate_k,
                },
                best_path,
            )
            print(f"  saved {best_path}", flush=True)
        elif _ddp_is_main():
            epochs_no_improve += 1
        if use_ddp:
            import torch.distributed as dist

            flag = torch.tensor(
                [1 if (_ddp_is_main() and epoch >= args.min_epochs and epochs_no_improve >= args.patience) else 0],
                device=device,
            )
            dist.broadcast(flag, src=0)
            stop = bool(flag.item())
        else:
            stop = epoch >= args.min_epochs and epochs_no_improve >= args.patience
        if _ddp_is_main():
            report["best_epoch"] = best_epoch
        if stop:
            if _ddp_is_main():
                print(f"early stop at epoch {epoch} (best={best_epoch})", flush=True)
            break
        del improved

    # reload best for leak check (rank0)
    if _ddp_is_main() and best_path.is_file():
        ckpt = torch.load(best_path, map_location=device, weights_only=False)
        raw_model.load_state_dict(ckpt["model"])

    if _ddp_is_main():
        leak = leak_selfcheck(
            raw_model, pack, windows, device=device, n_samples=args.leak_samples
        )
        report["leak_selfcheck"] = leak
        report["elapsed_s"] = float(time.perf_counter() - t0)
        report["best_hold_hits_at_10"] = best_hold
        report["opt_steps"] = int(opt_steps_box[0])
        report["approx_global_query_updates"] = int(opt_steps_box[0]) * ws * accumulate_k

        best_row = next(
            (r for r in report["epochs"] if r["epoch"] == best_epoch),
            report["epochs"][-1] if report["epochs"] else None,
        )
        if best_row:
            tr_h = best_row["train"]["mean_hits_at_10"]
            ho_h = best_row["hold"]["mean_hits_at_10"]
            gap = (tr_h - ho_h) / max(tr_h, 1e-6)
            report["hold_gap_frac_at_best"] = float(gap)
            report["foldout_gap_gate_fail"] = bool(gap > 0.20 and args.no_foldout)
            report["caution_large_hold_gap"] = bool(gap > 0.20)

        report_path = out_dir / (args.report_name or "train_report.json")
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(
            json.dumps(
                {
                    "leak": leak,
                    "params": report["n_params"],
                    "best_hold": best_hold,
                    "best_epoch": best_epoch,
                    "world_size": ws,
                    "opt_steps": report["opt_steps"],
                },
                indent=2,
            ),
            flush=True,
        )
        if not leak["leak_gate_ok"]:
            print("LEAK GATE FAILED", flush=True)
        elif report.get("foldout_gap_gate_fail"):
            print("FOLDOUT REQUIRED (hold gap>20% without fold-out)", flush=True)
        else:
            print(f"wrote {report_path}", flush=True)
    else:
        report = {"leak_selfcheck": {"leak_gate_ok": True}, "foldout_gap_gate_fail": False}

    if use_ddp:
        import torch.distributed as dist

        dist.barrier()
        if dist.is_initialized():
            dist.destroy_process_group()
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="GAT MVP train (B-clean / fold-out)")
    ap.add_argument("--domain", default="ewm")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--windows", default=None)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--min-epochs", type=int, default=5)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--neg-per-pos", type=int, default=8)
    ap.add_argument("--max-queries", type=int, default=0)
    ap.add_argument("--leak-samples", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--ablation", default="E4")
    ap.add_argument("--ckpt-name", default="ckpt_e4.pt")
    ap.add_argument("--report-name", default="train_report_e4.json")
    ap.add_argument("--freeze-struct", action="store_true")
    ap.add_argument(
        "--struct-residual-scale",
        type=float,
        default=1.0,
        help="Multiply struct residual term (R4b press; <1 forces GNN/L4)",
    )
    ap.add_argument("--l4-train-k", type=int, default=32, help="L4 neighbor sample size in train")
    ap.add_argument("--l4-eval-k", type=int, default=32, help="L4 neighbor size in hold/eval")
    ap.add_argument("--struct-only-train", action="store_true", help="true E3")
    ap.add_argument("--no-foldout", action="store_true")
    ap.add_argument("--no-exact-b", action="store_true")
    ap.add_argument("--full-window-train", action="store_true")
    ap.add_argument("--expand-hop", type=int, default=0)
    ap.add_argument("--hop-budget", type=int, default=64)
    ap.add_argument("--use-ce-feat", action="store_true", help="L1: CE-sent scalar into L5")
    ap.add_argument("--ce-cache", default=None, help="ce_scores_*.npz aligned to --windows")
    ap.add_argument("--init-ckpt", default=None, help="warm-start (E4) before cefeat finetune")
    ap.add_argument(
        "--freeze-gnn",
        action="store_true",
        help="L1: freeze GNN/L4; train MLP (+ optional struct)",
    )
    ap.add_argument(
        "--accumulate-k",
        type=int,
        default=1,
        help="Gradient accumulate over K queries before optimizer.step (DDP default 1)",
    )
    ap.add_argument(
        "--ddp-lr-scale",
        action="store_true",
        help="Multiply lr by sqrt(world_size) under DDP (default OFF per CE lesson)",
    )
    args = ap.parse_args(argv)
    report = train_loop(args)
    if not _ddp_is_main():
        return 0
    if not report.get("leak_selfcheck", {}).get("leak_gate_ok", False):
        return 2
    if report.get("foldout_gap_gate_fail"):
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
