"""§5.3.9-B: K-fold fold-out adjacency + exact target-edge drop helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp

from rwcite.gat.preprocess import _save_csr, load_csr


def assign_folds(
    source_ids: list[str],
    *,
    k: int = 4,
    seed: int = 0,
) -> dict[str, Any]:
    """Deterministic fold assignment for train source ids."""
    ids = sorted(source_ids)
    rng = np.random.default_rng(seed)
    order = np.arange(len(ids))
    rng.shuffle(order)
    fold_of: dict[str, int] = {}
    folds: dict[str, list[str]] = {str(i): [] for i in range(k)}
    for rank, oi in enumerate(order):
        f = int(rank % k)
        sid = ids[int(oi)]
        fold_of[sid] = f
        folds[str(f)].append(sid)
    return {"k": k, "seed": seed, "fold_of": fold_of, "folds": folds, "n_sources": len(ids)}


def filter_adj_in_drop_citers(adj_in: sp.csr_matrix, drop: set[int]) -> sp.csr_matrix:
    """adj_in[cand, citer] — remove columns where citer ∈ drop."""
    adj_in = adj_in.tocsr()
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    for i in range(adj_in.shape[0]):
        s, e = adj_in.indptr[i], adj_in.indptr[i + 1]
        for j, v in zip(adj_in.indices[s:e], adj_in.data[s:e]):
            j = int(j)
            if j in drop:
                continue
            rows.append(i)
            cols.append(j)
            data.append(float(v))
    if not rows:
        return sp.csr_matrix(adj_in.shape, dtype=np.float32)
    return sp.csr_matrix(
        (np.asarray(data, dtype=np.float32), (rows, cols)), shape=adj_in.shape
    )


def filter_adj_out_drop_sources(adj_out: sp.csr_matrix, drop: set[int]) -> sp.csr_matrix:
    """adj_out[citer, ref] — remove rows where citer ∈ drop."""
    adj_out = adj_out.tocsr()
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    for i in range(adj_out.shape[0]):
        if i in drop:
            continue
        s, e = adj_out.indptr[i], adj_out.indptr[i + 1]
        for j, v in zip(adj_out.indices[s:e], adj_out.data[s:e]):
            rows.append(i)
            cols.append(int(j))
            data.append(float(v))
    if not rows:
        return sp.csr_matrix(adj_out.shape, dtype=np.float32)
    return sp.csr_matrix(
        (np.asarray(data, dtype=np.float32), (rows, cols)), shape=adj_out.shape
    )


def rebuild_cocite_from_out(
    adj_out: sp.csr_matrix,
    *,
    cocite_topk: int = 64,
    hub_cap: int = 80,
) -> sp.csr_matrix:
    """Rebuild cocite top-K from a fold-visible out-adjacency."""
    from collections import defaultdict

    n = adj_out.shape[0]
    counts: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    adj_out = adj_out.tocsr()
    for u in range(n):
        s, e = adj_out.indptr[u], adj_out.indptr[u + 1]
        refs = [int(x) for x in adj_out.indices[s:e]]
        refs = list(dict.fromkeys(refs))
        if len(refs) < 2:
            continue
        if len(refs) > hub_cap:
            rng = np.random.default_rng(u)
            refs = list(rng.choice(refs, size=hub_cap, replace=False))
        m = len(refs)
        for a in range(m):
            for b in range(a + 1, m):
                ra, rb = refs[a], refs[b]
                counts[ra][rb] += 1
                counts[rb][ra] += 1
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    for i, nbrs in counts.items():
        top = sorted(nbrs.items(), key=lambda x: (-x[1], x[0]))[:cocite_topk]
        for j, c in top:
            rows.append(i)
            cols.append(j)
            data.append(float(c))
    if not rows:
        return sp.csr_matrix((n, n), dtype=np.float32)
    return sp.csr_matrix(
        (np.asarray(data, dtype=np.float32), (rows, cols)), shape=(n, n)
    )


def drop_target_edges(
    src: np.ndarray,
    dst: np.ndarray,
    *,
    query_local: int,
    cand_locals: set[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Exact B: drop messages src=query → dst∈cands (q→v on in-channel)."""
    if src.size == 0:
        return src, dst
    keep = np.ones(src.shape[0], dtype=bool)
    for i in range(src.shape[0]):
        if int(src[i]) == int(query_local) and int(dst[i]) in cand_locals:
            keep[i] = False
    return src[keep], dst[keep]


def build_fold_adjacencies(
    out_dir: Path,
    *,
    id_map_path: Path,
    windows_train_path: Path,
    k: int = 4,
    seed: int = 0,
) -> dict[str, Any]:
    """Write fold_masks.json + adj_{in,out,cocite}_fold{f}.npz."""
    ids = json.loads(id_map_path.read_text(encoding="utf-8"))["ids"]
    id_to_idx = {pid: i for i, pid in enumerate(ids)}
    win = np.load(windows_train_path, allow_pickle=True)
    train_sources = [str(x) for x in win["query_ids"].tolist()]
    # only sources present in id map
    train_sources = [s for s in train_sources if s in id_to_idx]
    meta = assign_folds(train_sources, k=k, seed=seed)
    meta_path = out_dir / "fold_masks.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    adj_in = load_csr(out_dir / "adj_in.npz")
    adj_out = load_csr(out_dir / "adj_out.npz")
    report = {"k": k, "folds": {}}

    for f in range(k):
        drop_ids = meta["folds"][str(f)]
        drop_idx = {id_to_idx[s] for s in drop_ids if s in id_to_idx}
        ain = filter_adj_in_drop_citers(adj_in, drop_idx)
        aout = filter_adj_out_drop_sources(adj_out, drop_idx)
        print(f"  fold {f}: drop_sources={len(drop_idx)} rebuilding cocite…", flush=True)
        aco = rebuild_cocite_from_out(aout, cocite_topk=64)
        _save_csr(out_dir / f"adj_in_fold{f}.npz", ain)
        _save_csr(out_dir / f"adj_out_fold{f}.npz", aout)
        _save_csr(out_dir / f"adj_cocite_fold{f}.npz", aco)
        report["folds"][str(f)] = {
            "n_drop": len(drop_idx),
            "nnz_in": int(ain.nnz),
            "nnz_out": int(aout.nnz),
            "nnz_cocite": int(aco.nnz),
        }
        print(
            f"  fold {f}: nnz_in={ain.nnz} nnz_out={aout.nnz} nnz_cocite={aco.nnz}",
            flush=True,
        )

    report_path = out_dir / "fold_adj_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return {"meta": str(meta_path), "report": report}
