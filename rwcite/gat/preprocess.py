"""Preprocess split-masked graph features for the GAT ranker.

Builds under ``…/ranker/gat_mvp/``:
  - node_scibert.npy + id_map.json
  - whiten.npz (+ pair-cos diagnostics in meta)
  - adj_in.npz / adj_out.npz / adj_cocite.npz
  - neighbors_k32.npz (citer top-16 + cocite top-16; train pool top-64)
  - windows_test381.npz / windows_train3423.npz
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import scipy.sparse as sp

from rwcite.gat import NOFUTURE_GEXF_SHA256, SPLIT_ID
from rwcite.gat import protocol as P
from rwcite.gat.domain_layout import effective_nofuture_gexf, resolve_gat_domain
from rwcite.gat.phase0_window_indeg import _load_unique_queries, _sha256
from rwcite.ranker.reference_recommend import normalize_arxiv_id
from rwcite.ranker.rr_graph_mask import masked_citer_ids
from rwcite.ranker.rr_ranker import extract_features, load_ranker
from rwcite.ranker.rr_struct_coarse import rank_struct_coarse
from rwcite.ranker.rr_universe import build_rr_universe
from rwcite.runtime.deps import get_domain_cache, get_retriever_service


def _out_dir(root: Path, rel: str | None = None) -> Path:
    d = root / (rel or P.GAT_MVP_CKPT_DIR)
    d.mkdir(parents=True, exist_ok=True)
    return d


def encode_scibert(
    graph: nx.DiGraph,
    ids: list[str],
    *,
    model_dir: Path,
    batch_size: int = 64,
    max_len: int = 256,
    device: str | None = None,
) -> np.ndarray:
    """Frozen SciBERT (B0) mean-pool → [N, 768] float32."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(str(model_dir))
    # Prefer safetensors (torch<2.6 blocks torch.load of .bin via transformers).
    try:
        model = AutoModel.from_pretrained(str(model_dir), use_safetensors=True)
    except (TypeError, OSError):
        model = AutoModel.from_pretrained(str(model_dir))
    model.eval()
    model.to(device)
    for p in model.parameters():
        p.requires_grad_(False)

    texts: list[str] = []
    for pid in ids:
        data = graph.nodes.get(pid) or graph.nodes.get(pid.replace(".", "")) or {}
        # networkx may keep original key
        if not data and pid in graph:
            data = graph.nodes[pid]
        title = str(data.get("title") or "")
        abstract = str(data.get("abstract") or "")
        texts.append(f"{title}\n{abstract}".strip()[:2000] or pid)

    outs: list[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = tok(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_len,
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            hs = model(**enc).last_hidden_state  # [B, T, H]
            mask = enc["attention_mask"].unsqueeze(-1).float()
            summed = (hs * mask).sum(dim=1)
            denom = mask.sum(dim=1).clamp(min=1.0)
            pooled = summed / denom
            outs.append(pooled.float().cpu().numpy())
            if (i // batch_size) % 20 == 0:
                print(f"  SciBERT {min(i + batch_size, len(texts))}/{len(texts)}", flush=True)
    return np.concatenate(outs, axis=0).astype(np.float32)


def _build_id_map(graph: nx.DiGraph) -> tuple[list[str], dict[str, int]]:
    ids = sorted(normalize_arxiv_id(str(n)) for n in graph.nodes())
    ids = [x for x in ids if x]
    id_to_idx = {pid: i for i, pid in enumerate(ids)}
    return ids, id_to_idx


def _cand_from_graph(
    graph: nx.DiGraph,
    node_by: dict[str, Any],
    pid: str,
) -> dict[str, Any] | None:
    node = node_by.get(pid)
    if node is None:
        return None
    data = graph.nodes.get(node) or {}
    return {
        "id": pid,
        "title": str(data.get("title") or data.get("Title") or ""),
        "abstract": str(data.get("abstract") or data.get("Abstract") or ""),
        "source": "gold_inject",
    }


def build_windows_for_queries(
    queries: list[dict[str, Any]],
    *,
    domain: str,
    window: int,
    id_to_idx: dict[str, int],
    X_raw: np.ndarray | None,
    shard_idx: int = 0,
    shard_count: int = 1,
    pool_mode: str = P.POOL_MODE_STRUCT,
    pool_cap: int = 0,
) -> dict[str, Any]:
    """Freeze candidate pools with φ_struct + emb_sim + gold mask.

    pool_mode=struct (default): structural Top-``window``.
    pool_mode=u_only: structural Top-K(U), K=pool_cap; **no gold injection**.
    pool_mode=full_u: DEPRECATED and invalid for evaluation — U∪golds plus a
        gold-priority cap.
    pool_mode=u_full: legacy full |U| (no Top-K); prefer u_only for trainability.
    """
    os.environ.setdefault("RR_STRUCT_COARSE", "linear")
    os.environ["RR_RANKER_WITH_EMB"] = "0"
    os.environ.pop("RR_RANKER_CE_SENT_PATH", None)
    os.environ.pop("RR_RANKER_CE_PATH", None)

    cache = get_domain_cache().get(domain)
    graph = cache.graph
    assert graph is not None
    node_by = _node_by_map(graph)
    retriever = get_retriever_service()
    dpaths = resolve_gat_domain(domain)
    struct_path = dpaths.struct_model
    if not struct_path.is_file():
        raise SystemExit(f"missing struct model for domain={domain}: {struct_path}")
    ranker = load_ranker(str(struct_path))

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

    qs = [q for i, q in enumerate(queries) if i % shard_count == shard_idx]
    mode = (pool_mode or P.POOL_MODE_STRUCT).strip().lower()
    full_u = mode == P.POOL_MODE_FULL_U
    u_only = mode == P.POOL_MODE_U_ONLY
    u_full = mode == P.POOL_MODE_U_FULL
    variable_pool = full_u or u_only or u_full

    # Build variable-length rows first, then pad to W_max or fixed window.
    rows_out: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    for qi, row in enumerate(qs):
        sid = row["source_id"]
        pack = build_rr_universe(
            row["title"],
            row["abstract"],
            graph,
            search_fn=search_fn,
            exclude_id=sid,
            multi_query=True,
            prf=True,
        )
        universe = list(pack.get("universe") or [])
        support = pack.get("support") or {}
        dense_rank = pack.get("dense_rank") or {}
        focused = set(pack.get("focused_hubs") or [])
        max_s = float(max(support.values()) if support else 1.0)
        mask_citers = masked_citer_ids(query_id=sid)
        gold = {normalize_arxiv_id(str(g)) for g in row["gold_ids"]}
        gold.discard("")
        gold.discard(normalize_arxiv_id(sid))

        q_vec = None
        if X_raw is not None and sid in id_to_idx:
            q_vec = X_raw[id_to_idx[sid]]
            qn = float(np.linalg.norm(q_vec) + 1e-8)

        seen: set[str] = set()
        cands: list[tuple[str, np.ndarray, float, bool]] = []

        def _push(cand: dict[str, Any]) -> None:
            pid = normalize_arxiv_id(str(cand.get("id") or ""))
            if not pid or pid in seen or pid not in id_to_idx:
                return
            seen.add(pid)
            idx = id_to_idx[pid]
            es = 0.0
            if q_vec is not None:
                cv = X_raw[idx]
                es = float(np.dot(q_vec, cv) / (qn * (np.linalg.norm(cv) + 1e-8)))
            feat = extract_features(
                title=row["title"],
                abstract=row["abstract"],
                cand=cand,
                support=support,
                dense_rank=dense_rank,
                focused_hubs=focused,
                graph=graph,
                node_by=node_by,
                emb_sim=es,
                max_support=max_s,
                query_id=sid,
                mask_citers=mask_citers,
            )
            cands.append((pid, feat.astype(np.float32), es, pid in gold))

        n_injected = 0
        if u_full:
            # Legacy: entire universe order; no struct Top-K; no gold inject.
            for cand in universe:
                _push(cand)
        elif u_only:
            # Fair 4.0: struct-Top-K(U); never inject golds; no gold-priority trim.
            k = int(pool_cap) if pool_cap and pool_cap > 0 else int(P.GAT_U_DEFAULT_POOL_CAP)
            k = max(1, min(k, max(len(universe), 1)))
            ranked = rank_struct_coarse(
                title=row["title"],
                abstract=row["abstract"],
                universe_pack=pack,
                graph=graph,
                n=k,
                encode=None,
                retriever=None,
                query_id=sid,
            )
            for cand in ranked[:k]:
                _push(cand)
        else:
            if full_u:
                n_rank = int(pool_cap) if pool_cap and pool_cap > 0 else max(len(universe), 1)
                n_rank = max(1, min(n_rank, max(len(universe), 1)))
            else:
                n_rank = window
            ranked = rank_struct_coarse(
                title=row["title"],
                abstract=row["abstract"],
                universe_pack=pack,
                graph=graph,
                n=n_rank,
                encode=None,
                retriever=None,
                query_id=sid,
            )
            if not full_u:
                ranked = ranked[:window]
            for cand in ranked:
                _push(cand)
            if full_u:
                for gid in sorted(gold):
                    if gid in seen:
                        continue
                    gcand = _cand_from_graph(graph, node_by, gid)
                    if gcand is None:
                        continue
                    before = len(seen)
                    _push(gcand)
                    if len(seen) > before:
                        n_injected += 1
                if pool_cap and pool_cap > 0 and len(cands) > pool_cap:
                    gold_rows = [c for c in cands if c[3]]
                    other = [c for c in cands if not c[3]]
                    budget = max(int(pool_cap) - len(gold_rows), 0)
                    cands = other[:budget] + gold_rows

        rows_out.append(
            {
                "source_id": sid,
                "cands": cands,
                "n_gold": len(gold),
                "n_injected": n_injected,
                "n_universe": len(universe),
            }
        )
        if (qi + 1) % 10 == 0 or qi == 0:
            print(
                f"  [shard {shard_idx}/{shard_count}] {qi+1}/{len(qs)} {sid} "
                f"n={len(cands)} gold_in={sum(1 for *_, g in cands if g)} "
                f"inject={n_injected} {time.perf_counter()-t0:.1f}s",
                flush=True,
            )

    q = len(rows_out)
    if variable_pool:
        W = max((len(r["cands"]) for r in rows_out), default=1)
    else:
        W = int(window)

    cand_idx = np.full((q, W), -1, dtype=np.int32)
    phi = np.zeros((q, W, 10), dtype=np.float32)
    emb_sim = np.zeros((q, W), dtype=np.float32)
    gold_mask = np.zeros((q, W), dtype=np.bool_)
    n_cands = np.zeros((q,), dtype=np.int32)
    query_ids: list[str] = []
    n_gold = np.zeros((q,), dtype=np.int32)
    n_gold_in = np.zeros((q,), dtype=np.int32)
    n_injected = np.zeros((q,), dtype=np.int32)
    n_universe = np.zeros((q,), dtype=np.int32)

    for qi, r in enumerate(rows_out):
        query_ids.append(r["source_id"])
        n_gold[qi] = int(r["n_gold"])
        n_injected[qi] = int(r["n_injected"])
        n_universe[qi] = int(r["n_universe"])
        for j, (pid, feat, es, is_g) in enumerate(r["cands"][:W]):
            cand_idx[qi, j] = id_to_idx[pid]
            phi[qi, j] = feat
            emb_sim[qi, j] = es
            if is_g:
                gold_mask[qi, j] = True
                n_gold_in[qi] += 1
            n_cands[qi] = j + 1

    # silence unused ranker bind (keeps load path warm / validates model exists)
    _ = ranker

    return {
        "query_ids": np.asarray(query_ids, dtype=object),
        "cand_idx": cand_idx,
        "phi_struct": phi,
        "emb_sim": emb_sim,
        "gold_mask": gold_mask,
        "n_cands": n_cands,
        "n_gold": n_gold,
        "n_gold_in_window": n_gold_in,
        "n_injected": n_injected,
        "n_universe": n_universe,
        "pool_mode": pool_mode,
        "pool_cap": int(pool_cap) if pool_cap else 0,
        "W": W,
        "elapsed_s": float(time.perf_counter() - t0),
    }


def fit_whiten(X: np.ndarray, *, n_components: int | None = None) -> dict[str, np.ndarray]:
    """ZCA-style whitening stats; optional PCA truncate to n_components."""
    mu = X.mean(axis=0)
    Xc = X - mu
    # covariance
    cov = (Xc.T @ Xc) / max(X.shape[0] - 1, 1)
    # eigendecomp
    evals, evecs = np.linalg.eigh(cov)
    evals = np.clip(evals, 1e-6, None)
    order = np.argsort(evals)[::-1]
    evals = evals[order]
    evecs = evecs[:, order]
    if n_components is not None and n_components < len(evals):
        evals = evals[:n_components]
        evecs = evecs[:, :n_components]
    inv_sqrt = 1.0 / np.sqrt(evals)
    W = (evecs * inv_sqrt) @ evecs.T if n_components is None else evecs * inv_sqrt
    # For PCA path W is [D, k]; for full ZCA W is [D, D]
    return {"mu": mu.astype(np.float32), "W": W.astype(np.float32), "evals": evals.astype(np.float32)}


def apply_whiten(X: np.ndarray, stats: dict[str, np.ndarray]) -> np.ndarray:
    Xc = X - stats["mu"]
    W = stats["W"]
    if W.ndim == 2 and W.shape[0] == Xc.shape[1] and W.shape[1] == Xc.shape[1]:
        return (Xc @ W).astype(np.float32)
    # PCA: W is [D, k]
    return (Xc @ W).astype(np.float32)


def pair_cos_stats(X: np.ndarray, *, n_pairs: int = 5000, seed: int = 0) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    i = rng.integers(0, n, size=n_pairs)
    j = rng.integers(0, n, size=n_pairs)
    # avoid self
    mask = i != j
    i, j = i[mask], j[mask]
    a = X[i]
    b = X[j]
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
    cos = (a * b).sum(axis=1)
    return {
        "n_pairs": int(len(cos)),
        "mean": float(cos.mean()),
        "std": float(cos.std()),
        "p10": float(np.percentile(cos, 10)),
        "p50": float(np.percentile(cos, 50)),
        "p90": float(np.percentile(cos, 90)),
    }


def build_adjacencies(
    graph: nx.DiGraph,
    id_to_idx: dict[str, int],
    *,
    cocite_topk: int = 64,
) -> dict[str, sp.csr_matrix]:
    n = len(id_to_idx)
    rows_in: list[int] = []
    cols_in: list[int] = []
    rows_out: list[int] = []
    cols_out: list[int] = []

    # map possible node key variants
    def _idx(node: Any) -> int | None:
        pid = normalize_arxiv_id(str(node))
        return id_to_idx.get(pid)

    for u, v in graph.edges():
        iu, iv = _idx(u), _idx(v)
        if iu is None or iv is None:
            continue
        # edge u→v : u cites v
        rows_out.append(iu)
        cols_out.append(iv)
        rows_in.append(iv)
        cols_in.append(iu)

    adj_out = sp.coo_matrix(
        (np.ones(len(rows_out), dtype=np.float32), (rows_out, cols_out)), shape=(n, n)
    ).tocsr()
    adj_in = sp.coo_matrix(
        (np.ones(len(rows_in), dtype=np.float32), (rows_in, cols_in)), shape=(n, n)
    ).tocsr()

    # cocite: shared citers count
    cocite_counts: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    for u in graph.nodes():
        iu = _idx(u)
        if iu is None:
            continue
        refs = []
        for v in graph.successors(u):
            iv = _idx(v)
            if iv is not None:
                refs.append(iv)
        if len(refs) < 2:
            continue
        # unique refs
        refs = list(dict.fromkeys(refs))
        m = len(refs)
        if m > 80:
            # cap combinatorial blow-up on hubs: sample pairs via star on random subset
            rng = np.random.default_rng(iu)
            refs = list(rng.choice(refs, size=80, replace=False))
            m = len(refs)
        for a in range(m):
            for b in range(a + 1, m):
                ra, rb = refs[a], refs[b]
                cocite_counts[ra][rb] += 1
                cocite_counts[rb][ra] += 1

    rows_c: list[int] = []
    cols_c: list[int] = []
    data_c: list[float] = []
    for i, nbrs in cocite_counts.items():
        top = sorted(nbrs.items(), key=lambda x: (-x[1], x[0]))[:cocite_topk]
        for j, c in top:
            rows_c.append(i)
            cols_c.append(j)
            data_c.append(float(c))
    adj_cocite = sp.coo_matrix(
        (np.asarray(data_c, dtype=np.float32), (rows_c, cols_c)), shape=(n, n)
    ).tocsr()

    return {"in": adj_in, "out": adj_out, "cocite": adj_cocite}


def build_neighbors_k(
    adj_in: sp.csr_matrix,
    adj_cocite: sp.csr_matrix,
    Xw: np.ndarray,
    *,
    k_citer: int = 16,
    k_cocite: int = 16,
    pool: int = 64,
) -> dict[str, np.ndarray]:
    """Offline N_K: citer top-16 by cos(whitened) + cocite top-16 by count; also top-64 pool."""
    n = adj_in.shape[0]
    k = k_citer + k_cocite
    # L2-normalize for cos
    xn = Xw / (np.linalg.norm(Xw, axis=1, keepdims=True) + 1e-8)

    nbr_k = np.full((n, k), -1, dtype=np.int32)
    nbr_pool = np.full((n, pool), -1, dtype=np.int32)
    nbr_src = np.zeros((n, k), dtype=np.int8)  # 1=citer, 2=cocite

    for i in range(n):
        # citers
        start, end = adj_in.indptr[i], adj_in.indptr[i + 1]
        citers = adj_in.indices[start:end]
        if len(citers):
            sims = xn[citers] @ xn[i]
            order = np.argsort(-sims)
            citers_sorted = citers[order]
        else:
            citers_sorted = np.array([], dtype=np.int32)

        # cocite by weight
        cs, ce = adj_cocite.indptr[i], adj_cocite.indptr[i + 1]
        coc = adj_cocite.indices[cs:ce]
        if len(coc):
            w = adj_cocite.data[cs:ce]
            order = np.argsort(-w)
            coc_sorted = coc[order]
        else:
            coc_sorted = np.array([], dtype=np.int32)

        chosen: list[int] = []
        srcs: list[int] = []
        seen: set[int] = set()
        for j in citers_sorted[:k_citer]:
            j = int(j)
            if j == i or j in seen:
                continue
            seen.add(j)
            chosen.append(j)
            srcs.append(1)
        for j in coc_sorted:
            if len(chosen) >= k:
                break
            j = int(j)
            if j == i or j in seen:
                continue
            seen.add(j)
            chosen.append(j)
            srcs.append(2)
        # pad from remaining citers/cocite for pool
        pool_list = list(chosen)
        for j in list(citers_sorted) + list(coc_sorted):
            if len(pool_list) >= pool:
                break
            j = int(j)
            if j == i or j in seen:
                continue
            seen.add(j)
            pool_list.append(j)

        for t, j in enumerate(chosen[:k]):
            nbr_k[i, t] = j
            nbr_src[i, t] = srcs[t]
        for t, j in enumerate(pool_list[:pool]):
            nbr_pool[i, t] = j

        if (i + 1) % 5000 == 0:
            print(f"  neighbors {i+1}/{n}", flush=True)

    return {"neighbors_k32": nbr_k, "neighbors_pool64": nbr_pool, "neighbor_src": nbr_src}


def _node_by_map(graph: nx.DiGraph) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for n in graph.nodes():
        out[normalize_arxiv_id(str(n))] = n
    return out


def _save_csr(path: Path, mat: sp.csr_matrix) -> None:
    np.savez_compressed(
        path,
        data=mat.data.astype(np.float32),
        indices=mat.indices.astype(np.int32),
        indptr=mat.indptr.astype(np.int32),
        shape=np.asarray(mat.shape, dtype=np.int64),
    )


# re-export for fold_mask
save_csr = _save_csr


def load_csr(path: Path) -> sp.csr_matrix:
    z = np.load(path)
    return sp.csr_matrix(
        (z["data"], z["indices"], z["indptr"]), shape=tuple(int(x) for x in z["shape"])
    )


def run_preprocess(args: argparse.Namespace) -> dict[str, Any]:
    root = P.root_path()
    dpaths = resolve_gat_domain(args.domain, root)
    out_rel = getattr(args, "out_dir", None) or dpaths.gat_mvp_dir
    out = _out_dir(root, out_rel)
    pool_mode = getattr(args, "pool_mode", P.POOL_MODE_STRUCT) or P.POOL_MODE_STRUCT
    encoder_dir = root / "models/base/scibert_scivocab_uncased"
    gexf = effective_nofuture_gexf(dpaths, root)
    gexf_sha = _sha256(gexf) if gexf.is_file() else ""
    if args.domain == "ewm" and gexf_sha and gexf_sha != NOFUTURE_GEXF_SHA256:
        print(f"WARN: gexf sha mismatch {gexf_sha[:16]}…", flush=True)
    elif args.domain != "ewm":
        print(f"domain={args.domain} gexf={gexf} sha={gexf_sha[:16]}…", flush=True)

    os.environ.setdefault("RR_GEXF_OVERRIDE", str(gexf))
    cache = get_domain_cache().get(args.domain)
    graph = cache.graph
    assert graph is not None

    ids, id_to_idx = _build_id_map(graph)
    # ensure graph node lookup by normalized id
    # rebuild lightweight DiGraph keyed by normalized ids if needed
    if any(normalize_arxiv_id(str(n)) != str(n) for n in list(graph.nodes())[:100]):
        g2 = nx.DiGraph()
        for n, data in graph.nodes(data=True):
            pid = normalize_arxiv_id(str(n))
            if pid:
                g2.add_node(pid, **data)
        for u, v, data in graph.edges(data=True):
            uu, vv = normalize_arxiv_id(str(u)), normalize_arxiv_id(str(v))
            if uu and vv and uu in g2 and vv in g2:
                g2.add_edge(uu, vv, **data)
        graph = g2

    report: dict[str, Any] = {
        "phase": "P1_preprocess",
        "domain": args.domain,
        "gexf": str(gexf),
        "gexf_sha256": gexf_sha,
        "split_id": SPLIT_ID,
        "n_nodes": len(ids),
        "out_dir": str(out),
        "pool_mode": pool_mode,
        "steps": {},
    }

    id_map_path = out / "id_map.json"
    if args.step in ("all", "id_map", "encode", "whiten", "adj", "neighbors", "windows"):
        id_map_path.write_text(
            json.dumps(
                {"ids": ids, "split_id": SPLIT_ID, "gexf_sha256": gexf_sha},
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"wrote {id_map_path} n={len(ids)}", flush=True)

    X_path = out / "node_scibert.npy"
    if args.step in ("all", "encode"):
        t0 = time.perf_counter()
        X = encode_scibert(
            graph, ids, model_dir=encoder_dir, batch_size=args.batch_size
        )
        np.save(X_path, X)
        report["steps"]["encode"] = {
            "path": str(X_path),
            "shape": list(X.shape),
            "elapsed_s": float(time.perf_counter() - t0),
        }
        print(f"wrote {X_path} {X.shape}", flush=True)
    else:
        X = np.load(X_path) if X_path.is_file() else None

    whiten_path = out / "whiten.npz"
    if args.step in ("all", "whiten"):
        assert X is not None
        t0 = time.perf_counter()
        raw_cos = pair_cos_stats(X, n_pairs=5000, seed=0)
        must_whiten = raw_cos["mean"] > 0.9 or raw_cos["std"] < 0.05
        stats = fit_whiten(X)  # full ZCA
        Xw = apply_whiten(X, stats)
        whiten_cos = pair_cos_stats(Xw, n_pairs=5000, seed=0)
        np.savez_compressed(
            whiten_path,
            mu=stats["mu"],
            W=stats["W"],
            evals=stats["evals"],
            raw_cos_mean=np.array([raw_cos["mean"]]),
            raw_cos_std=np.array([raw_cos["std"]]),
            whiten_cos_mean=np.array([whiten_cos["mean"]]),
            whiten_cos_std=np.array([whiten_cos["std"]]),
        )
        # also save whitened vectors for train convenience
        np.save(out / "node_scibert_whitened.npy", Xw)
        report["steps"]["whiten"] = {
            "path": str(whiten_path),
            "must_whiten": must_whiten,
            "raw_cos": raw_cos,
            "whiten_cos": whiten_cos,
            "elapsed_s": float(time.perf_counter() - t0),
        }
        print(
            f"whiten: raw_cos mean={raw_cos['mean']:.4f} std={raw_cos['std']:.4f} "
            f"→ whitened mean={whiten_cos['mean']:.4f} std={whiten_cos['std']:.4f} "
            f"must_whiten={must_whiten}",
            flush=True,
        )
    else:
        Xw = None
        if (out / "node_scibert_whitened.npy").is_file():
            Xw = np.load(out / "node_scibert_whitened.npy")
        elif X is not None and whiten_path.is_file():
            z = np.load(whiten_path)
            Xw = apply_whiten(X, {"mu": z["mu"], "W": z["W"]})

    if args.step in ("all", "adj"):
        t0 = time.perf_counter()
        adjs = build_adjacencies(graph, id_to_idx, cocite_topk=64)
        for name, mat in adjs.items():
            p = out / f"adj_{name}.npz"
            _save_csr(p, mat)
            print(f"wrote {p} nnz={mat.nnz}", flush=True)
        report["steps"]["adj"] = {
            "nnz_in": int(adjs["in"].nnz),
            "nnz_out": int(adjs["out"].nnz),
            "nnz_cocite": int(adjs["cocite"].nnz),
            "elapsed_s": float(time.perf_counter() - t0),
        }

    if args.step in ("all", "neighbors"):
        assert Xw is not None
        t0 = time.perf_counter()
        adj_in = load_csr(out / "adj_in.npz")
        adj_cocite = load_csr(out / "adj_cocite.npz")
        nbr = build_neighbors_k(adj_in, adj_cocite, Xw)
        np.savez_compressed(out / "neighbors_k32.npz", **nbr)
        report["steps"]["neighbors"] = {
            "path": str(out / "neighbors_k32.npz"),
            "elapsed_s": float(time.perf_counter() - t0),
        }
        print(f"wrote neighbors_k32.npz", flush=True)

    if args.step in ("all", "windows"):
        which = args.windows
        jsonl = dpaths.test_jsonl if which == "test" else dpaths.train_jsonl
        if not jsonl.is_file():
            raise SystemExit(f"missing jsonl for windows: {jsonl}")
        queries = _load_unique_queries(jsonl)
        if args.max_samples > 0:
            queries = queries[: args.max_samples]
        t0 = time.perf_counter()
        # Prefer SciBERT from this out_dir; ewm-only legacy fall back to shared encode.
        if X is None:
            mvp_x = out / "node_scibert.npy"
            if (
                not mvp_x.is_file()
                and args.domain == "ewm"
                and (root / P.GAT_MVP_CKPT_DIR / "node_scibert.npy").is_file()
            ):
                mvp_x = root / P.GAT_MVP_CKPT_DIR / "node_scibert.npy"
            if mvp_x.is_file():
                X = np.load(mvp_x)
        pack = build_windows_for_queries(
            queries,
            domain=args.domain,
            window=args.window,
            id_to_idx=id_to_idx,
            X_raw=X,
            shard_idx=args.shard_idx,
            shard_count=args.shard_count,
            pool_mode=pool_mode,
            pool_cap=int(getattr(args, "pool_cap", 0) or 0),
        )
        meta_split = dpaths.split_dir / "split_meta.json"
        n_train, n_test = 3423, 381
        if meta_split.is_file():
            sm = json.loads(meta_split.read_text(encoding="utf-8"))
            n_train = int(sm.get("n_train_sources") or n_train)
            n_test = int(sm.get("n_test_sources") or n_test)
        tag = f"test{n_test}" if which == "test" else f"train{n_train}"
        if pool_mode == P.POOL_MODE_FULL_U:
            tag = f"{tag}_fullU"
        elif pool_mode in (P.POOL_MODE_U_ONLY, P.POOL_MODE_U_FULL):
            tag = f"{tag}_U"
        if args.shard_count > 1:
            path = out / f"windows_{tag}.shard{args.shard_idx}of{args.shard_count}.npz"
        else:
            path = out / f"windows_{tag}.npz"
        W = int(pack["W"])
        n_c = pack["n_cands"].astype(np.float64)
        save_kw: dict[str, Any] = {
            "query_ids": pack["query_ids"],
            "cand_idx": pack["cand_idx"],
            "phi_struct": pack["phi_struct"],
            "emb_sim": pack["emb_sim"],
            "gold_mask": pack["gold_mask"],
            "n_cands": pack["n_cands"],
            "n_gold": pack["n_gold"],
            "n_gold_in_window": pack["n_gold_in_window"],
            "gexf_sha256": np.asarray([gexf_sha]),
            "split_id": np.asarray([SPLIT_ID]),
            "window": np.asarray([W]),
            "pool_mode": np.asarray([pool_mode]),
            "pool_cap": np.asarray([int(pack.get("pool_cap") or 0)]),
        }
        if "n_injected" in pack:
            save_kw["n_injected"] = pack["n_injected"]
            save_kw["n_universe"] = pack["n_universe"]
        np.savez_compressed(path, **save_kw)
        recall = float(
            (pack["n_gold_in_window"] / np.maximum(pack["n_gold"], 1)).mean()
        ) if len(pack["n_gold"]) else 0.0
        report["steps"]["windows"] = {
            "path": str(path),
            "which": which,
            "pool_mode": pool_mode,
            "W": W,
            "n_queries": int(len(pack["query_ids"])),
            "mean_n_cands": float(n_c.mean()) if len(n_c) else None,
            "p50_n_cands": float(np.median(n_c)) if len(n_c) else None,
            "p95_n_cands": float(np.percentile(n_c, 95)) if len(n_c) else None,
            "max_n_cands": int(n_c.max()) if len(n_c) else None,
            "mean_gold_in_window": float(pack["n_gold_in_window"].mean())
            if len(pack["n_gold_in_window"])
            else None,
            "pool_gold_recall": recall,
            "mean_injected": float(pack["n_injected"].mean())
            if "n_injected" in pack and len(pack["n_injected"])
            else None,
            "elapsed_s": float(time.perf_counter() - t0),
        }
        print(
            f"wrote {path} W={W} mean_n={report['steps']['windows']['mean_n_cands']} "
            f"max_n={report['steps']['windows']['max_n_cands']} "
            f"pool_gold_recall={recall:.4f}",
            flush=True,
        )

    report_path = out / "preprocess_report.json"
    # merge with existing
    if report_path.is_file():
        try:
            prev = json.loads(report_path.read_text(encoding="utf-8"))
            if isinstance(prev, dict) and "steps" in prev:
                merged = dict(prev)
                merged_steps = dict(prev.get("steps") or {})
                merged_steps.update(report.get("steps") or {})
                merged.update({k: v for k, v in report.items() if k != "steps"})
                merged["steps"] = merged_steps
                report = merged
        except Exception:  # noqa: BLE001
            pass
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {report_path}", flush=True)
    return report


def merge_window_shards(
    out_dir: Path,
    *,
    tag: str,
    shard_count: int,
) -> Path:
    parts = []
    for i in range(shard_count):
        p = out_dir / f"windows_{tag}.shard{i}of{shard_count}.npz"
        if not p.is_file():
            raise FileNotFoundError(p)
        parts.append(np.load(p, allow_pickle=True))
    qids = np.concatenate([p["query_ids"] for p in parts])
    order = np.argsort(qids.astype(str))

    W = max(int(p["cand_idx"].shape[1]) for p in parts)

    def pad_cat(key: str, fill) -> np.ndarray:
        chunks = []
        for p in parts:
            arr = p[key]
            if arr.ndim >= 2 and arr.shape[1] < W:
                pad_width = [(0, 0)] * arr.ndim
                pad_width[1] = (0, W - arr.shape[1])
                arr = np.pad(arr, pad_width, constant_values=fill)
            chunks.append(arr)
        return np.concatenate(chunks, axis=0)[order]

    dest = out_dir / f"windows_{tag}.npz"
    save_kw: dict[str, Any] = {
        "query_ids": np.concatenate([p["query_ids"] for p in parts])[order],
        "cand_idx": pad_cat("cand_idx", -1),
        "phi_struct": pad_cat("phi_struct", 0),
        "emb_sim": pad_cat("emb_sim", 0),
        "gold_mask": pad_cat("gold_mask", False),
        "n_cands": np.concatenate([p["n_cands"] for p in parts])[order],
        "n_gold": np.concatenate([p["n_gold"] for p in parts])[order],
        "n_gold_in_window": np.concatenate([p["n_gold_in_window"] for p in parts])[order],
        "gexf_sha256": parts[0]["gexf_sha256"],
        "split_id": parts[0]["split_id"],
        "window": np.asarray([W]),
    }
    if "pool_mode" in parts[0]:
        save_kw["pool_mode"] = parts[0]["pool_mode"]
    if "n_injected" in parts[0]:
        save_kw["n_injected"] = np.concatenate([p["n_injected"] for p in parts])[order]
        save_kw["n_universe"] = np.concatenate([p["n_universe"] for p in parts])[order]
    np.savez_compressed(dest, **save_kw)
    n_c = save_kw["n_cands"].astype(np.float64)
    recall = float(
        (save_kw["n_gold_in_window"] / np.maximum(save_kw["n_gold"], 1)).mean()
    )
    print(
        f"merged → {dest} W={W} n={len(save_kw['query_ids'])} "
        f"mean_n={n_c.mean():.1f} p95={np.percentile(n_c,95):.1f} max={int(n_c.max())} "
        f"pool_gold_recall={recall:.4f}",
        flush=True,
    )
    return dest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="GAT MVP / fullU preprocess")
    ap.add_argument("--domain", default="ewm")
    ap.add_argument(
        "--step",
        choices=(
            "all",
            "encode",
            "whiten",
            "adj",
            "neighbors",
            "windows",
            "merge_windows",
            "folds",
        ),
        default="all",
    )
    ap.add_argument("--windows", choices=("test", "train"), default="test")
    ap.add_argument("--window", type=int, default=P.STRUCT_WINDOW)
    ap.add_argument(
        "--pool-mode",
        choices=(
            P.POOL_MODE_STRUCT,
            P.POOL_MODE_FULL_U,
            P.POOL_MODE_U_ONLY,
            P.POOL_MODE_U_FULL,
        ),
        default=P.POOL_MODE_STRUCT,
    )
    ap.add_argument(
        "--pool-cap",
        type=int,
        default=0,
        help="u_only: struct-Top-K size (0→GAT_U_DEFAULT_POOL_CAP); full_u: gold-priority cap",
    )
    ap.add_argument(
        "--out-dir",
        default=None,
        help="Relative to RWCITE_ROOT (default gat_mvp; u_only/u_full → gat_U)",
    )
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--shard-idx", type=int, default=0)
    ap.add_argument("--shard-count", type=int, default=1)
    ap.add_argument("--merge-tag", default="train3423")
    args = ap.parse_args(argv)

    if args.step == "merge_windows":
        root = P.root_path()
        dpaths = resolve_gat_domain(args.domain, root)
        out_rel = args.out_dir or dpaths.gat_mvp_dir
        out = _out_dir(root, out_rel)
        dest = merge_window_shards(out, tag=args.merge_tag, shard_count=args.shard_count)
        print(f"merged → {dest}", flush=True)
        return 0

    if args.step == "folds":
        from rwcite.gat.fold_mask import build_fold_adjacencies

        root = P.root_path()
        dpaths = resolve_gat_domain(args.domain, root)
        out_rel = args.out_dir or dpaths.gat_mvp_dir
        out = _out_dir(root, out_rel)
        meta_split = dpaths.split_dir / "split_meta.json"
        n_train = 3423
        if meta_split.is_file():
            sm = json.loads(meta_split.read_text(encoding="utf-8"))
            n_train = int(sm.get("n_train_sources") or n_train)
        win_name = f"windows_train{n_train}.npz"
        if args.pool_mode == P.POOL_MODE_FULL_U:
            win_name = f"windows_train{n_train}_fullU.npz"
        elif args.pool_mode in (P.POOL_MODE_U_ONLY, P.POOL_MODE_U_FULL):
            win_name = f"windows_train{n_train}_U.npz"
        report = build_fold_adjacencies(
            out,
            id_map_path=out / "id_map.json",
            windows_train_path=out / win_name,
            k=4,
            seed=0,
        )
        print(json.dumps(report["report"], indent=2), flush=True)
        return 0

    run_preprocess(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
