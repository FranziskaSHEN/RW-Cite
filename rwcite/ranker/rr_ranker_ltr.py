"""Graph-conditioned listwise LTR for RR pool (LightGBM LambdaRank).

v2: fill emb_sim from domain embeddings; add orthogonal seed/rank features.
Activate via RR_RANKER_LTR_PATH=/path/to/model.txt
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Callable

import numpy as np

from rwcite.ranker.reference_recommend import normalize_arxiv_id
from rwcite.ranker.rr_ranker import (
    FEATURE_NAMES,
    extract_features,
    rank_universe,
)

LTR_EXTRA_FEATURES = [
    "common_nbr_seed",
    "in_2hop_seed",
    "ppr_seed",
    "log_seed_cite",
    "cites_seed",
    "common_succ_seed",
    "struct_rank_n",
    "emb_x_lex",
]

LTR_FEATURE_NAMES = FEATURE_NAMES + LTR_EXTRA_FEATURES


def rr_ltr_params() -> dict[str, Any]:
    def _i(name: str, default: int) -> int:
        raw = (os.environ.get(name) or "").strip()
        if not raw:
            return default
        try:
            return max(1, int(raw))
        except ValueError:
            return default

    path = (os.environ.get("RR_RANKER_LTR_PATH") or "").strip()
    return {
        "path": path,
        "window": _i("RR_LTR_WINDOW", 400),
        "seed_cap": _i("RR_LTR_SEED_CAP", 64),
        "ppr_steps": _i("RR_LTR_PPR_STEPS", 20),
    }


def _resolve_path(path: str) -> Path | None:
    if not path:
        return None
    cands = [Path(path)]
    if not Path(path).is_absolute():
        here = Path(__file__).resolve()
        repo = here.parents[3]
        cands.append(repo / path)
        cands.append(Path.cwd() / path)
    for c in cands:
        if c.is_file():
            return c.resolve()
    return None


_LTR_CACHE: Any = None
_LTR_PATH: str | None = None
_LTR_NAMES: list[str] | None = None


def load_ltr_ranker(path: str | None = None):
    """Load LightGBM Booster; returns None if path unset/missing."""
    global _LTR_CACHE, _LTR_PATH, _LTR_NAMES
    p = path if path is not None else rr_ltr_params()["path"]
    resolved = _resolve_path(p) if p else None
    if resolved is None:
        return None
    key = str(resolved)
    if _LTR_CACHE is not None and _LTR_PATH == key:
        return _LTR_CACHE
    import lightgbm as lgb

    booster = lgb.Booster(model_file=str(resolved))
    meta_path = resolved.with_suffix(".meta.json")
    if not meta_path.is_file():
        meta_path = resolved.parent / "meta.json"
    names = list(LTR_FEATURE_NAMES)
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        names = list(meta.get("feature_names") or names)
    _LTR_CACHE = booster
    _LTR_PATH = key
    _LTR_NAMES = names
    return booster


def compute_emb_sims(
    *,
    retriever: Any,
    domain_id: str,
    title: str,
    abstract: str,
    cand_ids: list[str],
    encode: Callable[[list[str]], np.ndarray] | None = None,
) -> dict[str, float]:
    """Cosine(query, cand) via domain embedding cache (+ optional encode for query)."""
    if retriever is None or not cand_ids:
        return {}
    try:
        emb_map = retriever._ensure_domain_embeddings(domain_id, cand_ids)
    except Exception:  # noqa: BLE001
        emb_map = {}
    qv = None
    if encode is not None:
        try:
            qv = encode([f"{title}\n{abstract}".strip()[:800]])[0]
        except Exception:  # noqa: BLE001
            qv = None
    if qv is None:
        # fallback: mean of dense seeds if present in emb_map — weak
        return {pid: 0.0 for pid in cand_ids}
    qv = np.asarray(qv, dtype=np.float32)
    qn = float(np.linalg.norm(qv))
    if qn < 1e-9:
        return {pid: 0.0 for pid in cand_ids}
    qv = qv / qn
    out: dict[str, float] = {}
    for pid in cand_ids:
        v = emb_map.get(pid)
        if v is None:
            out[pid] = 0.0
            continue
        vv = np.asarray(v, dtype=np.float32)
        vn = float(np.linalg.norm(vv))
        if vn < 1e-9:
            out[pid] = 0.0
        else:
            out[pid] = float(np.dot(qv, vv / vn))
    return out


def _seed_set(pack: dict[str, Any], seed_cap: int) -> set[str]:
    focused = pack.get("focused_hubs") or set()
    dense = pack.get("dense_ids") or []
    seeds: list[str] = []
    for pid in list(focused) + list(dense):
        pid = normalize_arxiv_id(str(pid))
        if pid and pid not in seeds:
            seeds.append(pid)
        if len(seeds) >= seed_cap:
            break
    return set(seeds)


def _node(node_by: dict[str, Any], pid: str):
    return node_by.get(normalize_arxiv_id(pid))


def _pred_ids(graph: Any, node_by: dict[str, Any], pid: str, cap: int = 80) -> set[str]:
    node = _node(node_by, pid)
    if node is None or graph is None:
        return set()
    out: set[str] = set()
    try:
        for u in graph.predecessors(node):
            out.add(normalize_arxiv_id(str(u)))
            if len(out) >= cap:
                break
    except Exception:  # noqa: BLE001
        return out
    return out


def _succ_ids(graph: Any, node_by: dict[str, Any], pid: str, cap: int = 80) -> set[str]:
    node = _node(node_by, pid)
    if node is None or graph is None:
        return set()
    out: set[str] = set()
    try:
        for v in graph.successors(node):
            out.add(normalize_arxiv_id(str(v)))
            if len(out) >= cap:
                break
    except Exception:  # noqa: BLE001
        return out
    return out


def _build_seed_nbrs(
    graph: Any, node_by: dict[str, Any], seeds: set[str]
) -> tuple[set[str], set[str], dict[str, float]]:
    """Return (seed_pred_union, seed_succ_union, seed_cite_weight on citees)."""
    pred_u: set[str] = set()
    succ_u: set[str] = set()
    cite_w: dict[str, float] = {}
    for s in seeds:
        preds = _pred_ids(graph, node_by, s, 60)
        succs = _succ_ids(graph, node_by, s, 80)
        pred_u |= preds
        succ_u |= succs
        node = _node(node_by, s)
        od = 1
        if node is not None and graph is not None:
            try:
                od = max(1, int(graph.out_degree(node)))
            except Exception:  # noqa: BLE001
                od = 1
        w = 1.0 / math.log1p(od)
        for t in succs:
            cite_w[t] = cite_w.get(t, 0.0) + w
    return pred_u, succ_u, cite_w


def _approx_ppr(
    graph: Any,
    node_by: dict[str, Any],
    seeds: set[str],
    targets: list[str],
    *,
    damping: float = 0.85,
    steps: int = 20,
) -> dict[str, float]:
    """Lightweight PPR on seeds ∪ targets ∪ one-hop expansions (capped)."""
    if not seeds or not targets:
        return {t: 0.0 for t in targets}
    nodes: list[str] = []
    seen: set[str] = set()
    for pid in list(seeds) + list(targets):
        pid = normalize_arxiv_id(pid)
        if pid and pid not in seen:
            seen.add(pid)
            nodes.append(pid)
    extra: list[str] = []
    for s in list(seeds)[:48]:
        for nbr in list(_succ_ids(graph, node_by, s, 40)) + list(
            _pred_ids(graph, node_by, s, 20)
        ):
            if nbr not in seen:
                seen.add(nbr)
                extra.append(nbr)
            if len(extra) >= 600:
                break
        if len(extra) >= 600:
            break
    nodes.extend(extra)
    idx = {p: i for i, p in enumerate(nodes)}
    n = len(nodes)
    if n == 0:
        return {t: 0.0 for t in targets}

    outs: list[list[int]] = [[] for _ in range(n)]
    for pid in nodes:
        i = idx[pid]
        for v in _succ_ids(graph, node_by, pid, 60):
            j = idx.get(v)
            if j is not None:
                outs[i].append(j)

    personal = np.zeros(n, dtype=np.float64)
    for s in seeds:
        if s in idx:
            personal[idx[s]] = 1.0
    if personal.sum() <= 0:
        return {t: 0.0 for t in targets}
    personal /= personal.sum()

    scores = personal.copy()
    for _ in range(max(1, steps)):
        nxt = np.zeros(n, dtype=np.float64)
        for i, dests in enumerate(outs):
            if not dests:
                nxt += scores[i] * personal
                continue
            share = scores[i] / len(dests)
            for j in dests:
                nxt[j] += share
        scores = damping * nxt + (1.0 - damping) * personal
    return {t: float(scores[idx[t]]) if t in idx else 0.0 for t in targets}


def extract_ltr_features(
    *,
    title: str,
    abstract: str,
    cand: dict[str, Any],
    pack: dict[str, Any],
    graph: Any,
    emb_sim: float = 0.0,
    max_support: float = 1.0,
    seed_pred_u: set[str] | None = None,
    seed_succ_u: set[str] | None = None,
    seeds: set[str] | None = None,
    cite_w: dict[str, float] | None = None,
    ppr: dict[str, float] | None = None,
    struct_rank: int = 0,
    window: int = 400,
) -> np.ndarray:
    base = extract_features(
        title=title,
        abstract=abstract,
        cand=cand,
        support=pack.get("support") or {},
        dense_rank=pack.get("dense_rank") or {},
        focused_hubs=pack.get("focused_hubs") or set(),
        graph=graph,
        node_by=pack.get("node_by") or {},
        emb_sim=emb_sim,
        max_support=max_support,
    )
    # lex is FEATURE_NAMES[2]
    lex = float(base[2]) if len(base) > 2 else 0.0
    pid = cand["id"]
    node_by = pack.get("node_by") or {}
    seed_pred_u = seed_pred_u or set()
    seed_succ_u = seed_succ_u or set()
    seeds = seeds or set()
    cite_w = cite_w or {}
    ppr = ppr or {}

    c_pred = _pred_ids(graph, node_by, pid, 80)
    c_succ = _succ_ids(graph, node_by, pid, 80)
    common = len(c_pred & seed_pred_u) if c_pred and seed_pred_u else 0
    in_2hop = 1.0 if (c_pred & seed_succ_u) else 0.0
    if in_2hop < 1.0 and seed_succ_u:
        for mid in list(seed_succ_u)[:40]:
            if pid in _succ_ids(graph, node_by, mid, 40):
                in_2hop = 1.0
                break
    ppr_v = float(ppr.get(pid, 0.0))
    seed_cite = float(cite_w.get(pid, 0.0))
    cites_seed = len(c_succ & seeds) if c_succ and seeds else 0
    common_succ = len(c_succ & seed_succ_u) if c_succ and seed_succ_u else 0
    win = max(1, int(window))
    struct_rank_n = float(struct_rank) / float(win)
    emb_x_lex = float(emb_sim) * lex
    extra = np.array(
        [
            min(common, 20) / 20.0,
            in_2hop,
            math.log1p(ppr_v * 1000.0),
            math.log1p(seed_cite),
            min(cites_seed, 10) / 10.0,
            min(common_succ, 20) / 20.0,
            struct_rank_n,
            emb_x_lex,
        ],
        dtype=np.float32,
    )
    return np.concatenate([base, extra]).astype(np.float32)


def graded_label(pid: str, gold: set[str], cite_w: dict[str, float]) -> float:
    if pid not in gold:
        return 0.0
    if float(cite_w.get(pid, 0.0)) > 0:
        return 4.0
    return 2.0


def rank_universe_ltr(
    *,
    title: str,
    abstract: str,
    universe_pack: dict[str, Any],
    graph: Any,
    booster: Any,
    n: int = 30,
    window: int | None = None,
    emb_sims: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    """Structure coarse window → LTR re-rank → Top-n."""
    params = rr_ltr_params()
    win = int(window if window is not None else params["window"])
    coarse = rank_universe(
        title=title,
        abstract=abstract,
        universe_pack=universe_pack,
        graph=graph,
        n=min(win, len(universe_pack.get("universe") or [])),
        emb_sims=None,
        model=None,
    )
    if not coarse:
        return []
    seeds = _seed_set(universe_pack, int(params["seed_cap"]))
    node_by = universe_pack.get("node_by") or {}
    seed_pred_u, seed_succ_u, cite_w = _build_seed_nbrs(graph, node_by, seeds)
    targets = [c["id"] for c in coarse]
    ppr = _approx_ppr(
        graph,
        node_by,
        seeds,
        targets,
        steps=int(params["ppr_steps"]),
    )
    max_s = max(
        [float(v) for v in (universe_pack.get("support") or {}).values()], default=1.0
    )
    emb_sims = emb_sims or {}
    X = np.stack(
        [
            extract_ltr_features(
                title=title,
                abstract=abstract,
                cand=c,
                pack=universe_pack,
                graph=graph,
                emb_sim=float(emb_sims.get(c["id"], 0.0)),
                max_support=max_s,
                seed_pred_u=seed_pred_u,
                seed_succ_u=seed_succ_u,
                seeds=seeds,
                cite_w=cite_w,
                ppr=ppr,
                struct_rank=ri,
                window=win,
            )
            for ri, c in enumerate(coarse)
        ]
    )
    # If booster trained with older feature dim, pad/truncate safely
    n_feat = int(getattr(booster, "num_feature", lambda: X.shape[1])())
    if X.shape[1] != n_feat:
        if X.shape[1] > n_feat:
            X = X[:, :n_feat]
        else:
            X = np.pad(X, ((0, 0), (0, n_feat - X.shape[1])), mode="constant")
    scores = booster.predict(X)
    order = np.argsort(-scores)
    out: list[dict[str, Any]] = []
    for i in order[:n]:
        c = coarse[int(i)]
        out.append(
            {
                "id": c["id"],
                "title": c.get("title") or "",
                "abstract": c.get("abstract") or "",
                "source": c.get("source") or "ltr",
                "score": float(scores[int(i)]),
            }
        )
    return out


def save_ltr_meta(out_dir: Path, feature_names: list[str], extra: dict | None = None) -> None:
    meta = {
        "kind": "lgbm_lambdarank_rr_ltr_v2",
        "feature_names": feature_names,
        **(extra or {}),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
