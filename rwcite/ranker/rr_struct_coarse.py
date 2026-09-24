"""Standard struct coarse ranker: linear ``model.npz`` by default.

Pipeline (v1 linear / AS-aligned):
  train: weighted logistic BCE with emb_sim filled
  infer: ``RR_STRUCT_COARSE=linear`` → rank_universe(model, emb_sims=None)

Fallbacks (RR_STRUCT_COARSE=auto): linear → logistic+emb → mlp → heuristic.
"""

from __future__ import annotations

import os
from typing import Any, Callable

import numpy as np

from rwcite.ranker.rr_ranker import extract_features, load_ranker, rank_universe
from rwcite.ranker.rr_ranker_mlp import load_mlp_ranker, pack_vector


def rr_struct_coarse_params() -> dict[str, Any]:
    def _i(name: str, default: int) -> int:
        raw = (os.environ.get(name) or "").strip()
        if not raw:
            return default
        try:
            return max(1, int(raw))
        except ValueError:
            return default

    mode = (os.environ.get("RR_STRUCT_COARSE") or "linear").strip().lower()
    if mode not in ("auto", "mlp", "logistic", "linear", "heuristic"):
        mode = "linear"
    full_u = (os.environ.get("RR_RANKER_FULL_EMB") or "0").strip().lower() not in (
        "0",
        "false",
        "no",
    )
    return {
        "mode": mode,
        "prefilter": _i("RR_RANKER_STRUCT_PREFILTER", 800),
        "logistic_prefilter": _i("RR_RANKER_LOGISTIC_PREFILTER", 500),
        "full_u": full_u,
    }


def encode_texts(retriever: Any, texts: list[str]) -> np.ndarray:
    """Batch-encode texts with the domain embedder (normalized)."""
    import torch

    retriever._load_embedder()
    tok, emb = retriever._tokenizer, retriever._embedder
    device = next(emb.parameters()).device
    outs: list[np.ndarray] = []
    for i in range(0, len(texts), 32):
        batch = texts[i : i + 32]
        inputs = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=256)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            v = emb(**inputs).last_hidden_state[:, 0, :].float()
            v = torch.nn.functional.normalize(v, dim=-1)
        outs.append(v.cpu().numpy())
    return np.concatenate(outs, 0)


def _resolve_mode(
    mode: str,
    *,
    mlp: Any,
    model: Any,
    encode: Callable[[list[str]], np.ndarray] | None,
) -> str:
    if mode != "auto":
        return mode
    # Prefer domain linear model.npz (standard); MLP/logistic are explicit overrides.
    if model is not None:
        return "linear"
    if mlp is not None and encode is not None:
        return "mlp"
    return "heuristic"


def _rank_mlp(
    *,
    title: str,
    abstract: str,
    universe_pack: dict[str, Any],
    graph: Any,
    encode: Callable[[list[str]], np.ndarray],
    mlp: Any,
    n: int,
    prefilter: int,
    full_u: bool,
    prefilter_model: Any = None,
    query_id: str = "",
) -> list[dict[str, Any]]:
    universe = list(universe_pack.get("universe") or [])
    if not universe:
        return []
    if not full_u:
        coarse = rank_universe(
            title=title,
            abstract=abstract,
            universe_pack=universe_pack,
            graph=graph,
            n=min(prefilter, len(universe)),
            emb_sims=None,
            model=prefilter_model,
            query_id=query_id,
        )
        id_set = {c["id"] for c in coarse}
        universe = [c for c in universe if c["id"] in id_set]

    qv = encode([f"{title}\n{abstract}".strip()[:800]])[0]
    texts = [f"{c.get('title') or ''}\n{c.get('abstract') or ''}"[:800] for c in universe]
    cvs = encode(texts)
    max_s = max([float(v) for v in (universe_pack.get("support") or {}).values()], default=1.0)
    scored: list[tuple[float, dict[str, Any]]] = []
    for c, cv in zip(universe, cvs):
        feat = extract_features(
            title=title,
            abstract=abstract,
            cand=c,
            support=universe_pack.get("support") or {},
            dense_rank=universe_pack.get("dense_rank") or {},
            focused_hubs=universe_pack.get("focused_hubs") or set(),
            graph=graph,
            node_by=universe_pack.get("node_by") or {},
            emb_sim=float(cv @ qv),
            max_support=max_s,
            query_id=query_id,
        )
        vec = pack_vector(feat, qv, cv)
        sc = mlp.score_vec(vec)
        scored.append(
            (
                sc,
                {
                    "id": c["id"],
                    "title": c.get("title") or "",
                    "abstract": c.get("abstract") or "",
                    "source": c.get("source") or "ranked",
                    "score": sc,
                },
            )
        )
    scored.sort(key=lambda x: -x[0])
    return [c for _, c in scored[:n]]


def _rank_logistic_emb(
    *,
    title: str,
    abstract: str,
    universe_pack: dict[str, Any],
    graph: Any,
    encode: Callable[[list[str]], np.ndarray],
    model: Any,
    n: int,
    prefilter: int,
    query_id: str = "",
) -> list[dict[str, Any]]:
    coarse = rank_universe(
        title=title,
        abstract=abstract,
        universe_pack=universe_pack,
        graph=graph,
        n=min(prefilter, len(universe_pack.get("universe") or [])),
        emb_sims=None,
        model=None,
        query_id=query_id,
    )
    ids = [c["id"] for c in coarse]
    by = {c["id"]: c for c in universe_pack.get("universe") or []}
    qv = encode([f"{title}\n{abstract}".strip()[:800]])[0]
    texts = [
        f"{by[i].get('title') or ''}\n{by[i].get('abstract') or ''}"[:800]
        for i in ids
        if i in by
    ]
    emb_sims: dict[str, float] = {}
    if texts:
        pv = encode(texts)
        emb_sims = {i: float(v @ qv) for i, v in zip(ids, pv)}
    return rank_universe(
        title=title,
        abstract=abstract,
        universe_pack=universe_pack,
        graph=graph,
        n=n,
        emb_sims=emb_sims,
        model=model,
        query_id=query_id,
    )


def rank_struct_coarse(
    *,
    title: str,
    abstract: str,
    universe_pack: dict[str, Any],
    graph: Any,
    n: int,
    encode: Callable[[list[str]], np.ndarray] | None = None,
    retriever: Any | None = None,
    query_id: str = "",
) -> list[dict[str, Any]]:
    """Score universe struct layer and return top-N (standard Ranker coarse path)."""
    params = rr_struct_coarse_params()
    mlp = load_mlp_ranker()
    model = load_ranker()

    enc = encode
    if enc is None and retriever is not None:
        try:
            enc = lambda texts: encode_texts(retriever, texts)  # noqa: E731
        except Exception:  # noqa: BLE001
            enc = None
    if enc is None and params["mode"] in ("auto", "mlp", "logistic"):
        try:
            from rwcite.runtime.deps import get_retriever_service

            retriever = get_retriever_service()
            enc = lambda texts: encode_texts(retriever, texts)  # noqa: E731
        except Exception:  # noqa: BLE001
            enc = None

    mode = _resolve_mode(params["mode"], mlp=mlp, model=model, encode=enc)

    if mode == "mlp" and mlp is not None and enc is not None:
        return _rank_mlp(
            title=title,
            abstract=abstract,
            universe_pack=universe_pack,
            graph=graph,
            encode=enc,
            mlp=mlp,
            n=n,
            prefilter=params["prefilter"],
            full_u=params["full_u"],
            prefilter_model=None,
            query_id=query_id,
        )
    if mode == "logistic" and model is not None and enc is not None:
        return _rank_logistic_emb(
            title=title,
            abstract=abstract,
            universe_pack=universe_pack,
            graph=graph,
            encode=enc,
            model=model,
            n=n,
            prefilter=params["logistic_prefilter"],
            query_id=query_id,
        )
    if mode == "linear" and model is not None:
        return rank_universe(
            title=title,
            abstract=abstract,
            universe_pack=universe_pack,
            graph=graph,
            n=n,
            emb_sims=None,
            model=model,
            query_id=query_id,
        )
    return rank_universe(
        title=title,
        abstract=abstract,
        universe_pack=universe_pack,
        graph=graph,
        n=n,
        emb_sims=None,
        model=None,
        query_id=query_id,
    )
