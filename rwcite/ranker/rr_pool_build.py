"""Shared helper: build Top-N RR pool with universe + ranker (+ optional emb / MLP)."""

from __future__ import annotations

import os
import time
from typing import Any, Callable

import numpy as np

from rwcite.ranker.rr_ranker import (
    load_ranker,
    rank_universe,
    rr_ranker_params,
)
from rwcite.ranker.rr_ranker_ce import load_ce_ranker, rank_universe_ce, rr_ce_params
from rwcite.ranker.rr_ranker_ce_sent import (
    load_ce_sent_ranker,
    rank_universe_ce_sent,
    rr_ce_sent_params,
)
from rwcite.ranker.rr_ranker_cite_link import (
    load_citelink_ranker,
    rank_universe_citelink,
    rr_citelink_params,
)
from rwcite.ranker.rr_ranker_fusion import (
    load_fusion_ranker,
    rank_universe_fusion,
    rr_fusion_params,
)
from rwcite.ranker.rr_ranker_sent import (
    load_sent_index,
    rank_universe_sent,
    rr_sent_params,
)
from rwcite.ranker.rr_ranker_ltr import (
    compute_emb_sims,
    load_ltr_ranker,
    rank_universe_ltr,
    rr_ltr_params,
)
from rwcite.ranker.rr_ranker_mlp import load_mlp_ranker
from rwcite.ranker.rr_struct_coarse import encode_texts, rank_struct_coarse, rr_struct_coarse_params
from rwcite.ranker.rr_universe import build_rr_universe


def _encode_factory(retriever) -> Callable[[list[str]], np.ndarray] | None:
    try:
        retriever._load_embedder()
    except Exception:  # noqa: BLE001
        return None

    def encode(texts: list[str]) -> np.ndarray:
        return encode_texts(retriever, texts)

    return encode


_R1A_CACHE: Any = None
_R1A_PATH: str | None = None


def _maybe_r1a_rerank(ranked: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Optional LightGBM LambdaRank second head (RR_RANKER_R1A_PATH)."""
    global _R1A_CACHE, _R1A_PATH
    path = (os.environ.get("RR_RANKER_R1A_PATH") or "").strip()
    if not path or not ranked:
        return ranked
    try:
        from pathlib import Path

        import lightgbm as lgb

        p = Path(path)
        if not p.is_absolute():
            repo = Path(__file__).resolve().parents[3]
            for cand in (Path(path), Path.cwd() / path, repo / path):
                if cand.is_file() or (cand.is_dir() and (cand / "model.txt").is_file()):
                    p = cand.resolve()
                    break
        model_file = p if p.is_file() else p / "model.txt"
        if not model_file.is_file():
            return ranked
        key = str(model_file)
        if _R1A_CACHE is None or _R1A_PATH != key:
            _R1A_CACHE = lgb.Booster(model_file=str(model_file))
            _R1A_PATH = key
        booster = _R1A_CACHE
        n = len(ranked)
        X = np.zeros((n, 6), dtype=np.float32)
        for i, c in enumerate(ranked):
            sc = float(c.get("score") or 0.0)
            ce = float(c["ce_score"]) if c.get("ce_score") is not None else sc
            cl = float(c["citelink_score"]) if c.get("citelink_score") is not None else 0.0
            X[i, 0] = sc
            X[i, 1] = float(i)
            X[i, 2] = ce
            X[i, 3] = cl
            X[i, 4] = 1.0 / (1.0 + float(i))
            X[i, 5] = ce - cl
        scores = booster.predict(X)
        order = np.argsort(-np.asarray(scores, dtype=np.float32))
        out = []
        for i in order:
            c = dict(ranked[int(i)])
            c["score"] = float(scores[int(i)])
            c["source"] = (c.get("source") or "ce_sent") + "+r1a"
            out.append(c)
        return out
    except Exception:  # noqa: BLE001
        return ranked


def _rank_with_mlp(
    title: str,
    abstract: str,
    pack: dict[str, Any],
    graph: Any,
    encode: Callable[[list[str]], np.ndarray],
    mlp,
    n: int,
    full_u: bool,
) -> list[dict[str, Any]]:
    """Legacy wrapper; prefer rank_struct_coarse()."""
    params = rr_struct_coarse_params()
    from rwcite.ranker.rr_struct_coarse import _rank_mlp

    return _rank_mlp(
        title=title,
        abstract=abstract,
        universe_pack=pack,
        graph=graph,
        encode=encode,
        mlp=mlp,
        n=n,
        prefilter=params["prefilter"],
        full_u=full_u,
        prefilter_model=None,
    )


def build_ranked_pool(
    title: str,
    abstract: str,
    graph: Any,
    *,
    search_fn: Callable[[str, int], list[dict[str, Any]]],
    exclude_id: str = "",
    n: int | None = None,
    retriever: Any = None,
    with_emb: bool | None = None,
    domain_id: str = "",
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Full cold-start path used by inference and eval scripts."""
    t_total0 = time.perf_counter()
    timings: dict[str, float] = {}
    params = rr_ranker_params()
    n_out = int(n if n is not None else params["n"])
    domain_id = (domain_id or "").strip()
    t0 = time.perf_counter()
    pack = build_rr_universe(
        title,
        abstract,
        graph,
        search_fn=search_fn,
        exclude_id=exclude_id,
        multi_query=True,
        prf=True,
    )
    timings["universe_s"] = float(time.perf_counter() - t0)
    use_emb = with_emb
    if use_emb is None:
        use_emb = (os.environ.get("RR_RANKER_WITH_EMB") or "1").strip().lower() not in (
            "0",
            "false",
            "no",
        )

    ce_sent = load_ce_sent_ranker()
    ce = load_ce_ranker()
    sent_index = load_sent_index()
    fusion = load_fusion_ranker()
    citelink = load_citelink_ranker()
    ltr = load_ltr_ranker()
    mlp = load_mlp_ranker()
    encode = _encode_factory(retriever) if (use_emb and retriever is not None) else None

    if ce_sent is not None:
        sp = rr_ce_sent_params()
        win = int(sp["window"])
        coarse_mode = str(sp.get("coarse") or "struct")
        need_cl = (
            float(sp["citelink_blend"]) > 0
            or coarse_mode in ("citelink", "union")
        )
        q_emb = None
        cand_embs = None
        cl_model = citelink if need_cl else None
        if cl_model is not None and encode is not None and retriever is not None:
            t0 = time.perf_counter()
            q_emb = encode([f"{title}\n{abstract}".strip()[:800]])[0]
            timings["query_encode_s"] = float(time.perf_counter() - t0)
            # citelink/union coarse needs embeddings for (almost) full U
            t0 = time.perf_counter()
            if coarse_mode in ("citelink", "union"):
                ids = [c["id"] for c in (pack.get("universe") or []) if c.get("id")]
            else:
                coarse_probe = rank_struct_coarse(
                    title=title,
                    abstract=abstract,
                    universe_pack=pack,
                    graph=graph,
                    n=min(win, len(pack.get("universe") or [])),
                    encode=encode,
                    retriever=retriever,
                    query_id=exclude_id,
                )
                ids = [c["id"] for c in coarse_probe]
            cand_embs = retriever._ensure_domain_embeddings(domain_id, ids)
            timings["cand_embed_s"] = float(time.perf_counter() - t0)
        rank_timings: dict[str, float] = {}
        ranked = rank_universe_ce_sent(
            title=title,
            abstract=abstract,
            universe_pack=pack,
            graph=graph,
            ce=ce_sent,
            n=n_out,
            window=win,
            exclude_id=exclude_id,
            citelink=cl_model,
            cand_embs=cand_embs,
            q_emb=q_emb,
            citelink_blend=float(sp["citelink_blend"]),
            coarse_mode=coarse_mode,
            score_m=int(sp.get("score_m") or win),
            encode=encode,
            retriever=retriever,
            timings_out=rank_timings,
        )
        timings.update(rank_timings)
        ranker_name = f"ce_sent_{coarse_mode}_m{int(sp.get('score_m') or win)}"
        # Optional R1a listwise second head
        t0 = time.perf_counter()
        ranked = _maybe_r1a_rerank(ranked)
        timings["r1a_rerank_s"] = float(time.perf_counter() - t0)
    elif ce is not None:
        ce_p = rr_ce_params()
        universe = list(pack.get("universe") or [])
        pre_n = min(int(ce_p["prefilter"]), len(universe)) if universe else 0
        coarse = rank_universe(
            title=title,
            abstract=abstract,
            universe_pack=pack,
            graph=graph,
            n=pre_n if pre_n else 1,
            emb_sims=None,
            model=None,
            query_id=exclude_id,
        )
        struct_ranks = {c["id"]: i for i, c in enumerate(coarse)}
        if ce_p["full_u"] or len(universe) <= int(ce_p["prefilter"]):
            cands = universe
            # structure ranks only known for coarse; others get large rank
            for i, c in enumerate(universe):
                struct_ranks.setdefault(c["id"], pre_n + i)
        else:
            id_set = set(struct_ranks)
            cands = [c for c in universe if c["id"] in id_set]
        ranked = rank_universe_ce(
            title,
            abstract,
            cands,
            ce,
            n=n_out,
            batch_size=int(ce_p["batch_size"]),
            struct_blend=float(ce_p["struct_blend"]),
            struct_ranks=struct_ranks,
        )
        ranker_name = "ce"
    elif sent_index is not None and encode is not None and retriever is not None:
        sp = rr_sent_params()
        win = int(sp["window"])
        coarse_probe = rank_universe(
            title=title,
            abstract=abstract,
            universe_pack=pack,
            graph=graph,
            n=min(win, len(pack.get("universe") or [])),
            emb_sims=None,
            model=None,
            query_id=exclude_id,
        )
        ids = [c["id"] for c in coarse_probe]
        q_emb = encode([f"{title}\n{abstract}".strip()[:800]])[0]
        cand_embs = None
        cl_model = citelink if float(sp["citelink_blend"]) > 0 else None
        if cl_model is not None:
            cand_embs = retriever._ensure_domain_embeddings(domain_id, ids)
        ranked = rank_universe_sent(
            title=title,
            abstract=abstract,
            universe_pack=pack,
            graph=graph,
            sent_index=sent_index,
            q_emb=q_emb,
            n=n_out,
            window=win,
            exclude_id=exclude_id,
            citelink=cl_model,
            cand_embs=cand_embs,
            citelink_blend=float(sp["citelink_blend"]),
        )
        ranker_name = "sent"
    elif (
        fusion is not None
        and citelink is not None
        and encode is not None
        and retriever is not None
    ):
        fu_p = rr_fusion_params()
        win = int(fu_p["window"])
        coarse_probe = rank_universe(
            title=title,
            abstract=abstract,
            universe_pack=pack,
            graph=graph,
            n=min(win, len(pack.get("universe") or [])),
            emb_sims=None,
            model=None,
            query_id=exclude_id,
        )
        ids = [c["id"] for c in coarse_probe]
        q_emb = encode([f"{title}\n{abstract}".strip()[:800]])[0]
        cand_embs = retriever._ensure_domain_embeddings(domain_id, ids)
        ranked = rank_universe_fusion(
            title=title,
            abstract=abstract,
            universe_pack=pack,
            graph=graph,
            citelink=citelink,
            fusion_booster=fusion,
            q_emb=q_emb,
            cand_embs=cand_embs,
            n=n_out,
            window=win,
        )
        ranker_name = "fusion"
    elif citelink is not None and encode is not None and retriever is not None:
        cl_p = rr_citelink_params()
        win = int(cl_p["window"])
        coarse_probe = rank_universe(
            title=title,
            abstract=abstract,
            universe_pack=pack,
            graph=graph,
            n=min(win, len(pack.get("universe") or [])),
            emb_sims=None,
            model=None,
            query_id=exclude_id,
        )
        ids = [c["id"] for c in coarse_probe]
        q_emb = encode([f"{title}\n{abstract}".strip()[:800]])[0]
        cand_embs = retriever._ensure_domain_embeddings(domain_id, ids)
        ranked = rank_universe_citelink(
            title=title,
            abstract=abstract,
            universe_pack=pack,
            graph=graph,
            model=citelink,
            q_emb=q_emb,
            cand_embs=cand_embs,
            n=n_out,
            window=win,
            struct_blend=float(cl_p["struct_blend"]),
        )
        ranker_name = "citelink"
    elif ltr is not None:
        ltr_p = rr_ltr_params()
        win = int(ltr_p["window"])
        # coarse ids for emb (structure top-window); reuse ranker's coarse inside
        coarse_probe = rank_universe(
            title=title,
            abstract=abstract,
            universe_pack=pack,
            graph=graph,
            n=min(win, len(pack.get("universe") or [])),
            emb_sims=None,
            model=None,
            query_id=exclude_id,
        )
        emb_sims: dict[str, float] = {}
        if encode is not None and retriever is not None:
            emb_sims = compute_emb_sims(
                retriever=retriever,
                domain_id=domain_id,
                title=title,
                abstract=abstract,
                cand_ids=[c["id"] for c in coarse_probe],
                encode=encode,
            )
        ranked = rank_universe_ltr(
            title=title,
            abstract=abstract,
            universe_pack=pack,
            graph=graph,
            booster=ltr,
            n=n_out,
            window=win,
            emb_sims=emb_sims,
        )
        ranker_name = "ltr"
    else:
        ranked = rank_struct_coarse(
            title=title,
            abstract=abstract,
            universe_pack=pack,
            graph=graph,
            n=n_out,
            encode=encode,
            retriever=retriever,
            query_id=exclude_id,
        )
        if mlp is not None and encode is not None:
            ranker_name = "mlp"
        elif load_ranker() is not None and encode is not None:
            ranker_name = "logistic"
        elif load_ranker() is not None:
            ranker_name = "linear"
        else:
            ranker_name = "heuristic"

    if "rank_total_s" not in timings:
        timings["rank_total_s"] = float(
            time.perf_counter() - t_total0 - float(timings.get("universe_s") or 0.0)
        )
    timings["total_s"] = float(time.perf_counter() - t_total0)

    out = [
        {
            "id": c["id"],
            "title": c.get("title") or "",
            "abstract": c.get("abstract") or "",
            "score": c.get("score"),
            "ce_score": c.get("ce_score"),
            "citelink_score": c.get("citelink_score"),
            "source": c.get("source") or "ranked",
        }
        for c in ranked
    ]
    meta = {
        "cand_mode": "universe_ranker",
        "universe_size": pack.get("meta", {}).get("universe_size", 0),
        "pool_size": len(out),
        "with_emb": bool(encode is not None),
        "ranker": ranker_name,
        "universe_ids": [c["id"] for c in pack.get("universe") or []],
        "timings": timings,
    }
    return out, meta
