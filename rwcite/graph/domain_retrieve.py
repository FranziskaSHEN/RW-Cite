"""Domain corpus retrieve: multi-prototype merge + year window + dual gate (N_max ∧ τ).

Used by ``rwcite.cli.graph_pipeline``.

Gate order: **year window → rank ≤ N_max ∧ score ≥ τ**. Older classics are
expected to enter the graph as citees, not as retrieval seeds.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


def _load_embedder(embedder_name: str):
    import torch
    from transformers import AutoModel, AutoTokenizer

    load_kwargs = {"local_files_only": True}
    try:
        tokenizer = AutoTokenizer.from_pretrained(embedder_name, **load_kwargs)
        model = AutoModel.from_pretrained(embedder_name, **load_kwargs)
    except OSError:
        tokenizer = AutoTokenizer.from_pretrained(embedder_name)
        model = AutoModel.from_pretrained(embedder_name)
    return tokenizer, model.to(device="cuda", dtype=torch.float16)


def _encode_queries(
    tokenizer, model, queries: Sequence[str]
) -> np.ndarray:
    import torch

    texts = [q.strip() for q in queries if q and str(q).strip()]
    if not texts:
        raise ValueError("No non-empty queries for domain retrieve")
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs.to("cuda"))
        embs = outputs.last_hidden_state[:, 0, :].float().cpu().numpy()
    return embs


def _load_candidate_matrix(
    config: dict[str, Any], paper_list: list[str]
) -> np.ndarray:
    from rwcite.retrieve.retriever.corpus_utils import load_merged_embeddings

    use_hf = bool(config["retriever"].get("load_arxiv_embeds", True))
    merged_ids, all_candidate_embs = load_merged_embeddings(
        use_hf_embeds=use_hf,
        cache_dir="datasets/topic_level_embeds",
    )
    if merged_ids != paper_list:
        id_to_emb = dict(zip(merged_ids, all_candidate_embs))
        all_candidate_embs = np.stack([id_to_emb[pid] for pid in paper_list])
    return np.stack(all_candidate_embs)


def arxiv_year(pid: str) -> int | None:
    """Year from modern arXiv id ``YYMM.xxxxx``; old ``archive/NNNNN`` → None."""
    s = str(pid).strip()
    if "/" in s:
        return None
    head = s.split(".")[0]
    if len(head) == 4 and head.isdigit():
        yy, mm = int(head[:2]), int(head[2:])
        if not (1 <= mm <= 12):
            return None
        return 2000 + yy if yy < 80 else 1900 + yy
    return None


def resolve_min_arxiv_year(
    ret_cfg: dict[str, Any],
    *,
    recent_years: int | None = None,
    min_arxiv_year: int | None = None,
    as_of_year: int | None = None,
) -> int | None:
    """Return absolute min year, or None to disable year gate."""
    if min_arxiv_year is not None:
        return int(min_arxiv_year)
    if ret_cfg.get("min_arxiv_year") is not None:
        return int(ret_cfg["min_arxiv_year"])
    ry = recent_years
    if ry is None and ret_cfg.get("recent_years") is not None:
        ry = int(ret_cfg["recent_years"])
    if ry is None or int(ry) <= 0:
        return None
    ref = as_of_year or datetime.now(timezone.utc).year
    # e.g. 2026 with recent_years=10 → keep year >= 2016
    return int(ref) - int(ry)


def apply_year_window(
    paper_list: Sequence[str],
    scores: np.ndarray,
    *,
    min_year: int | None,
    drop_unknown_year: bool = True,
) -> tuple[list[str], np.ndarray, dict[str, Any]]:
    """Filter candidates before N∧τ. Unknown-year ids dropped by default (old-style)."""
    if min_year is None:
        return list(paper_list), np.asarray(scores, dtype=float), {
            "enabled": False,
            "min_year": None,
            "n_in": len(paper_list),
            "n_out": len(paper_list),
            "n_dropped_old": 0,
            "n_dropped_unknown": 0,
        }
    kept_ids: list[str] = []
    kept_sc: list[float] = []
    n_old = n_unk = 0
    for pid, sc in zip(paper_list, scores):
        y = arxiv_year(pid)
        if y is None:
            n_unk += 1
            if drop_unknown_year:
                continue
            kept_ids.append(pid)
            kept_sc.append(float(sc))
            continue
        if y < min_year:
            n_old += 1
            continue
        kept_ids.append(pid)
        kept_sc.append(float(sc))
    meta = {
        "enabled": True,
        "min_year": int(min_year),
        "drop_unknown_year": bool(drop_unknown_year),
        "n_in": len(paper_list),
        "n_out": len(kept_ids),
        "n_dropped_old": n_old,
        "n_dropped_unknown": n_unk if drop_unknown_year else 0,
    }
    return kept_ids, np.asarray(kept_sc, dtype=float), meta


def merge_prototype_scores(
    query_embs: np.ndarray, candidate_embs: np.ndarray
) -> np.ndarray:
    """Per-paper best cosine across prototypes (rows of query_embs)."""
    from sklearn.metrics.pairwise import cosine_similarity

    sims = cosine_similarity(query_embs, candidate_embs)  # (Q, N)
    return sims.max(axis=0)


DEFAULT_SCORE_TAU_MARGIN = 0.05


def default_tau(
    sorted_scores_desc: Sequence[float],
    head_k: int = 200,
    *,
    margin: float = DEFAULT_SCORE_TAU_MARGIN,
) -> float:
    """τ = median(score[head_k]) − margin (DOMAIN §2.3.2; default margin=0.05)."""
    if not sorted_scores_desc:
        return 0.0
    head = list(sorted_scores_desc[: min(head_k, len(sorted_scores_desc))])
    return float(np.median(head) - float(margin))


def apply_dual_gate(
    paper_list: Sequence[str],
    scores: np.ndarray,
    *,
    n_max: int,
    tau: float | None = None,
    apply_tau: bool = True,
    head_k: int = 200,
    tau_margin: float = DEFAULT_SCORE_TAU_MARGIN,
) -> tuple[list[tuple[str, float]], dict[str, Any]]:
    """Keep ids with rank < n_max AND (optional) score ≥ τ; preserve score order."""
    order = np.argsort(-scores)
    ranked = [(paper_list[i], float(scores[i])) for i in order]
    ranked_scores = [s for _, s in ranked]
    margin = float(tau_margin)
    if tau is None and apply_tau:
        tau = default_tau(ranked_scores, head_k=head_k, margin=margin)
    kept: list[tuple[str, float]] = []
    for rank, (pid, sc) in enumerate(ranked):
        if rank >= n_max:
            break
        if apply_tau and tau is not None and sc < tau:
            continue
        kept.append((pid, sc))
    meta = {
        "n_max": int(n_max),
        "apply_tau": bool(apply_tau),
        "tau": float(tau) if tau is not None else None,
        "tau_margin": margin,
        "head_k": int(head_k),
        "n_before_gate": len(ranked),
        "n_after_gate": len(kept),
        "score_head": ranked_scores[0] if ranked_scores else None,
        "score_at_n_max": ranked_scores[min(n_max, len(ranked_scores)) - 1]
        if ranked_scores and n_max > 0
        else None,
    }
    return kept, meta


def retrieve_domain(
    queries: Sequence[str],
    config_path: str | Path,
    *,
    n_max: int | None = None,
    tau: float | None = None,
    apply_tau: bool | None = None,
    recent_years: int | None = None,
    min_arxiv_year: int | None = None,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Score corpus; year window then dual-gate; return ordered {id: score} + meta."""
    from rwcite.retrieve.retriever.corpus_utils import build_paper_list
    from rwcite.latex.utils import read_yaml_file

    config = read_yaml_file(str(config_path))
    ret_cfg = config.get("retriever") or {}
    if n_max is None:
        n_max = int(ret_cfg.get("num_retrievals", 5000))
    if apply_tau is None:
        apply_tau = bool(ret_cfg.get("apply_score_tau", True))
    if tau is None and ret_cfg.get("score_tau") is not None:
        tau = float(ret_cfg["score_tau"])
    tau_margin = float(
        ret_cfg.get("score_tau_margin", DEFAULT_SCORE_TAU_MARGIN)
    )

    q_list = [str(q).strip() for q in queries if str(q).strip()]
    if not q_list and ret_cfg.get("queries"):
        q_list = [str(q).strip() for q in ret_cfg["queries"] if str(q).strip()]
    if not q_list and ret_cfg.get("query"):
        q_list = [str(ret_cfg["query"]).strip()]
    if not q_list:
        raise ValueError("No queries provided (CLI / retriever.queries / retriever.query)")

    min_year = resolve_min_arxiv_year(
        ret_cfg, recent_years=recent_years, min_arxiv_year=min_arxiv_year
    )

    embedder = ret_cfg["embedder"]
    tokenizer, model = _load_embedder(embedder)
    paper_list = build_paper_list(cache_dir="datasets/arxiv_topics")
    cand = _load_candidate_matrix(config, paper_list)
    q_embs = _encode_queries(tokenizer, model, q_list)
    best = merge_prototype_scores(q_embs, cand)

    filtered_ids, filtered_scores, year_meta = apply_year_window(
        paper_list, best, min_year=min_year
    )
    kept, gate_meta = apply_dual_gate(
        filtered_ids,
        filtered_scores,
        n_max=n_max,
        tau=tau,
        apply_tau=apply_tau,
        tau_margin=tau_margin,
    )
    out = {pid: sc for pid, sc in kept}
    meta = {
        "queries": q_list,
        "n_prototypes": len(q_list),
        "embedder": embedder,
        "year_window": year_meta,
        "gate": gate_meta,
        "gate_order": ["year_window", "N_max_and_tau"],
    }
    return out, meta


def retrieve_and_write(
    queries: Sequence[str],
    retrieval_nodes_path: str | Path,
    config_path: str | Path,
    *,
    n_max: int | None = None,
    tau: float | None = None,
    apply_tau: bool | None = None,
    recent_years: int | None = None,
    min_arxiv_year: int | None = None,
    write_bool_compat: bool = False,
) -> tuple[list[str], dict[str, Any]]:
    """Write retrieval_nodes JSON (values = scores unless write_bool_compat)."""
    results, meta = retrieve_domain(
        queries,
        config_path,
        n_max=n_max,
        tau=tau,
        apply_tau=apply_tau,
        recent_years=recent_years,
        min_arxiv_year=min_arxiv_year,
    )
    path = Path(retrieval_nodes_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any]
    if write_bool_compat:
        payload = {pid: True for pid in results}
    else:
        payload = results
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    meta_path = path.with_name(path.stem + "_retrieve_meta.json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    meta["retrieval_nodes_path"] = str(path)
    meta["retrieve_meta_path"] = str(meta_path)
    return list(results.keys()), meta


def normalize_queries_arg(
    query: str | None,
    queries: Iterable[str] | None,
    config: dict[str, Any] | None = None,
) -> list[str]:
    """CLI ``--query`` / ``--queries`` + config ``retriever.queries``."""
    out: list[str] = []
    if queries:
        for q in queries:
            q = str(q).strip()
            if q:
                out.append(q)
    if query and str(query).strip():
        parts = [p.strip() for p in str(query).split("|") if p.strip()]
        for p in parts:
            if p not in out:
                out.append(p)
    if not out and config:
        ret = config.get("retriever") or {}
        for q in ret.get("queries") or []:
            q = str(q).strip()
            if q and q not in out:
                out.append(q)
        if not out and ret.get("query"):
            out.append(str(ret["query"]).strip())
    return out
