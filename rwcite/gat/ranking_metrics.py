"""Binary IR metrics for GAT / L0 window rankings.

Relevance is 0/1 (``gold_mask``). Gate metric remains Hits@k (count);
nDCG / MRR / R@k are reporting supplements.

R@k uses **in-window gold** as the denominator (ranking quality inside the
scored pool). The reference framework scores a Top-400 window, so R@400 is
window-capped and R@1000 is not defined under this protocol.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# Recommended L0 reporting set (no R@1000 on 400-cand windows).
DEFAULT_RECALL_KS = (30, 100, 400)
DEFAULT_NDCG_K = 10
DEFAULT_MRR_K = 10


def hits_at_k(scores: np.ndarray, gold: np.ndarray, k: int) -> int:
    """|Top-k ∩ G| count."""
    if scores.size == 0:
        return 0
    top = np.asarray(scores, dtype=np.float64).argsort()[::-1][: max(1, int(k))]
    g = np.asarray(gold, dtype=np.float64)
    return int(g[top].sum())


def recall_at_k(scores: np.ndarray, gold: np.ndarray, k: int) -> float:
    """R@k = |Top-k ∩ G_window| / |G_window| (0 if no in-window gold)."""
    g = np.asarray(gold, dtype=np.float64)
    n_rel = float(g.sum())
    if n_rel <= 0.0:
        return 0.0
    return float(hits_at_k(scores, g, k) / n_rel)


def mrr_at_k(scores: np.ndarray, gold: np.ndarray, k: int) -> float:
    """MRR@k: reciprocal rank of the first relevant in Top-k (else 0)."""
    if scores.size == 0:
        return 0.0
    k = max(1, int(k))
    order = np.asarray(scores, dtype=np.float64).argsort()[::-1][:k]
    g = np.asarray(gold, dtype=np.float64)
    hits = np.flatnonzero(g[order] > 0.0)
    if hits.size == 0:
        return 0.0
    return float(1.0 / (int(hits[0]) + 1))


def ndcg_at_k(scores: np.ndarray, gold: np.ndarray, k: int) -> float:
    """Binary nDCG@k (rel ∈ {0,1})."""
    if scores.size == 0:
        return 0.0
    k = max(1, int(k))
    order = np.asarray(scores, dtype=np.float64).argsort()[::-1][:k]
    g = np.asarray(gold, dtype=np.float64)
    rel = g[order]
    discounts = 1.0 / np.log2(np.arange(2, rel.size + 2, dtype=np.float64))
    dcg = float((rel * discounts).sum())
    n_ideal = int(min(k, int(g.sum())))
    if n_ideal <= 0:
        return 0.0
    idcg = float(discounts[:n_ideal].sum())
    return float(dcg / idcg) if idcg > 0.0 else 0.0


def query_ir_metrics(
    scores: np.ndarray,
    gold: np.ndarray,
    *,
    recall_ks: tuple[int, ...] = DEFAULT_RECALL_KS,
    ndcg_k: int = DEFAULT_NDCG_K,
    mrr_k: int = DEFAULT_MRR_K,
) -> dict[str, float | int]:
    """Per-query Hits + recommended IR supplements."""
    g = np.asarray(gold, dtype=np.float64)
    sc = np.asarray(scores, dtype=np.float64)
    out: dict[str, float | int] = {
        "hits_at_10": hits_at_k(sc, g, 10),
        "hits_at_30": hits_at_k(sc, g, 30),
        "ndcg_at_10": ndcg_at_k(sc, g, ndcg_k),
        "mrr_at_10": mrr_at_k(sc, g, mrr_k),
    }
    for kk in recall_ks:
        out[f"recall_at_{int(kk)}"] = recall_at_k(sc, g, int(kk))
    return out


def mean_ir_fields(per_query: list[dict[str, Any]]) -> dict[str, float]:
    """Mean of IR fields over non-skipped queries (zeros if empty)."""
    keys = (
        "hits_at_10",
        "hits_at_30",
        "ndcg_at_10",
        "mrr_at_10",
        "recall_at_30",
        "recall_at_100",
        "recall_at_400",
    )
    rows = [r for r in per_query if not r.get("skip")]
    n = max(len(rows), 1)
    out: dict[str, float] = {}
    for k in keys:
        out[f"mean_{k}"] = float(sum(float(r.get(k) or 0.0) for r in rows) / n)
    out["n_scored"] = float(len(rows))
    return out
