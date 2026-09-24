from __future__ import annotations

import math
from statistics import mean
from typing import Any, Iterable


def metrics_at_k(ranked_ids: Iterable[str], gold_ids: Iterable[str], k: int = 10) -> dict[str, float]:
    ranked = list(ranked_ids)[:k]
    gold = set(gold_ids)
    hits = sum(pid in gold for pid in ranked)
    reciprocal_rank = next((1.0 / rank for rank, pid in enumerate(ranked, 1) if pid in gold), 0.0)
    dcg = sum((1.0 if pid in gold else 0.0) / math.log2(rank + 1) for rank, pid in enumerate(ranked, 1))
    ideal_hits = min(k, len(gold))
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return {
        f"hits_at_{k}": float(hits),
        f"precision_at_{k}": hits / k if k else 0.0,
        f"recall_at_{k}": hits / len(gold) if gold else 0.0,
        f"mrr_at_{k}": reciprocal_rank,
        f"ndcg_at_{k}": dcg / idcg if idcg else 0.0,
    }


def summarize(rows: list[dict[str, Any]], metric_names: list[str]) -> dict[str, float | int]:
    result: dict[str, float | int] = {"n_queries": len(rows)}
    for name in metric_names:
        values = [float(row[name]) for row in rows if row.get(name) is not None]
        result[name] = mean(values) if values else 0.0
    return result
