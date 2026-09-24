#!/usr/bin/env python3
"""Evaluate one or more Top-10 prediction files with a single metric implementation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.ewm_top10.common import parse_month, read_jsonl, write_jsonl
from experiments.ewm_top10.metrics import metrics_at_k, summarize


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", default="outputs/ewm_top10/data/test.jsonl")
    parser.add_argument("--corpus", default="outputs/ewm_top10/data/corpus.jsonl")
    parser.add_argument("--predictions", nargs="+", required=True)
    parser.add_argument("--out-dir", default="outputs/ewm_top10/metrics")
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()

    queries = {row["query_id"]: row for row in read_jsonl(args.test)}
    corpus = {row["paper_id"]: row for row in read_jsonl(args.corpus)}
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    names = [f"{name}_at_{args.k}" for name in ("hits", "precision", "recall", "mrr", "ndcg")]
    names += ["r_at_10_full", "mrr", "recall_at_30", "recall_at_100", "recall_at_400", "recall_at_1000"]

    for prediction_path_raw in args.predictions:
        prediction_path = Path(prediction_path_raw)
        rows = []
        seen = set()
        method = prediction_path.stem
        for prediction in read_jsonl(prediction_path):
            qid = prediction["query_id"]
            if qid in seen or qid not in queries:
                raise SystemExit(f"duplicate or unknown query ID in {prediction_path}: {qid}")
            seen.add(qid)
            ranked = prediction.get("ranked_ids") or []
            if len(ranked) < args.k or len(set(ranked)) != len(ranked):
                raise SystemExit(f"expected at least {args.k} unique IDs for {qid} in {prediction_path}")
            qmonth = parse_month(queries[qid]["published_at"])
            if qmonth is None:
                raise SystemExit(f"invalid query date for {qid}")
            for pid in ranked:
                candidate_month = parse_month(corpus[pid]["published_at"]) if pid in corpus else None
                if candidate_month is None or candidate_month >= qmonth:
                    raise SystemExit(f"ineligible prediction {pid} for {qid} in {prediction_path}")
            gold = queries[qid]["gold_ids"]
            full_mrr = next((1.0 / rank for rank, pid in enumerate(ranked, 1) if pid in set(gold)), 0.0)
            recall_by_depth = {
                f"recall_at_{depth}": (
                    sum(pid in set(gold) for pid in ranked[:depth]) / len(gold)
                    if len(ranked) >= depth and gold
                    else None
                )
                for depth in (30, 100, 400, 1000)
            }
            top_metrics = metrics_at_k(ranked, gold, args.k)
            row = {
                "query_id": qid,
                "method": prediction.get("method") or method,
                "ranked_ids": ranked,
                "gold_ids": gold,
                "latency_ms": prediction.get("latency_ms"),
                "r_at_10_full": top_metrics[f"recall_at_{args.k}"],
                "mrr": full_mrr,
                **recall_by_depth,
                **top_metrics,
            }
            rows.append(row)
        missing = set(queries) - seen
        if missing:
            raise SystemExit(f"{prediction_path} is missing {len(missing)} test queries")
        available_names = [
            name for name in names + ["latency_ms"] if any(row.get(name) is not None for row in rows)
        ]
        summary = summarize(rows, available_names)
        summary.update({"method": rows[0]["method"] if rows else method, "prediction_file": str(prediction_path)})
        write_jsonl(out_dir / f"{prediction_path.stem}_per_query.jsonl", rows)
        (out_dir / f"{prediction_path.stem}_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        summaries.append(summary)
    (out_dir / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
