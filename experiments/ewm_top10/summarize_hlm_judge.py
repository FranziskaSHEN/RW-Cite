#!/usr/bin/env python3
"""Summarize HLM API validity, cost, latency, and cross-run stability."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from statistics import mean

import numpy as np

from experiments.ewm_top10.common import read_jsonl


def usage_total(row: dict, key: str) -> int:
    total = 0
    for stage in ("analyzer", "decider"):
        usage = (row.get("usage") or {}).get(stage) or {}
        total += int(usage.get(key) or 0)
    return total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", nargs="+", required=True)
    parser.add_argument("--input-price-per-million", type=float, default=0.0)
    parser.add_argument("--output-price-per-million", type=float, default=0.0)
    parser.add_argument(
        "--cost-status",
        choices=("priced", "not_provided", "no_charge"),
        default="priced",
        help="whether supplied prices are billable, unavailable, or institutionally free",
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    runs = [list(read_jsonl(path)) for path in args.predictions]
    if not runs or not runs[0]:
        raise SystemExit("no HLM predictions")
    id_sets = [{row["query_id"] for row in run} for run in runs]
    if any(ids != id_sets[0] for ids in id_sets[1:]):
        raise SystemExit("HLM runs do not contain identical query IDs")

    rows = [row for run in runs for row in run]
    prompt_tokens = sum(usage_total(row, "prompt_tokens") for row in rows)
    completion_tokens = sum(usage_total(row, "completion_tokens") for row in rows)
    total_cost = None
    if args.cost_status != "not_provided":
        total_cost = (
            prompt_tokens * args.input_price_per_million
            + completion_tokens * args.output_price_per_million
        ) / 1_000_000.0
    latencies = [float(row.get("latency_ms") or 0.0) for row in rows]

    pairwise_jaccard = []
    if len(runs) > 1:
        indexed = [{row["query_id"]: row for row in run} for run in runs]
        for left, right in itertools.combinations(indexed, 2):
            for qid in sorted(left):
                a, b = set(left[qid]["ranked_ids"]), set(right[qid]["ranked_ids"])
                pairwise_jaccard.append(len(a & b) / len(a | b) if a or b else 1.0)

    result = {
        "n_runs": len(runs),
        "n_queries_per_run": len(runs[0]),
        "n_requests_expected": 2 * len(rows),
        "analyzer_failure_rate": mean(not bool(row.get("analyzer_ok")) for row in rows),
        "decider_failure_rate": mean(not bool(row.get("parse_ok")) for row in rows),
        "fallback_rate": mean(bool(row.get("padded")) for row in rows),
        "mean_latency_ms_per_query": mean(latencies),
        "p95_latency_ms_per_query": float(np.quantile(latencies, 0.95)),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "estimated_cost": total_cost,
        "cost_status": args.cost_status,
        "input_price_per_million": args.input_price_per_million,
        "output_price_per_million": args.output_price_per_million,
        "mean_pairwise_top10_jaccard": (
            mean(pairwise_jaccard) if pairwise_jaccard else None
        ),
        "prediction_files": args.predictions,
    }
    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
