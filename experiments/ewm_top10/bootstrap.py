#!/usr/bin/env python3
"""Paired query bootstrap for two per-query evaluation files."""

from __future__ import annotations

import argparse
import json

import numpy as np

from experiments.ewm_top10.common import read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--system", required=True)
    parser.add_argument("--metric", default="hits_at_10")
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    base = {row["query_id"]: row for row in read_jsonl(args.baseline)}
    system = {row["query_id"]: row for row in read_jsonl(args.system)}
    if set(base) != set(system):
        raise SystemExit("paired files do not contain identical query IDs")
    ids = sorted(base)
    differences = np.asarray([float(system[q][args.metric]) - float(base[q][args.metric]) for q in ids])
    rng = np.random.default_rng(args.seed)
    draws = np.empty(args.samples, dtype=np.float64)
    null_extreme = 0
    observed = float(differences.mean())
    for start in range(0, args.samples, 256):
        stop = min(args.samples, start + 256)
        draws[start:stop] = rng.choice(
            differences, size=(stop - start, len(differences)), replace=True
        ).mean(axis=1)
        signs = rng.choice(np.asarray([-1.0, 1.0]), size=(stop - start, len(differences)))
        null_means = (signs * differences).mean(axis=1)
        null_extreme += int(np.sum(np.abs(null_means) >= abs(observed)))
    result = {
        "metric": args.metric,
        "n_queries": len(ids),
        "samples": args.samples,
        "mean_difference": observed,
        "ci95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
        "two_sided_p": float((null_extreme + 1) / (args.samples + 1)),
    }
    text = json.dumps(result, indent=2) + "\n"
    if args.out:
        from pathlib import Path

        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
