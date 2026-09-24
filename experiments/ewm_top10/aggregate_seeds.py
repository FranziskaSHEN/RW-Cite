#!/usr/bin/env python3
"""Aggregate metric summaries across training seeds as mean and sample std."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean, stdev


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summaries", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    rows = [json.loads(Path(path).read_text(encoding="utf-8")) for path in args.summaries]
    if not rows:
        raise SystemExit("no summaries")
    ignored = {"n_queries", "method", "prediction_file"}
    metrics = sorted(set.intersection(*[{key for key, value in row.items() if key not in ignored and isinstance(value, (int, float)) and math.isfinite(value)} for row in rows]))
    result = {
        "method": rows[0].get("method"),
        "n_seeds": len(rows),
        "n_queries": rows[0].get("n_queries"),
        "metrics": {
            metric: {
                "mean": mean(float(row[metric]) for row in rows),
                "std": stdev(float(row[metric]) for row in rows) if len(rows) > 1 else 0.0,
            }
            for metric in metrics
        },
        "source_summaries": args.summaries,
    }
    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
