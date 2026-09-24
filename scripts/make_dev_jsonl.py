#!/usr/bin/env python3
"""Carve a dev set out of a train jsonl, one row per source.

Window/shortlist parameters have to be chosen somewhere other than the test
split; this samples train sources so `eval_pool_ranker` can score them with the
same code path as the gate.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from rwcite.ranker.reference_recommend import normalize_arxiv_id


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-jsonl", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rows: list[dict] = []
    seen: set[str] = set()
    with Path(args.train_jsonl).open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            sid = normalize_arxiv_id(str(r.get("source_id") or ""))
            if not sid or sid in seen:
                continue
            seen.add(sid)
            rows.append(r)
    random.Random(args.seed).shuffle(rows)
    rows = rows[: args.n]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {out} n={len(rows)} (from {len(seen)} unique train sources)")


if __name__ == "__main__":
    main()
