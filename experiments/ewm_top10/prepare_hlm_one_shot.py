#!/usr/bin/env python3
"""Prepare a training-only HLM Analyzer--Decider one-shot draft."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from experiments.ewm_top10.common import read_jsonl, sha256
from experiments.ewm_top10.rerank_hlm import (
    ANALYZER_SYSTEM,
    call_chat_completions,
    candidate_prompt,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", required=True)
    parser.add_argument("--windows", required=True)
    parser.add_argument("--query-id", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--api-key-env", default="HLM_API_KEY")
    parser.add_argument("--out", required=True)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--max-retries", type=int, default=3)
    args = parser.parse_args()

    api_key = os.environ.get(args.api_key_env, "")
    if not api_key:
        raise SystemExit(f"missing API key environment variable: {args.api_key_env}")

    queries = {row["query_id"]: row for row in read_jsonl(args.queries)}
    windows = {row["query_id"]: row for row in read_jsonl(args.windows)}
    if args.query_id not in queries or args.query_id not in windows:
        raise SystemExit(f"query {args.query_id} is absent from the supplied training files")

    query = queries[args.query_id]
    window = windows[args.query_id]
    ranked_ids = window.get("ranked_ids") or []
    if len(ranked_ids) != 30 or len(set(ranked_ids)) != 30:
        raise SystemExit("one-shot source must have exactly 30 unique candidate IDs")
    evidence_by_id = {
        row["paper_id"]: row for row in (window.get("candidate_evidence") or [])
    }
    if any(pid not in evidence_by_id for pid in ranked_ids):
        raise SystemExit("one-shot source is missing frozen evidence")

    gold = set(query.get("gold_ids") or [])
    target = [pid for pid in ranked_ids if pid in gold][:10]
    if len(target) != 10:
        raise SystemExit(
            f"one-shot query has only {len(target)} gold candidates in its Top-30; need ten"
        )

    candidates = [evidence_by_id[pid] for pid in ranked_ids]
    prompt = candidate_prompt(query, candidates)
    raw_analysis, usage, attempts = call_chat_completions(
        args.endpoint,
        api_key,
        args.model,
        ANALYZER_SYSTEM,
        prompt,
        args.timeout,
        max_tokens=2048,
        max_retries=args.max_retries,
    )
    try:
        analysis = json.loads(raw_analysis)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Analyzer did not return valid JSON: {exc}") from exc
    missing = set(ranked_ids) - set(analysis if isinstance(analysis, dict) else {})
    if missing:
        raise SystemExit(f"Analyzer omitted {len(missing)} candidate IDs: {sorted(missing)}")

    text = (
        f"{prompt}\n\n"
        f"Analyzer assessments\n{json.dumps(analysis, ensure_ascii=False, indent=2)}\n\n"
        f"Decider\n{json.dumps({'ranked_ids': target}, ensure_ascii=False)}\n"
    )
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
    meta = {
        "query_id": args.query_id,
        "query_month": query.get("published_at"),
        "model": args.model,
        "endpoint": args.endpoint,
        "queries_sha256": sha256(args.queries),
        "windows_sha256": sha256(args.windows),
        "n_candidates": len(ranked_ids),
        "n_gold_in_window": sum(pid in gold for pid in ranked_ids),
        "n_target": len(target),
        "analyzer_attempts": attempts,
        "usage": usage,
        "target_uses_training_labels": True,
        "manual_review_required": True,
    }
    output.with_suffix(".meta.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote one-shot draft: {output}")
    print(f"manual review required; target={target}")


if __name__ == "__main__":
    main()
