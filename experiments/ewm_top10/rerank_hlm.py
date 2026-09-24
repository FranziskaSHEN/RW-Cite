#!/usr/bin/env python3
"""HLM-style LLM reranking with deterministic JSON repair and per-query cache."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from experiments.ewm_top10.common import read_jsonl, sha256, unique_top_k, write_jsonl


ANALYZER_SYSTEM = """You are the analyzer in a scientific citation-ranking workflow. Use only the query and candidate evidence supplied in the prompt. Do not use tools, web search, or unstated bibliographic facts. Analyze why each candidate may or may not be a core citation. Do not rank or omit candidates. Return concise JSON mapping every candidate ID to an analysis."""

DECIDER_SYSTEM = """You are the decider in a scientific citation-ranking workflow. Use the query, candidate metadata, and the analyzer's assessments to select exactly ten papers most likely to be cited. Prioritize directly reused methods, experimental baselines, datasets, evaluation protocols, foundations, and closely related solutions. Return JSON only as {\"ranked_ids\":[\"id1\",...]} with unique IDs from the candidates."""


def _candidate_evidence(row: dict[str, Any]) -> str:
    return str(
        row.get("evidence_text")
        or f"Title: {row['title']}\nAbstract: {row['abstract']}"
    )


def candidate_prompt(query: dict[str, Any], candidates: list[dict[str, Any]], one_shot: str = "") -> str:
    blocks = "\n\n".join(
        f"[{row['paper_id']}]\nEvidence: {_candidate_evidence(row)}" for row in candidates
    )
    guide = f"Human-reviewed ranking example\n{one_shot}\n\n" if one_shot else ""
    return f"{guide}Query\nTitle: {query['title']}\nAbstract: {query['abstract']}\n\nCandidates\n{blocks}"


def call_chat_completions(
    endpoint: str,
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    timeout: int,
    *,
    max_tokens: int,
    max_retries: int,
    disable_thinking: bool,
) -> tuple[str, dict[str, Any], int]:
    payload_data = {
        "model": model,
        "temperature": 0,
        "max_tokens": max_tokens,
        "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
        "response_format": {"type": "json_object"},
    }
    if disable_thinking:
        # Qwen thinking models can otherwise consume the complete output
        # budget in reasoning_content and return an empty content field.
        payload_data["chat_template_kwargs"] = {"enable_thinking": False}
    payload = json.dumps(payload_data).encode()
    request = urllib.request.Request(
        endpoint.rstrip("/") + "/chat/completions",
        data=payload,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    for attempt in range(1, max_retries + 2):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - user-selected API endpoint
                body = json.loads(response.read().decode())
            return body["choices"][0]["message"]["content"], body.get("usage") or {}, attempt
        except urllib.error.HTTPError as exc:
            if exc.code not in (408, 409, 429) and exc.code < 500:
                raise
            if attempt > max_retries:
                raise
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError):
            if attempt > max_retries:
                raise
        time.sleep(min(30.0, 2.0 ** (attempt - 1)))
    raise RuntimeError("unreachable API retry state")


def parse_ids(text: str) -> list[str]:
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.S)
        value = json.loads(match.group(0)) if match else {}
    return list(value.get("ranked_ids") or []) if isinstance(value, dict) else []


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval", required=True)
    parser.add_argument("--test", default="outputs/ewm_top10/data/test.jsonl")
    parser.add_argument("--corpus", default="outputs/ewm_top10/data/corpus.jsonl")
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--endpoint", default="https://api.openai.com/v1")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--cache-dir", default="outputs/ewm_top10/cache/hlm")
    parser.add_argument("--one-shot", default="", help="human-reviewed HLM guider example text")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--dry-run", action="store_true", help="repair retrieval order without calling an API")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument(
        "--disable-thinking",
        action="store_true",
        help="disable hidden reasoning for APIs that support Qwen chat_template_kwargs",
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=0,
        help="process only the first N queries for a smoke test; zero uses all queries",
    )
    parser.add_argument("--replicate", type=int, default=1, help="independent stability run ID")
    parser.add_argument(
        "--require-frozen-evidence",
        action="store_true",
        help="fail unless every candidate has a precomputed temporal evidence view",
    )
    parser.add_argument(
        "--require-retrieval-method-prefix",
        default="",
        help="reject a candidate window not produced by the declared retriever",
    )
    parser.add_argument(
        "--expected-window-size",
        type=int,
        default=0,
        help="require exactly this many unique candidate IDs per query",
    )
    args = parser.parse_args()

    queries = {row["query_id"]: row for row in read_jsonl(args.test)}
    corpus = {row["paper_id"]: row for row in read_jsonl(args.corpus)}
    retrieval = list(read_jsonl(args.retrieval))
    if args.max_queries < 0:
        raise SystemExit("--max-queries must be non-negative")
    if args.max_queries:
        retrieval = retrieval[: args.max_queries]
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    api_key = os.environ.get(args.api_key_env, "")
    if not args.dry_run and not api_key:
        raise SystemExit(f"missing API key environment variable: {args.api_key_env}")
    one_shot = Path(args.one_shot).read_text(encoding="utf-8") if args.one_shot else ""
    import hashlib

    fingerprint = hashlib.sha256(
        json.dumps(
            {
                "model": args.model,
                "retrieval_sha256": sha256(args.retrieval),
                "one_shot": one_shot,
                "top_k": args.top_k,
                "analyzer_system": ANALYZER_SYSTEM,
                "decider_system": DECIDER_SYSTEM,
                "analyzer_max_tokens": 2048,
                "decider_max_tokens": 512,
                "replicate": args.replicate,
                "disable_thinking": args.disable_thinking,
                "runner_version": 3,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()[:16]
    cache_dir = cache_dir / fingerprint
    cache_dir.mkdir(parents=True, exist_ok=True)

    output = []
    for row in retrieval:
        qid = row["query_id"]
        retrieval_method = str(row.get("method") or "").strip()
        if args.require_retrieval_method_prefix and not retrieval_method.startswith(
            args.require_retrieval_method_prefix
        ):
            raise SystemExit(
                f"{qid} has candidate-window method {retrieval_method!r}; expected "
                f"prefix {args.require_retrieval_method_prefix!r}"
            )
        supplied_evidence = {
            item["paper_id"]: item for item in (row.get("candidate_evidence") or [])
        }
        retrieval_ids = [pid for pid in row["ranked_ids"] if pid in corpus]
        if len(retrieval_ids) != len(set(retrieval_ids)):
            raise SystemExit(f"candidate window contains duplicate IDs for query {qid}")
        if args.expected_window_size and len(retrieval_ids) != args.expected_window_size:
            raise SystemExit(
                f"{qid} has {len(retrieval_ids)} candidates; expected "
                f"{args.expected_window_size}"
            )
        if args.require_frozen_evidence and any(pid not in supplied_evidence for pid in retrieval_ids):
            raise SystemExit(f"fixed temporal evidence is missing for query {qid}")
        candidate_rows = [supplied_evidence.get(pid, corpus[pid]) for pid in retrieval_ids]
        allowed = set(retrieval_ids)
        cache = cache_dir / f"{qid.replace('/', '_')}.json"
        tick = time.perf_counter()
        usage: dict[str, Any] = {}
        parse_ok = True
        analyzer_ok = True
        api_attempts = 0
        api_latency_ms = 0.0
        if args.dry_run:
            proposed = retrieval_ids
        elif cache.is_file():
            cached = json.loads(cache.read_text(encoding="utf-8"))
            proposed, usage = cached.get("proposed_ids") or [], cached.get("usage") or {}
            parse_ok = bool(cached.get("parse_ok", True))
            analyzer_ok = bool(cached.get("analyzer_ok", True))
            api_attempts = int(cached.get("api_attempts") or 0)
            api_latency_ms = float(cached.get("api_latency_ms") or 0.0)
        else:
            base_prompt = candidate_prompt(queries[qid], candidate_rows, one_shot)
            api_tick = time.perf_counter()
            raw_analysis, analyzer_usage, analyzer_attempts = call_chat_completions(
                args.endpoint,
                api_key,
                args.model,
                ANALYZER_SYSTEM,
                base_prompt,
                args.timeout,
                max_tokens=2048,
                max_retries=args.max_retries,
                disable_thinking=args.disable_thinking,
            )
            raw, decider_usage, decider_attempts = call_chat_completions(
                args.endpoint,
                api_key,
                args.model,
                DECIDER_SYSTEM,
                f"{base_prompt}\n\nAnalyzer assessments\n{raw_analysis}",
                args.timeout,
                max_tokens=512,
                max_retries=args.max_retries,
                disable_thinking=args.disable_thinking,
            )
            api_latency_ms = (time.perf_counter() - api_tick) * 1000.0
            api_attempts = analyzer_attempts + decider_attempts
            usage = {"analyzer": analyzer_usage, "decider": decider_usage}
            try:
                analyzer_value = json.loads(raw_analysis)
                analyzer_ok = isinstance(analyzer_value, dict) and allowed.issubset(
                    set(analyzer_value)
                )
            except (json.JSONDecodeError, TypeError):
                analyzer_ok = False
            try:
                proposed = parse_ids(raw)
            except (json.JSONDecodeError, TypeError):
                proposed, parse_ok = [], False
            parse_ok = (
                parse_ok
                and len(proposed) == args.top_k
                and len(set(proposed)) == args.top_k
                and set(proposed).issubset(allowed)
            )
            cache.write_text(
                json.dumps(
                    {
                        "raw_analysis": raw_analysis,
                        "raw_decision": raw,
                        "proposed_ids": proposed,
                        "usage": usage,
                        "parse_ok": parse_ok,
                        "analyzer_ok": analyzer_ok,
                        "api_attempts": api_attempts,
                        "api_latency_ms": api_latency_ms,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
        ranked = unique_top_k(proposed, allowed, args.top_k)
        ranked = unique_top_k(ranked + retrieval_ids, allowed, args.top_k)
        output.append(
            {
                "query_id": qid,
                "method": (
                    "hlm_style_analyzer_decider"
                    if not args.dry_run
                    else "hlm_retriever_window_order"
                ),
                "candidate_window_method": retrieval_method,
                "ranked_ids": ranked,
                "latency_ms": api_latency_ms if not args.dry_run else (time.perf_counter() - tick) * 1000.0,
                "parse_ok": parse_ok,
                "analyzer_ok": analyzer_ok,
                "padded": len(unique_top_k(proposed, allowed, args.top_k)) < args.top_k,
                "usage": usage,
                "api_attempts": api_attempts,
            }
        )
    write_jsonl(args.out, output)
    Path(args.out).with_suffix(".meta.json").write_text(
        json.dumps(
            {
                "model": args.model,
                "endpoint": args.endpoint,
                "temperature": 0,
                "web_access": False,
                "input_temporally_controlled": args.require_frozen_evidence,
                "model_training_cutoff_certified": False,
                "one_shot": args.one_shot,
                "retrieval_sha256": sha256(args.retrieval),
                "retrieval_method_prefix": args.require_retrieval_method_prefix,
                "expected_window_size": args.expected_window_size,
                "replicate": args.replicate,
                "max_queries": args.max_queries,
                "max_retries": args.max_retries,
                "disable_thinking": args.disable_thinking,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {args.out} n={len(output)}")


if __name__ == "__main__":
    main()
