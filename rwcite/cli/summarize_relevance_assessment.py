#!/usr/bin/env python3
"""Summarize fixed-list LLM relevance judgments used in the paper."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import networkx as nx

from rwcite.ranker.reference_recommend import normalize_arxiv_id


LABELS = ("core", "related", "peripheral", "none")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_no}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"expected object at {path}:{line_no}")
            rows.append(row)
    return rows


def _share(count: int, total: int) -> float:
    return count / total if total else 0.0


def summarize(
    analyses: Path,
    graph_path: Path,
    *,
    expected_candidates: int = 30,
    expected_queries: int = 0,
) -> dict[str, Any]:
    graph = nx.read_gexf(str(graph_path), node_type=None, relabel=False, version="1.2draft")
    node_by_id = {normalize_arxiv_id(str(node)): node for node in graph.nodes}

    by_query: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in _read_jsonl(analyses):
        query_id = normalize_arxiv_id(str(row.get("source_id") or row.get("query_id") or ""))
        candidate_id = normalize_arxiv_id(str(row.get("cand_id") or row.get("candidate_id") or ""))
        label = str(row.get("relevance") or "").strip().lower()
        if not query_id or not candidate_id:
            raise ValueError("every judgment must contain source/query and candidate IDs")
        if label not in LABELS:
            raise ValueError(f"invalid relevance label {label!r} for {query_id}/{candidate_id}")
        by_query[query_id].append(
            {"candidate_id": candidate_id, "label": label, "parse_ok": bool(row.get("parse_ok", True))}
        )

    if expected_queries and len(by_query) != expected_queries:
        raise ValueError(f"expected {expected_queries} queries, found {len(by_query)}")
    if expected_candidates:
        bad = {qid: len(rows) for qid, rows in by_query.items() if len(rows) != expected_candidates}
        if bad:
            examples = list(sorted(bad.items()))[:5]
            raise ValueError(
                f"expected {expected_candidates} judgments per query; mismatches include {examples}"
            )

    all_labels: Counter[str] = Counter()
    gold_labels: Counter[str] = Counter()
    nongold_labels: Counter[str] = Counter()
    n_parsed = 0
    gold_per_query: list[int] = []

    for query_id, rows in by_query.items():
        node = node_by_id.get(query_id)
        if node is None:
            raise ValueError(f"query {query_id} is absent from graph {graph_path}")
        gold = {normalize_arxiv_id(str(dst)) for dst in graph.successors(node)}
        gold_in_assessment = 0
        seen_candidates: set[str] = set()
        for row in rows:
            candidate_id = row["candidate_id"]
            if candidate_id in seen_candidates:
                raise ValueError(f"duplicate candidate {candidate_id} for query {query_id}")
            seen_candidates.add(candidate_id)
            label = row["label"]
            all_labels[label] += 1
            n_parsed += int(row["parse_ok"])
            if candidate_id in gold:
                gold_labels[label] += 1
                gold_in_assessment += 1
            else:
                nongold_labels[label] += 1
        gold_per_query.append(gold_in_assessment)

    n_total = sum(all_labels.values())
    n_gold = sum(gold_labels.values())
    n_nongold = sum(nongold_labels.values())
    sorted_gold = sorted(gold_per_query)
    median_gold = (
        0.0
        if not sorted_gold
        else float(sorted_gold[len(sorted_gold) // 2])
        if len(sorted_gold) % 2
        else (sorted_gold[len(sorted_gold) // 2 - 1] + sorted_gold[len(sorted_gold) // 2]) / 2
    )

    def distribution(counts: Counter[str], total: int) -> dict[str, dict[str, float | int]]:
        return {
            label: {"count": int(counts[label]), "share": _share(int(counts[label]), total)}
            for label in LABELS
        }

    return {
        "n_queries": len(by_query),
        "candidates_per_query": expected_candidates or None,
        "n_judgments": n_total,
        "n_parsed": n_parsed,
        "parse_rate": _share(n_parsed, n_total),
        "mean_gold_per_query": _share(n_gold, len(by_query)),
        "median_gold_per_query": median_gold,
        "zero_gold_queries": sum(value == 0 for value in gold_per_query),
        "labels": distribution(all_labels, n_total),
        "core_or_related_share": _share(all_labels["core"] + all_labels["related"], n_total),
        "gold": {
            "count": n_gold,
            "share": _share(n_gold, n_total),
            "labels": distribution(gold_labels, n_gold),
            "core_or_related_share": _share(gold_labels["core"] + gold_labels["related"], n_gold),
        },
        "non_gold": {
            "count": n_nongold,
            "share": _share(n_nongold, n_total),
            "labels": distribution(nongold_labels, n_nongold),
            "core_or_related_share": _share(
                nongold_labels["core"] + nongold_labels["related"], n_nongold
            ),
        },
        "analyses": str(analyses),
        "graph": str(graph_path),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analyses", type=Path, required=True)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expected-candidates", type=int, default=30)
    parser.add_argument("--expected-queries", type=int, default=0)
    args = parser.parse_args(argv)

    report = summarize(
        args.analyses,
        args.graph,
        expected_candidates=args.expected_candidates,
        expected_queries=args.expected_queries,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
