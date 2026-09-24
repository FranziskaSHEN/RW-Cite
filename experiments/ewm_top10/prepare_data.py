#!/usr/bin/env python3
"""Create one auditable EWM train/dev/test contract for all compared systems."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import networkx as nx

from experiments.ewm_top10.common import (
    chronological_split,
    month_string,
    normalize_id,
    paper_month,
    parse_month,
    read_jsonl,
    sha256,
    write_jsonl,
)


def adapt_query(row: dict[str, Any], graph: nx.DiGraph) -> dict[str, Any]:
    qid = normalize_id(row.get("query_id") or row.get("source_id") or row.get("id"))
    if not qid:
        raise ValueError("query row has no ID")
    attrs = dict(graph.nodes[qid]) if qid in graph else {}
    raw_gold = row.get("gold_ids") or row.get("gold") or []
    gold_ids = []
    for item in raw_gold:
        pid = normalize_id(item.get("id") if isinstance(item, dict) else item)
        if pid and pid not in gold_ids:
            gold_ids.append(pid)
    qmonth = paper_month(qid, attrs)
    eligible_gold = [
        pid
        for pid in gold_ids
        if pid in graph
        and qmonth is not None
        and paper_month(pid, dict(graph.nodes[pid])) is not None
        and paper_month(pid, dict(graph.nodes[pid])) < qmonth
    ]
    return {
        "query_id": qid,
        "title": str(row.get("title") or attrs.get("title") or attrs.get("label") or qid).strip(),
        "abstract": str(row.get("abstract") or attrs.get("abstract") or "").strip(),
        "published_at": month_string(qmonth),
        "gold_ids": eligible_gold,
        "raw_gold_count": len(gold_ids),
        "eligible_gold_count": len(eligible_gold),
    }


def deduplicate_queries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        previous = by_id.get(row["query_id"])
        if previous is None or len(row["gold_ids"]) > len(previous["gold_ids"]):
            by_id[row["query_id"]] = row
    return list(by_id.values())


def main() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    data_root = Path(
        os.environ.get("RWCITE_DATA_ROOT") or repository_root
    ).expanduser().resolve()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-jsonl",
        default=str(
            data_root
            / "embodied_world_model_retrieval/data/reference_recommend/train.jsonl"
        ),
    )
    parser.add_argument(
        "--test-jsonl",
        default=str(
            data_root
            / "embodied_world_model_retrieval/data/reference_recommend/test.jsonl"
        ),
    )
    parser.add_argument(
        "--graph",
        default=str(
            data_root
            / "embodied_world_model_retrieval/description/as_sclean/test_graph_rr.gexf"
        ),
    )
    parser.add_argument("--out", default="outputs/ewm_top10/data")
    parser.add_argument("--dev-ratio", type=float, default=0.1)
    args = parser.parse_args()

    graph_path = Path(args.graph)
    train_path, test_path = Path(args.train_jsonl), Path(args.test_jsonl)
    for path in (graph_path, train_path, test_path):
        if not path.is_file():
            raise SystemExit(f"missing required asset: {path}")

    graph = nx.read_gexf(graph_path)
    if not graph.is_directed():
        graph = graph.to_directed()

    raw_train = [adapt_query(row, graph) for row in read_jsonl(train_path)]
    train_all = deduplicate_queries(raw_train)
    test = deduplicate_queries([adapt_query(row, graph) for row in read_jsonl(test_path)])
    empty_test = [row["query_id"] for row in test if not row["gold_ids"]]
    if empty_test:
        raise SystemExit(
            f"{len(empty_test)} test queries have no temporally eligible gold; "
            "fix dates/corpus instead of silently changing the benchmark"
        )
    test_ids = {row["query_id"] for row in test}
    if test_ids & {row["query_id"] for row in train_all}:
        raise SystemExit("train/dev and test query IDs overlap")
    dated_test = [parse_month(row["published_at"]) for row in test]
    dated_test = [month for month in dated_test if month is not None]
    if not dated_test:
        raise SystemExit("official test set has no parseable query months")
    test_cutoff = min(dated_test)

    # The source release can contain distinct training queries from the same
    # calendar month as the earliest test queries. With month-resolution dates,
    # their within-month order is unknowable, so they cannot safely train one
    # checkpoint that is evaluated on the complete test period.
    pretest_train = [
        row
        for row in train_all
        if (month := parse_month(row["published_at"])) is not None and month < test_cutoff
    ]
    excluded_unknown = sum(parse_month(row["published_at"]) is None for row in train_all)
    excluded_at_or_after_test = sum(
        (month := parse_month(row["published_at"])) is not None and month >= test_cutoff
        for row in train_all
    )
    train, dev, dev_cutoff = chronological_split(pretest_train, args.dev_ratio)
    dated_dev = [parse_month(row["published_at"]) for row in dev]
    dated_dev = [month for month in dated_dev if month is not None]
    if not dated_dev or not dated_test or max(dated_dev) >= min(dated_test):
        raise SystemExit("development and test periods are not strictly chronological")

    corpus = []
    corpus_seen: set[str] = set()
    for node, attrs_raw in graph.nodes(data=True):
        pid = normalize_id(node)
        attrs = dict(attrs_raw)
        month = paper_month(pid, attrs)
        title = str(attrs.get("title") or attrs.get("label") or pid).strip()
        if not pid or not month or not title or pid in corpus_seen:
            continue
        corpus_seen.add(pid)
        corpus.append(
            {
                "paper_id": pid,
                "title": title,
                "abstract": str(attrs.get("abstract") or "").strip(),
                "published_at": month_string(month),
            }
        )

    out = Path(args.out)
    write_jsonl(out / "train.jsonl", train)
    write_jsonl(out / "dev.jsonl", dev)
    write_jsonl(out / "pretest_train.jsonl", pretest_train)
    write_jsonl(out / "test.jsonl", test)
    write_jsonl(out / "corpus.jsonl", corpus)
    manifest = {
        "rwcite_data_root": str(data_root),
        "split_protocol": "strict_month_boundary_train_dev",
        "dev_cutoff_inclusive": month_string(dev_cutoff),
        "test_cutoff_exclusive_for_training": month_string(test_cutoff),
        "unknown_train_month_policy": "exclude",
        "dev_ratio": args.dev_ratio,
        "n_train_source_rows": len(raw_train),
        "n_duplicate_train_rows_removed": len(raw_train) - len(train_all),
        "n_unique_train_before_test_cutoff": len(pretest_train),
        "n_unique_train_excluded_unknown_month": excluded_unknown,
        "n_unique_train_excluded_at_or_after_test_cutoff": excluded_at_or_after_test,
        "n_train": len(train),
        "n_dev": len(dev),
        "n_test": len(test),
        "n_corpus": len(corpus),
        "graph": str(graph_path),
        "graph_sha256": sha256(graph_path),
        "train_source_sha256": sha256(train_path),
        "test_source_sha256": sha256(test_path),
        "dropped_test_gold": sum(r["raw_gold_count"] - r["eligible_gold_count"] for r in test),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
