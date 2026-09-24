#!/usr/bin/env python3
"""Freeze query-time-valid candidate IDs and evidence for ranker-only tests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import networkx as nx

from experiments.ewm_top10.common import normalize_id, read_jsonl, sha256, write_jsonl
from rwcite.ranker.graph_protocol import graph_for_query
from rwcite.ranker.rr_ranker_ce_sent import cand_text_from_sentences
from rwcite.ranker.temporal_evidence import select_temporal_cite_records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--graph", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--window", type=int, required=True)
    parser.add_argument("--max-sents", type=int, default=3)
    parser.add_argument(
        "--expected-method-prefix",
        default="",
        help="reject retrieval rows whose method does not start with this value",
    )
    parser.add_argument(
        "--output-method",
        default="",
        help="method recorded in the frozen-window rows",
    )
    args = parser.parse_args()
    if args.window < 10:
        raise SystemExit("fixed candidate window must contain at least ten papers")

    graph = nx.read_gexf(args.graph)
    if not graph.is_directed():
        graph = graph.to_directed()
    queries = {row["query_id"]: row for row in read_jsonl(args.test)}
    corpus = {row["paper_id"]: row for row in read_jsonl(args.corpus)}
    retrieval_path = Path(args.retrieval).resolve()
    retrieval_sha256 = sha256(retrieval_path)
    output = []
    source_methods: set[str] = set()
    for row in read_jsonl(args.retrieval):
        qid = normalize_id(row["query_id"])
        if qid not in queries:
            raise SystemExit(f"unknown query in retrieval file: {qid}")
        source_method = str(row.get("method") or "").strip()
        if args.expected_method_prefix and not source_method.startswith(
            args.expected_method_prefix
        ):
            raise SystemExit(
                f"{qid} has retrieval method {source_method!r}; expected prefix "
                f"{args.expected_method_prefix!r}"
            )
        source_methods.add(source_method)
        visible = graph_for_query(
            graph,
            qid,
            protocol="rolling",
            same_month_policy="drop",
            unknown_node_policy="drop",
        )
        allowed = {normalize_id(node) for node in visible.nodes} - {qid}
        ids = []
        for raw_id in row.get("ranked_ids") or []:
            pid = normalize_id(raw_id)
            if pid in allowed and pid in corpus and pid not in ids:
                ids.append(pid)
            if len(ids) == args.window:
                break
        if len(ids) < args.window:
            raise SystemExit(f"{qid} has only {len(ids)} admissible candidates; need {args.window}")
        evidence = []
        for pid in ids:
            records = select_temporal_cite_records(
                graph,
                pid,
                qid,
                exclude_citer=qid,
                max_sents=args.max_sents,
                same_month_policy="drop",
                unknown_citer_policy="drop",
                selection_policy="recent",
            )
            sentences = [sentence for _, sentence in records]
            evidence.append(
                {
                    "paper_id": pid,
                    "title": corpus[pid]["title"],
                    "abstract": corpus[pid].get("abstract") or "",
                    "incoming_citation_sentences": sentences,
                    "evidence_text": cand_text_from_sentences(
                        corpus[pid], sentences, short=True
                    ),
                }
            )
        output.append(
            {
                "query_id": qid,
                "method": args.output_method
                or f"{source_method or 'retrieval'}_temporal_top{args.window}",
                "ranked_ids": ids,
                "candidate_evidence": evidence,
                "source_method": source_method,
                "source_retrieval": str(retrieval_path),
                "source_retrieval_sha256": retrieval_sha256,
            }
        )
    if set(queries) != {row["query_id"] for row in output}:
        raise SystemExit("fixed-window file does not cover the complete evaluation split")
    write_jsonl(args.out, output)
    Path(args.out).with_suffix(".meta.json").write_text(
        json.dumps(
            {
                "method": args.output_method
                or f"{next(iter(source_methods), 'retrieval')}_temporal_top{args.window}",
                "source_methods": sorted(source_methods),
                "source_retrieval": str(retrieval_path),
                "source_retrieval_sha256": retrieval_sha256,
                "test_sha256": sha256(args.test),
                "corpus_sha256": sha256(args.corpus),
                "graph_sha256": sha256(args.graph),
                "window": args.window,
                "n_queries": len(output),
                "same_month_policy": "drop",
                "unknown_date_policy": "drop",
                "evidence_selection": "temporal_recent",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {args.out} n={len(output)} window={args.window}")


if __name__ == "__main__":
    main()
