#!/usr/bin/env python3
"""Build explicitly retrospective HLM core/superficial training labels."""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import networkx as nx

from experiments.ewm_top10.common import normalize_id, paper_month, read_jsonl, write_jsonl


def main() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    data_root = Path(
        os.environ.get("RWCITE_DATA_ROOT") or repository_root
    ).expanduser().resolve()
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="outputs/ewm_top10/data/train.jsonl")
    parser.add_argument("--dev", default="outputs/ewm_top10/data/dev.jsonl")
    parser.add_argument("--test", default="outputs/ewm_top10/data/test.jsonl")
    parser.add_argument("--corpus", default="outputs/ewm_top10/data/corpus.jsonl")
    parser.add_argument(
        "--graph",
        default=str(
            data_root
            / "embodied_world_model_retrieval/description/as_sclean/test_graph_rr.gexf"
        ),
    )
    parser.add_argument("--out", default="outputs/ewm_top10/data/hlm_train.jsonl")
    parser.add_argument("--random-negatives", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--allow-retrospective-labels",
        action="store_true",
        help=(
            "acknowledge that HLM core labels use post-query co-citations; "
            "held-out IDs and the complete test period remain excluded"
        ),
    )
    args = parser.parse_args()
    if not args.allow_retrospective_labels:
        raise SystemExit(
            "HLM labels are retrospective. Pass --allow-retrospective-labels only "
            "for the separately reported paper-faithful HLM reproduction."
        )

    graph = nx.read_gexf(args.graph)
    if not graph.is_directed():
        graph = graph.to_directed()
    corpus_ids = [normalize_id(row["paper_id"]) for row in read_jsonl(args.corpus)]
    heldout_ids = {
        normalize_id(row["query_id"])
        for path in (args.dev, args.test)
        for row in read_jsonl(path)
    }
    test_months = [
        paper_month(normalize_id(row["query_id"]))
        for row in read_jsonl(args.test)
    ]
    if not test_months or any(month is None for month in test_months):
        raise SystemExit("test split contains an invalid query month")
    test_cutoff = min(month for month in test_months if month is not None)

    def training_evidence_month(node: str) -> tuple[int, int] | None:
        pid = normalize_id(node)
        if pid in heldout_ids:
            return None
        month = paper_month(pid, dict(graph.nodes[node]))
        return month if month is not None and month < test_cutoff else None

    rng = random.Random(args.seed)
    output = []

    for query in read_jsonl(args.train):
        qid = normalize_id(query["query_id"])
        if qid not in graph:
            continue
        qmonth = paper_month(qid, dict(graph.nodes[qid]))
        if qmonth is None:
            continue
        future_citers = {
            normalize_id(node)
            for node in graph.predecessors(qid)
            if training_evidence_month(node) is not None
            and training_evidence_month(node) > qmonth
        }
        core, superficial = [], []
        for cited in query["gold_ids"]:
            pid = normalize_id(cited)
            if pid not in graph:
                continue
            later_citers = {
                normalize_id(node)
                for node in graph.predecessors(pid)
                if training_evidence_month(node) is not None
            }
            (core if future_citers & later_citers else superficial).append(pid)

        forbidden = set(query["gold_ids"]) | {qid}
        eligible = [
            pid
            for pid in corpus_ids
            if pid not in forbidden and paper_month(pid) is not None and paper_month(pid) < qmonth
        ]
        negatives = rng.sample(eligible, min(args.random_negatives, len(eligible)))
        output.append({**query, "core_ids": core, "superficial_ids": superficial, "negative_ids": negatives})

    write_jsonl(args.out, output)
    meta = {
        "label_protocol": "retrospective_pretest_cocitation",
        "primary_temporal_comparison": False,
        "heldout_query_ids_excluded": True,
        "evidence_cutoff_exclusive": f"{test_cutoff[0]:04d}-{test_cutoff[1]:02d}",
        "n_training_queries": len(output),
        "seed": args.seed,
    }
    Path(args.out).with_suffix(".meta.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {args.out} n={len(output)} test_cutoff={test_cutoff}")


if __name__ == "__main__":
    main()
