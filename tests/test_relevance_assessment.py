from __future__ import annotations

import json
from pathlib import Path

import networkx as nx

from rwcite.cli.summarize_relevance_assessment import summarize


def test_summarize_relevance_assessment(tmp_path: Path):
    graph = nx.DiGraph()
    graph.add_edge("q1", "c1")
    graph.add_node("c2")
    graph_path = tmp_path / "graph.gexf"
    nx.write_gexf(graph, graph_path)

    analyses = tmp_path / "analyses.jsonl"
    rows = [
        {"source_id": "q1", "cand_id": "c1", "relevance": "core", "parse_ok": True},
        {"source_id": "q1", "cand_id": "c2", "relevance": "related", "parse_ok": True},
    ]
    analyses.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )

    report = summarize(analyses, graph_path, expected_candidates=2, expected_queries=1)
    assert report["n_judgments"] == 2
    assert report["core_or_related_share"] == 1.0
    assert report["gold"]["labels"]["core"]["share"] == 1.0
    assert report["non_gold"]["labels"]["related"]["share"] == 1.0
