from __future__ import annotations

import json
from unittest.mock import patch

import networkx as nx

from experiments.ewm_top10 import prepare_fixed_windows
from experiments.ewm_top10.common import read_jsonl


def _write_jsonl(path, rows) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_fixed_window_drops_future_candidates_and_sentences(tmp_path) -> None:
    query = "2503.00001"
    candidates = [f"2401.{index:05d}" for index in range(10)]
    future_candidate = "2504.00001"
    graph = nx.DiGraph()
    graph.add_nodes_from([query, future_candidate, *candidates])
    graph.add_edge("2502.00001", candidates[0], sentence="prior evidence")
    graph.add_edge("2504.00002", candidates[0], sentence="future evidence")
    graph_path = tmp_path / "graph.gexf"
    nx.write_gexf(graph, graph_path)

    test_path = tmp_path / "test.jsonl"
    corpus_path = tmp_path / "corpus.jsonl"
    retrieval_path = tmp_path / "retrieval.jsonl"
    out_path = tmp_path / "windows.jsonl"
    _write_jsonl(test_path, [{"query_id": query, "title": "q", "abstract": ""}])
    _write_jsonl(
        corpus_path,
        [
            {"paper_id": pid, "title": pid, "abstract": ""}
            for pid in [future_candidate, *candidates]
        ],
    )
    _write_jsonl(
        retrieval_path,
        [{"query_id": query, "ranked_ids": [future_candidate, *candidates]}],
    )
    argv = [
        "prepare_fixed_windows",
        "--retrieval",
        str(retrieval_path),
        "--test",
        str(test_path),
        "--corpus",
        str(corpus_path),
        "--graph",
        str(graph_path),
        "--out",
        str(out_path),
        "--window",
        "10",
    ]
    with patch("sys.argv", argv):
        prepare_fixed_windows.main()

    row = next(read_jsonl(out_path))
    assert future_candidate not in row["ranked_ids"]
    assert row["source_method"] == ""
    assert row["source_retrieval_sha256"]
    first = next(item for item in row["candidate_evidence"] if item["paper_id"] == candidates[0])
    assert first["incoming_citation_sentences"] == ["prior evidence"]
    assert "future evidence" not in first["evidence_text"]
    meta = json.loads(out_path.with_suffix(".meta.json").read_text(encoding="utf-8"))
    assert meta["window"] == 10
    assert meta["n_queries"] == 1


def test_fixed_window_rejects_wrong_retriever(tmp_path) -> None:
    query = "2503.00001"
    candidates = [f"2401.{index:05d}" for index in range(10)]
    graph = nx.DiGraph()
    graph.add_nodes_from([query, *candidates])
    graph_path = tmp_path / "graph.gexf"
    nx.write_gexf(graph, graph_path)
    test_path = tmp_path / "test.jsonl"
    corpus_path = tmp_path / "corpus.jsonl"
    retrieval_path = tmp_path / "retrieval.jsonl"
    out_path = tmp_path / "windows.jsonl"
    _write_jsonl(test_path, [{"query_id": query, "title": "q", "abstract": ""}])
    _write_jsonl(
        corpus_path,
        [{"paper_id": pid, "title": pid, "abstract": ""} for pid in candidates],
    )
    _write_jsonl(
        retrieval_path,
        [{"query_id": query, "method": "bge_neutral", "ranked_ids": candidates}],
    )
    argv = [
        "prepare_fixed_windows",
        "--retrieval",
        str(retrieval_path),
        "--test",
        str(test_path),
        "--corpus",
        str(corpus_path),
        "--graph",
        str(graph_path),
        "--out",
        str(out_path),
        "--window",
        "10",
        "--expected-method-prefix",
        "hlm_retriever",
    ]
    with patch("sys.argv", argv):
        try:
            prepare_fixed_windows.main()
        except SystemExit as error:
            assert "expected prefix" in str(error)
        else:
            raise AssertionError("BGE retrieval was accepted as an HLM window")
