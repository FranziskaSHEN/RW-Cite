from __future__ import annotations

import json
from unittest.mock import patch

from experiments.ewm_top10 import rerank_hlm
from experiments.ewm_top10.common import read_jsonl


def _write_jsonl(path, rows) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_hlm_judge_accepts_only_hlm_retriever_window(tmp_path) -> None:
    query = {"query_id": "2503.00001", "title": "Query", "abstract": "Abstract"}
    candidates = [f"2401.{index:05d}" for index in range(30)]
    corpus = [
        {
            "paper_id": pid,
            "title": f"Candidate {index}",
            "abstract": "Evidence",
        }
        for index, pid in enumerate(candidates)
    ]
    evidence = [
        {
            "paper_id": row["paper_id"],
            "title": row["title"],
            "abstract": row["abstract"],
            "incoming_citation_sentences": [],
            "evidence_text": f"Title: {row['title']} Abstract: {row['abstract']}",
        }
        for row in corpus
    ]
    test_path = tmp_path / "test.jsonl"
    corpus_path = tmp_path / "corpus.jsonl"
    window_path = tmp_path / "hlm_top30.jsonl"
    output_path = tmp_path / "predictions.jsonl"
    _write_jsonl(test_path, [query])
    _write_jsonl(corpus_path, corpus)
    _write_jsonl(
        window_path,
        [
            {
                "query_id": query["query_id"],
                "method": "hlm_retriever_seed42_temporal_top30",
                "ranked_ids": candidates,
                "candidate_evidence": evidence,
            }
        ],
    )
    argv = [
        "rerank_hlm",
        "--retrieval",
        str(window_path),
        "--test",
        str(test_path),
        "--corpus",
        str(corpus_path),
        "--out",
        str(output_path),
        "--model",
        "test-model",
        "--dry-run",
        "--require-frozen-evidence",
        "--require-retrieval-method-prefix",
        "hlm_retriever",
        "--expected-window-size",
        "30",
        "--cache-dir",
        str(tmp_path / "cache"),
    ]
    with patch("sys.argv", argv):
        rerank_hlm.main()
    row = next(read_jsonl(output_path))
    assert row["candidate_window_method"].startswith("hlm_retriever")
    assert row["ranked_ids"] == candidates[:10]


def test_hlm_judge_rejects_bge_window(tmp_path) -> None:
    test_path = tmp_path / "test.jsonl"
    corpus_path = tmp_path / "corpus.jsonl"
    window_path = tmp_path / "bge_top30.jsonl"
    output_path = tmp_path / "predictions.jsonl"
    _write_jsonl(test_path, [{"query_id": "q", "title": "q", "abstract": ""}])
    _write_jsonl(corpus_path, [])
    _write_jsonl(
        window_path,
        [{"query_id": "q", "method": "bge_neutral_temporal_top30", "ranked_ids": []}],
    )
    argv = [
        "rerank_hlm",
        "--retrieval",
        str(window_path),
        "--test",
        str(test_path),
        "--corpus",
        str(corpus_path),
        "--out",
        str(output_path),
        "--model",
        "test-model",
        "--dry-run",
        "--require-retrieval-method-prefix",
        "hlm_retriever",
        "--cache-dir",
        str(tmp_path / "cache"),
    ]
    with patch("sys.argv", argv):
        try:
            rerank_hlm.main()
        except SystemExit as error:
            assert "expected prefix" in str(error)
        else:
            raise AssertionError("BGE window was accepted by the HLM pipeline")
