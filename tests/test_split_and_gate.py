from __future__ import annotations

import json
import sys
from pathlib import Path

import networkx as nx

from rwcite.cli import dump_split, gate_citelink
from rwcite.graph.domain_paths import DEFAULT_SPLIT_ID


def test_arxiv_ym():
    assert dump_split.arxiv_ym("2101.00001") == (2021, 1)
    assert dump_split.arxiv_ym("hep-th/9901001") is None


def test_dump_time_elig10(tmp_path: Path, tiny_digraph: nx.DiGraph):
    gexf = tmp_path / "g.gexf"
    nx.write_gexf(tiny_digraph, gexf)
    out = tmp_path / "split"
    meta = dump_split.dump_time_elig10(
        tiny_digraph,
        out,
        gexf=gexf,
        gexf_hash="deadbeef",
        elig_k=2,
        test_frac=0.2,
        ret_pos=None,
        # Tiny fixture has no sentences; exercise T1 without D/E/F.
        frontier_margin_months=0,
        min_sentence_coverage=0.0,
        gold_quantile_lo=0.0,
        gold_quantile_hi=1.0,
    )
    assert meta["split_id"] == DEFAULT_SPLIT_ID
    assert meta["mode"] == "time_elig10"
    assert meta["n_test_sources"] >= 1
    assert (out / "train_sources.json").is_file()
    assert (out / "test_sources.json").is_file()
    train = set(json.loads((out / "train_sources.json").read_text()))
    test = set(json.loads((out / "test_sources.json").read_text()))
    assert not (train & test)


def test_gate_citelink_pass_and_fail(tmp_path: Path, monkeypatch, capsys):
    meta = tmp_path / "meta.json"
    meta.write_text(json.dumps({"best_val_loss": 0.2}), encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        ["gate", "--meta", str(meta), "--max-val-loss", "0.4"],
    )
    assert gate_citelink.main() == 0
    assert "GATE PASS" in capsys.readouterr().out

    meta.write_text(json.dumps({"best_val_loss": 0.9}), encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        ["gate", "--meta", str(meta), "--max-val-loss", "0.4"],
    )
    assert gate_citelink.main() == 1
