"""Frontier margin and test_admit_v1 filters for dump_split."""

from __future__ import annotations

import json

import networkx as nx

from rwcite.cli.dump_split import dump_time_elig10
from rwcite.graph.domain_paths import DEFAULT_SPLIT_ID


def _graph(*, with_sentences: bool = False, outdeg: int = 10) -> nx.DiGraph:
    g = nx.DiGraph()
    # 20 sources per month over 2605..2608.
    for mm in ("2605", "2606", "2607", "2608"):
        for i in range(20):
            src = f"{mm}.{i:05d}"
            for j in range(outdeg):
                tgt = f"1900.{j:05d}"
                if with_sentences:
                    g.add_edge(src, tgt, sentence=f"cite {j}.")
                else:
                    g.add_edge(src, tgt)
    return g


def _legacy_kwargs() -> dict:
    """Reproduce pre-admit time_elig10_v2 filter settings."""
    return {
        "frontier_margin_months": 0,
        "min_sentence_coverage": 0.0,
        "gold_quantile_lo": 0.0,
        "gold_quantile_hi": 1.0,
        "allowed_sources": None,
        "split_id": "time_elig10_v2",
    }


def _dump(tmp_path, months: int) -> dict:
    out = tmp_path / f"m{months}"
    gexf = tmp_path / "g.gexf"
    gexf.write_text("x", encoding="utf-8")
    return dump_time_elig10(
        _graph(),
        out,
        gexf=gexf,
        gexf_hash="0" * 64,
        elig_k=10,
        test_frac=0.10,
        ret_pos=None,
        frontier_margin_months=months,
        min_sentence_coverage=0.0,
        gold_quantile_lo=0.0,
        gold_quantile_hi=1.0,
    )


def test_margin_zero_tests_on_the_newest_month(tmp_path):
    meta = _dump(tmp_path, 0)
    test = json.loads((tmp_path / "m0" / "test_sources.json").read_text())
    assert meta["n_frontier_sources"] == 0
    assert all(s.startswith("2608") for s in test)


def test_margin_one_moves_the_test_window_back(tmp_path):
    meta = _dump(tmp_path, 1)
    d = tmp_path / "m1"
    test = json.loads((d / "test_sources.json").read_text())
    train = json.loads((d / "train_sources.json").read_text())
    frontier = json.loads((d / "frontier_sources.json").read_text())
    assert meta["n_frontier_sources"] == 20
    assert all(s.startswith("2607") for s in test)
    assert all(s.startswith("2608") for s in frontier)
    assert not any(s.startswith("2608") for s in train + test)
    assert "frontier margin" in meta["split_rule"]


def test_allowlist_keeps_enrichment_nodes_out_of_the_split(tmp_path):
    g = _graph()
    for j in range(12):
        g.add_edge("2301.99999", f"1800.{j:05d}")
    out = tmp_path / "allow"
    gexf = tmp_path / "g2.gexf"
    gexf.write_text("x", encoding="utf-8")
    allowed = {f"{mm}.{i:05d}" for mm in ("2605", "2606", "2607", "2608") for i in range(20)}
    meta = dump_time_elig10(
        g,
        out,
        gexf=gexf,
        gexf_hash="0" * 64,
        elig_k=10,
        test_frac=0.10,
        ret_pos=None,
        allowed_sources=allowed,
        frontier_margin_months=0,
        min_sentence_coverage=0.0,
        gold_quantile_lo=0.0,
        gold_quantile_hi=1.0,
    )
    both = json.loads((out / "train_sources.json").read_text()) + json.loads(
        (out / "test_sources.json").read_text()
    )
    assert "2301.99999" not in both
    assert meta["n_elig_before_source_allowlist"] == meta["n_elig"] + 1


def test_default_split_id_is_test_admit_v1():
    assert DEFAULT_SPLIT_ID == "test_admit_v1"


def test_defaults_apply_admit_filters(tmp_path):
    """Default D: bare graph has no sentences → empty pool."""
    out = tmp_path / "bare"
    gexf = tmp_path / "g.gexf"
    gexf.write_text("x", encoding="utf-8")
    try:
        dump_time_elig10(
            _graph(),
            out,
            gexf=gexf,
            gexf_hash="0" * 64,
            elig_k=10,
            test_frac=0.10,
            ret_pos=None,
        )
        raised = False
    except SystemExit:
        raised = True
    assert raised


def test_defaults_with_sentences_hold_frontier(tmp_path):
    out = tmp_path / "admit"
    gexf = tmp_path / "g.gexf"
    gexf.write_text("x", encoding="utf-8")
    meta = dump_time_elig10(
        _graph(with_sentences=True),
        out,
        gexf=gexf,
        gexf_hash="0" * 64,
        elig_k=10,
        test_frac=0.10,
        ret_pos=None,
    )
    assert meta["split_id"] == "test_admit_v1"
    assert meta["frontier_margin_months"] == 1
    assert meta["min_sentence_coverage"] == 0.95
    assert meta["gold_quantile_lo"] == 0.10
    assert meta["gold_quantile_hi"] == 0.99
    assert meta["n_drop_sentence_coverage"] == 0
    # Uniform outdeg → F keeps all.
    assert meta["n_drop_gold_quantile"] == 0
    test = json.loads((out / "test_sources.json").read_text())
    frontier = json.loads((out / "frontier_sources.json").read_text())
    assert all(s.startswith("2607") for s in test)
    assert all(s.startswith("2608") for s in frontier)


def test_sentence_coverage_drops_partial_sources(tmp_path):
    g = _graph(with_sentences=True)
    bad = "2606.00000"
    edges = list(g.out_edges(bad, data=True))
    u, v, _ = edges[0]
    g.edges[u, v]["sentence"] = ""
    out = tmp_path / "sent"
    gexf = tmp_path / "g.gexf"
    gexf.write_text("x", encoding="utf-8")
    meta = dump_time_elig10(
        g,
        out,
        gexf=gexf,
        gexf_hash="0" * 64,
        elig_k=10,
        test_frac=0.10,
        ret_pos=None,
        frontier_margin_months=0,
        gold_quantile_lo=0.0,
        gold_quantile_hi=1.0,
    )
    both = json.loads((out / "train_sources.json").read_text()) + json.loads(
        (out / "test_sources.json").read_text()
    )
    assert bad not in both
    assert meta["n_drop_sentence_coverage"] == 1


def test_gold_quantile_trims_outdeg_tails(tmp_path):
    g = _graph(with_sentences=True, outdeg=10)
    # Hub and near-floor outliers.
    hub = "2605.00000"
    for j in range(10, 100):
        g.add_edge(hub, f"1800.{j:05d}", sentence=f"hub {j}.")
    low = "2605.00001"
    # Remove edges until outdeg=10 stays; add none — keep at 10.
    # Make a second hub so 0.95 quantile is below 100.
    hub2 = "2605.00002"
    for j in range(10, 80):
        g.add_edge(hub2, f"1700.{j:05d}", sentence=f"hub2 {j}.")

    out = tmp_path / "gold"
    gexf = tmp_path / "g.gexf"
    gexf.write_text("x", encoding="utf-8")
    meta = dump_time_elig10(
        g,
        out,
        gexf=gexf,
        gexf_hash="0" * 64,
        elig_k=10,
        test_frac=0.10,
        ret_pos=None,
        frontier_margin_months=0,
        gold_quantile_lo=0.05,
        gold_quantile_hi=0.95,
    )
    both = json.loads((out / "train_sources.json").read_text()) + json.loads(
        (out / "test_sources.json").read_text()
    )
    assert hub not in both
    assert hub2 not in both
    assert meta["n_drop_gold_quantile"] >= 2
    assert meta["gold_outdeg_hi"] is not None
    assert meta["gold_outdeg_hi"] < 100


def test_legacy_kwargs_match_old_time_elig10(tmp_path):
    out = tmp_path / "legacy"
    gexf = tmp_path / "g.gexf"
    gexf.write_text("x", encoding="utf-8")
    meta = dump_time_elig10(
        _graph(),
        out,
        gexf=gexf,
        gexf_hash="0" * 64,
        elig_k=10,
        test_frac=0.10,
        ret_pos=None,
        **_legacy_kwargs(),
    )
    test = json.loads((out / "test_sources.json").read_text())
    assert meta["split_id"] == "time_elig10_v2"
    assert meta["n_frontier_sources"] == 0
    assert meta["n_drop_sentence_coverage"] == 0
    assert meta["n_drop_gold_quantile"] == 0
    assert all(s.startswith("2608") for s in test)
