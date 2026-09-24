"""Unit tests for frontier/test out-edge masking."""

from __future__ import annotations

import networkx as nx
import numpy as np

from rwcite.ranker.rr_graph_mask import masked_citer_ids, strip_out_edges
from rwcite.ranker.rr_ranker import extract_features


def test_strip_out_edges_removes_only_future_sources():
    g = nx.DiGraph()
    g.add_edges_from(
        [
            ("2607.00001", "2001.00001"),
            ("2608.00001", "2001.00001"),
            ("2001.00001", "1901.00001"),
        ]
    )
    g2, n = strip_out_edges(g, {"2607.00001", "2608.00001"})
    assert n == 2
    assert g2.number_of_edges() == 1
    assert list(g2.edges()) == [("2001.00001", "1901.00001")]
    assert g.number_of_edges() == 3  # original untouched


def test_masked_indeg_drops_query_and_future(monkeypatch):
    g = nx.DiGraph()
    # gold cited by query + frontier + unrelated
    g.add_edges_from(
        [
            ("2607.10000", "1801.00001"),  # query (test)
            ("2608.20000", "1801.00001"),  # frontier
            ("1501.00001", "1801.00001"),  # historical
        ]
    )
    node_by = {n: n for n in g.nodes()}
    cand = {"id": "1801.00001", "title": "t", "abstract": "a"}
    monkeypatch.setenv("RR_STRUCT_MASK_QUERY_OUT", "1")
    monkeypatch.setenv("RR_STRUCT_MASK_FUTURE_OUT", "1")
    # inject future set via env file-less path: monkeypatch resolve
    monkeypatch.setenv("RR_STRUCT_FUTURE_SOURCES", "")  # empty file path → use explicit below

    mask = masked_citer_ids(
        query_id="2607.10000",
        mask_query_out=True,
        mask_future_out=True,
        future_sources={"2607.10000", "2608.20000"},
    )
    feat = extract_features(
        title="q",
        abstract="q",
        cand=cand,
        support={},
        dense_rank={},
        focused_hubs=set(),
        graph=g,
        node_by=node_by,
        emb_sim=0.0,
        query_id="2607.10000",
        mask_citers=mask,
    )
    # FEATURE_NAMES: log_indeg at index 4 → only historical citer remains → log1p(1)
    assert abs(float(feat[4]) - float(np.log1p(1))) < 1e-5
