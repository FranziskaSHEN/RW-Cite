from __future__ import annotations

from pathlib import Path

import networkx as nx

from rwcite.graph import extract_rr_graph as eg
from rwcite.graph.domain_graph_resume import pick_zip_dirs


def test_load_title_index_from_jsonl(tiny_metadata_jsonl: Path):
    eg._META_SOURCE_CACHE.clear()
    title_index, known = eg.load_title_index(tiny_metadata_jsonl)
    assert "2101.00001" in known
    assert "2205.12345" in known
    # normalized title should map to id
    assert any(v == "2101.00001" for v in title_index.values())
    rows = eg.load_meta_rows(tiny_metadata_jsonl, ["2101.00001", "missing"])
    assert rows["2101.00001"]["title"].startswith("First Paper")
    assert "missing" not in rows


def test_ensure_nodes_and_add_edges(tiny_metadata_jsonl: Path):
    eg._META_SOURCE_CACHE.clear()
    g = nx.DiGraph()
    cache: dict = {}
    n = eg.ensure_nodes(g, ["2101.00001", "2205.12345"], tiny_metadata_jsonl, cache)
    assert n == 2
    assert g.nodes["2101.00001"]["title"].startswith("First Paper")
    ne, total = eg.add_edges(
        g, [("2101.00001", "2205.12345", "cites prior work.")]
    )
    assert total == 1 and ne == 1
    assert g.has_edge("2101.00001", "2205.12345")


def test_pick_zip_dirs_prefers_local_coverage(tmp_path: Path):
    parent = tmp_path / "dom_retrieval"
    z1 = parent / "research_papers_zip"
    z1.mkdir(parents=True)
    (z1 / "a.tar.gz").write_bytes(b"x")
    (z1 / "b.tar.gz").write_bytes(b"x")
    asset = tmp_path / "assets"
    z2 = asset / "dom_retrieval" / "research_papers_zip"
    z2.mkdir(parents=True)
    (z2 / "c.tar.gz").write_bytes(b"x")
    dirs = pick_zip_dirs(papers_parent=parent, asset_root=asset)
    assert dirs[0].resolve() == z1.resolve()
    assert any(p.resolve() == z2.resolve() for p in dirs)
