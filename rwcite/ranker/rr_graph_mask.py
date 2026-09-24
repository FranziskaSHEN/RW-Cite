"""Mask query / future-source out-edges for struct features (frontier+test leakage).

See docs/GRAPH_FRONTIER_TEST_IMPACT.zh-CN.md §§4.1–4.3.

Env:
  RR_STRUCT_MASK_QUERY_OUT=1     ignore query→cand when computing indeg/cocite
  RR_STRUCT_MASK_FUTURE_OUT=1    ignore out-edges from test∪frontier sources
  RR_SPLIT_DIR=…/data/splits     load test_sources.json + frontier_sources.json
  RR_STRUCT_FUTURE_SOURCES=path  optional JSON list/dict of future source ids
  RR_GEXF_OVERRIDE=path          DomainResource loads this GEXF instead of domains.yaml
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

from rwcite.ranker.reference_recommend import normalize_arxiv_id


def _env_flag(name: str, default: bool = False) -> bool:
    raw = (os.environ.get(name) or "").strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")


def load_id_list(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return {normalize_arxiv_id(str(x)) for x in data if str(x).strip()}
    if isinstance(data, dict):
        return {normalize_arxiv_id(str(x)) for x in data.keys()}
    return set()


@lru_cache(maxsize=8)
def future_sources_from_split(split_dir: str) -> frozenset[str]:
    d = Path(split_dir)
    ids: set[str] = set()
    ids |= load_id_list(d / "test_sources.json")
    ids |= load_id_list(d / "frontier_sources.json")
    # legacy names
    ids |= load_id_list(d / "test_nodes.json")
    return frozenset(x for x in ids if x)


@lru_cache(maxsize=4)
def future_sources_from_file(path: str) -> frozenset[str]:
    return frozenset(load_id_list(Path(path)))


def resolve_future_sources() -> set[str]:
    explicit = (os.environ.get("RR_STRUCT_FUTURE_SOURCES") or "").strip()
    if explicit:
        return set(future_sources_from_file(explicit))
    split = (os.environ.get("RR_SPLIT_DIR") or "").strip()
    if split:
        return set(future_sources_from_split(split))
    return set()


def mask_params() -> dict[str, Any]:
    return {
        "mask_query_out": _env_flag("RR_STRUCT_MASK_QUERY_OUT", False),
        "mask_future_out": _env_flag("RR_STRUCT_MASK_FUTURE_OUT", False),
        "future_sources": resolve_future_sources()
        if _env_flag("RR_STRUCT_MASK_FUTURE_OUT", False)
        else set(),
    }


def masked_citer_ids(
    *,
    query_id: str = "",
    mask_query_out: bool | None = None,
    mask_future_out: bool | None = None,
    future_sources: Iterable[str] | None = None,
) -> set[str]:
    """Citers whose out-edges must be ignored for struct indeg/cocite."""
    if mask_query_out is None or mask_future_out is None or future_sources is None:
        p = mask_params()
        if mask_query_out is None:
            mask_query_out = bool(p["mask_query_out"])
        if mask_future_out is None:
            mask_future_out = bool(p["mask_future_out"])
        if future_sources is None:
            future_sources = p["future_sources"]
    out: set[str] = set()
    q = normalize_arxiv_id(query_id) if query_id else ""
    if mask_future_out:
        out |= {normalize_arxiv_id(str(x)) for x in future_sources}
    if mask_query_out and q:
        out.add(q)
    out.discard("")
    return out


def strip_out_edges(graph: Any, source_ids: Iterable[str]) -> tuple[Any, int]:
    """Return a copy of ``graph`` with all out-edges from ``source_ids`` removed."""
    ids = {normalize_arxiv_id(str(x)) for x in source_ids if str(x).strip()}
    g = graph.copy()
    if not g.is_directed():
        g = g.to_directed()
    node_map = {normalize_arxiv_id(str(n)): n for n in g.nodes()}
    removed = 0
    for uid in ids:
        node = node_map.get(uid)
        if node is None:
            continue
        for v in list(g.successors(node)):
            g.remove_edge(node, v)
            removed += 1
    return g, removed


def export_nofuture_gexf(
    *,
    in_gexf: Path,
    out_gexf: Path,
    split_dir: Path,
) -> dict[str, Any]:
    import networkx as nx

    future = set(future_sources_from_split(str(split_dir.resolve())))
    g = nx.read_gexf(str(in_gexf), node_type=None, relabel=False, version="1.2draft")
    if not g.is_directed():
        g = g.to_directed()
    n_edges_before = g.number_of_edges()
    g2, removed = strip_out_edges(g, future)
    out_gexf.parent.mkdir(parents=True, exist_ok=True)
    nx.write_gexf(g2, str(out_gexf))
    meta = {
        "in_gexf": str(in_gexf),
        "out_gexf": str(out_gexf),
        "split_dir": str(split_dir),
        "n_future_sources": len(future),
        "n_nodes": g2.number_of_nodes(),
        "n_edges_before": n_edges_before,
        "n_edges_after": g2.number_of_edges(),
        "n_edges_removed": int(removed),
    }
    meta_path = out_gexf.with_suffix(out_gexf.suffix + ".nofuture_meta.json")
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return meta
