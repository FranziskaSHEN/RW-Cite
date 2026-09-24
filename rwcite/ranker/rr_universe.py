"""Focused large universe for RR cold-start (not fed to the LLM).

Pipeline: multi-query BGE + light PRF + focused hubs (cites_dense × 1/log outdeg)
+ weak 2-hop. Does not use the query paper's own out-edges.
"""

from __future__ import annotations

import math
import os
import re
from collections import defaultdict
from typing import Any, Callable

from rwcite.ranker.reference_recommend import (
    _STOP,
    _norm,
    _title_excluded,
    normalize_arxiv_id,
)


def rr_universe_params() -> dict[str, int | float]:
    def _i(name: str, default: int) -> int:
        raw = (os.environ.get(name) or "").strip()
        if not raw:
            return default
        try:
            return max(0, int(raw))
        except ValueError:
            return default

    return {
        "m": _i("RR_CAND_M", 200),
        "prf_k": _i("RR_PRF_K", 10),
        "prf_m": _i("RR_PRF_M", 80),
        "bridge_seeds": _i("RR_BRIDGE_SEEDS", 120),
        "hub_cap": _i("RR_HUB_CAP", 400),
        "top_dense": _i("RR_TOP_DENSE", 60),
        "min_od_direct": _i("RR_MIN_OD_DIRECT", 2),
        "pred_min_od": _i("RR_PRED_MIN_OD", 2),
        "hop2_k": _i("RR_HOP2_K", 100),
    }


def _toks(text: str) -> set[str]:
    return {
        t
        for t in re.findall(r"[a-z0-9]+", (text or "").lower())
        if t not in _STOP and len(t) > 2
    }


def _node_by_id(graph: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if graph is None:
        return out
    for node in graph.nodes():
        out[normalize_arxiv_id(str(node))] = node
    return out


def _resolve(
    graph: Any, node_by: dict[str, Any], pid: str
) -> tuple[str, str]:
    node = node_by.get(pid)
    if node is None or graph is None:
        return "", ""
    attrs = graph.nodes[node]
    title = (attrs.get("title") or attrs.get("label") or "").strip()
    abstract = (attrs.get("abstract") or "")[:700]
    return title, abstract


def _bge_append(
    search_fn: Callable[[str, int], list[dict[str, Any]]],
    query: str,
    limit: int,
    excl: str,
    excl_title: str,
    got: set[str],
    dense: list[str],
) -> None:
    q = (query or "").strip()
    if not q:
        return
    for r in search_fn(q, limit) or []:
        pid = normalize_arxiv_id(r.get("id") or r.get("paper_id") or "")
        if not pid or pid == excl or pid in got:
            continue
        title = (r.get("title") or "").strip()
        if _title_excluded(title, excl_title):
            continue
        got.add(pid)
        dense.append(pid)


def build_rr_universe(
    title: str,
    abstract: str,
    graph: Any,
    *,
    search_fn: Callable[[str, int], list[dict[str, Any]]],
    exclude_id: str = "",
    multi_query: bool = True,
    prf: bool = True,
) -> dict[str, Any]:
    """Build large candidate universe + structural signals for ranking.

    Returns dict with keys:
      universe: list[dict]  (id, title, abstract, source)
      dense_ids: list[str]
      support: dict[str, float]
      focused_hubs: set[str]
      dense_rank: dict[str, int]
      meta: dict
    """
    params = rr_universe_params()
    excl = normalize_arxiv_id(exclude_id)
    excl_title = _norm(title)
    ab = (abstract or "").strip()
    node_by = _node_by_id(graph)

    qs = [f"{title}\n{ab}".strip() or title]
    if multi_query:
        if title.strip():
            qs.append(title.strip())
        if len(ab) > 80:
            qs.append(ab[:600])

    got: set[str] = set()
    dense: list[str] = []
    m = int(params["m"])
    for q in qs:
        _bge_append(search_fn, q, m, excl, excl_title, got, dense)

    if prf:
        prf_k = int(params["prf_k"])
        prf_m = int(params["prf_m"])
        for pid in list(dense[:prf_k]):
            t, _ = _resolve(graph, node_by, pid)
            if t:
                _bge_append(search_fn, t, prf_m, excl, excl_title, got, dense)

    dense_rank = {p: i for i, p in enumerate(dense)}
    top_dense = set(dense[: int(params["top_dense"])])

    topic_hubs: list[str] = []
    for pid in dense[: int(params["bridge_seeds"])]:
        node = node_by.get(pid)
        if node is None or graph is None:
            continue
        try:
            od = int(graph.out_degree(node))
        except Exception:  # noqa: BLE001
            continue
        if od >= int(params["min_od_direct"]):
            topic_hubs.append(pid)
        try:
            for u in graph.predecessors(node):
                if int(graph.out_degree(u)) >= int(params["pred_min_od"]):
                    topic_hubs.append(normalize_arxiv_id(str(u)))
        except Exception:  # noqa: BLE001
            continue
    topic_hubs = list(dict.fromkeys(topic_hubs))[: int(params["hub_cap"])]

    focused: set[str] = set()
    support: dict[str, float] = defaultdict(float)
    for hid in topic_hubs:
        node = node_by.get(hid)
        if node is None or graph is None:
            continue
        try:
            outs = [normalize_arxiv_id(str(v)) for v in graph.successors(node)]
            od = max(1, int(graph.out_degree(node)))
        except Exception:  # noqa: BLE001
            continue
        cites_dense = sum(1 for v in outs if v in top_dense)
        if cites_dense >= 1:
            focused.add(hid)
        w = 1.0 + (1.5 if cites_dense >= 2 else 0.0) + (0.5 if cites_dense == 1 else 0.0)
        if cites_dense == 0:
            w *= 0.12
        w /= math.log2(2.0 + od)
        for vid in outs:
            if vid and vid != excl and vid != hid:
                support[vid] += w

    # Weak 2-hop from high-support papers that have outs.
    hop2_k = int(params["hop2_k"])
    if hop2_k > 0 and graph is not None:
        cands = sorted(
            (
                (p, s)
                for p, s in support.items()
                if s >= 0.5
                and node_by.get(p) is not None
                and int(graph.out_degree(node_by[p])) >= 1
            ),
            key=lambda x: -x[1],
        )[:hop2_k]
        for pid, _ in cands:
            node = node_by[pid]
            od = max(1, int(graph.out_degree(node)))
            w = 0.2 / math.log2(2.0 + od)
            for v in graph.successors(node):
                vid = normalize_arxiv_id(str(v))
                if vid and vid != excl and vid != pid:
                    support[vid] += w

    universe_ids = list(dict.fromkeys(dense + list(support.keys())))
    universe: list[dict[str, Any]] = []
    for pid in universe_ids:
        if excl and pid == excl:
            continue
        t, a = _resolve(graph, node_by, pid)
        if not t:
            # dense hit may carry title if not on graph — skip graph-less for ranker features
            continue
        if _title_excluded(t, excl_title):
            continue
        src = "dense" if pid in dense_rank else "proxy"
        universe.append(
            {
                "id": pid,
                "title": t,
                "abstract": a,
                "source": src,
                "support": float(support.get(pid, 0.0)),
            }
        )

    return {
        "universe": universe,
        "dense_ids": dense,
        "support": dict(support),
        "focused_hubs": focused,
        "dense_rank": dense_rank,
        "node_by": node_by,
        "meta": {
            "universe_size": len(universe),
            "dense_size": len(dense),
            "hub_size": len(topic_hubs),
            "focused_hubs": len(focused),
            "params": params,
        },
    }


def jaccard_query_cand(query_text: str, cand_title: str, cand_abs: str) -> float:
    a = _toks(query_text)
    b = _toks(f"{cand_title} {cand_abs}")
    if not a or not b:
        return 0.0
    return len(a & b) / max(1, len(a | b))
