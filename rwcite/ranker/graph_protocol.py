"""Split-aware graph views for citation recommendation experiments."""

from __future__ import annotations

from typing import Any, Literal

from rwcite.ranker.temporal_evidence import PaperMonth, paper_month, temporal_graph_view

GraphProtocol = Literal["full", "frozen", "rolling"]


def parse_month(value: str) -> PaperMonth:
    """Parse an exclusive ``YYYY-MM`` graph cutoff."""
    try:
        year_s, month_s = value.strip().split("-", 1)
        return PaperMonth(int(year_s), int(month_s))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid month {value!r}; expected YYYY-MM") from exc


def infer_frozen_cutoff(test_query_ids: list[str]) -> PaperMonth:
    """Use the earliest dated test query as the frozen snapshot boundary."""
    dated = [(paper_month(pid), pid) for pid in test_query_ids]
    missing = [pid for month, pid in dated if month is None]
    if missing:
        preview = ", ".join(missing[:5])
        raise ValueError(f"cannot date {len(missing)} test queries: {preview}")
    if not dated:
        raise ValueError("cannot infer frozen cutoff from an empty test split")
    return min(month for month, _ in dated if month is not None)


def frozen_graph_view(
    graph: Any,
    cutoff: PaperMonth,
    *,
    unknown_node_policy: Literal["keep", "drop"] = "drop",
):
    """Return a read-only graph snapshot strictly before ``cutoff``."""
    import networkx as nx

    if unknown_node_policy not in {"keep", "drop"}:
        raise ValueError("unknown_node_policy must be 'keep' or 'drop'")

    def allowed(node: Any) -> bool:
        month = paper_month(str(node))
        if month is None:
            return unknown_node_policy == "keep"
        return month < cutoff

    return nx.subgraph_view(
        graph,
        filter_node=allowed,
        filter_edge=lambda source, target: allowed(source) and allowed(target),
    )


def graph_for_query(
    graph: Any,
    query_id: str,
    *,
    protocol: GraphProtocol,
    frozen_cutoff: PaperMonth | None = None,
    same_month_policy: Literal["keep", "drop"] = "drop",
    unknown_node_policy: Literal["keep", "drop"] = "drop",
):
    """Select the graph visible to one query under a named protocol."""
    if protocol == "rolling":
        return temporal_graph_view(
            graph,
            query_id,
            same_month_policy=same_month_policy,
            unknown_node_policy=unknown_node_policy,
        )
    if protocol == "frozen":
        if frozen_cutoff is None:
            raise ValueError("frozen protocol requires frozen_cutoff")
        return frozen_graph_view(
            graph,
            frozen_cutoff,
            unknown_node_policy=unknown_node_policy,
        )
    if protocol == "full":
        from rwcite.ranker.temporal_evidence import hide_query_out_edges

        return hide_query_out_edges(graph, query_id)
    raise ValueError(f"unknown graph protocol: {protocol}")
