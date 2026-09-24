"""Publication-month utilities for temporally controlled citation evidence."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal

import networkx as nx

from rwcite.ranker.reference_recommend import normalize_arxiv_id

_MODERN_ID = re.compile(r"^(?P<yy>\d{2})(?P<mm>\d{2})\.\d+$")
_LEGACY_ID = re.compile(r"^[^/]+/(?P<yy>\d{2})(?P<mm>\d{2})\d+$")


@dataclass(frozen=True, order=True)
class PaperMonth:
    year: int
    month: int

    def __post_init__(self) -> None:
        if not 1 <= self.month <= 12:
            raise ValueError(f"invalid month: {self.month}")

    @property
    def label(self) -> str:
        return f"{self.year:04d}-{self.month:02d}"


def paper_month(paper_id: str) -> PaperMonth | None:
    """Return the arXiv submission month encoded in ``paper_id``."""
    pid = normalize_arxiv_id(paper_id)
    match = _MODERN_ID.match(pid)
    legacy = False
    if match is None:
        match = _LEGACY_ID.match(pid)
        legacy = match is not None
    if match is None:
        return None
    yy = int(match.group("yy"))
    month = int(match.group("mm"))
    if not 1 <= month <= 12:
        return None
    year = 1900 + yy if legacy and yy >= 90 else 2000 + yy
    return PaperMonth(year, month)


def temporal_citer_filter(
    query_id: str,
    *,
    same_month_policy: str = "keep",
    unknown_citer_policy: str = "keep",
):
    """Create a predicate admitting citation evidence available by query month."""
    if same_month_policy not in {"keep", "drop"}:
        raise ValueError("same_month_policy must be 'keep' or 'drop'")
    if unknown_citer_policy not in {"keep", "drop"}:
        raise ValueError("unknown_citer_policy must be 'keep' or 'drop'")
    cutoff = paper_month(query_id)
    if cutoff is None:
        raise ValueError(f"cannot infer query month from arXiv id: {query_id!r}")

    def allowed(citer_id: str) -> bool:
        month = paper_month(citer_id)
        if month is None:
            return unknown_citer_policy == "keep"
        if month < cutoff:
            return True
        if month == cutoff:
            return same_month_policy == "keep"
        return False

    return allowed


TemporalSelectionPolicy = Literal["stride", "recent", "spread"]


def _stable_record_key(record: tuple[str, str]) -> tuple[int, int, str]:
    citer, _ = record
    month = paper_month(citer)
    if month is None:
        return (0, 0, normalize_arxiv_id(citer))
    return (month.year, month.month, normalize_arxiv_id(citer))


def select_temporal_cite_records(
    graph: Any,
    citee_id: str,
    query_id: str,
    *,
    exclude_citer: str = "",
    max_sents: int = 3,
    same_month_policy: Literal["keep", "drop"] = "keep",
    unknown_citer_policy: Literal["keep", "drop"] = "keep",
    selection_policy: TemporalSelectionPolicy = "recent",
) -> list[tuple[str, str]]:
    """Select reproducibly ordered citation evidence available by query month."""
    from rwcite.ranker.rr_ranker_ce_sent import (
        collect_cite_sentence_pool,
        select_cite_sentence_records,
    )

    allowed = temporal_citer_filter(
        query_id,
        same_month_policy=same_month_policy,
        unknown_citer_policy=unknown_citer_policy,
    )
    records = collect_cite_sentence_pool(
        graph,
        citee_id,
        exclude_citer=exclude_citer,
        citer_filter=allowed,
    )
    if selection_policy == "stride":
        return select_cite_sentence_records(records, max_sents=max_sents)
    ordered = sorted(records, key=_stable_record_key)
    if selection_policy == "recent":
        return list(reversed(ordered))[:max_sents]
    if selection_policy == "spread":
        return select_cite_sentence_records(ordered, max_sents=max_sents)
    raise ValueError(f"unknown temporal selection policy: {selection_policy}")


def classify_citer_month(citer_id: str, query_id: str) -> str:
    """Classify a citer relative to a query as prior, same, later, or unknown."""
    query_month = paper_month(query_id)
    citer_month = paper_month(citer_id)
    if query_month is None or citer_month is None:
        return "unknown"
    if citer_month < query_month:
        return "prior"
    if citer_month == query_month:
        return "same"
    return "later"


def hide_query_out_edges(graph: Any, query_id: str):
    """Return a read-only graph view without the query's outgoing edges."""
    query = normalize_arxiv_id(query_id)
    return nx.subgraph_view(
        graph,
        filter_edge=lambda source, target: normalize_arxiv_id(str(source)) != query,
    )


def temporal_graph_view(
    graph: Any,
    query_id: str,
    *,
    same_month_policy: Literal["keep", "drop"] = "drop",
    unknown_node_policy: Literal["keep", "drop"] = "drop",
):
    """Return a read-only graph containing evidence admissible for a query."""
    query = normalize_arxiv_id(query_id)
    allowed = temporal_citer_filter(
        query,
        same_month_policy=same_month_policy,
        unknown_citer_policy=unknown_node_policy,
    )

    def node_ok(node: Any) -> bool:
        pid = normalize_arxiv_id(str(node))
        return pid == query or allowed(pid)

    def edge_ok(source: Any, target: Any) -> bool:
        source_id = normalize_arxiv_id(str(source))
        return source_id != query and allowed(source_id)

    return nx.subgraph_view(graph, filter_node=node_ok, filter_edge=edge_ok)
