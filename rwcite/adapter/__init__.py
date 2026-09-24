"""Downstream citation-sentence generation and relevance-analysis utilities."""

from __future__ import annotations

ADAPTER_REL = "adapter/rr_v6"
ADAPTER_REL_V7 = "adapter/rr_v7"
POOL_SOURCE = "l0_rrf_3.1"

# Compatibility constants for archived reranking experiments.
DECIDER_TASKS = frozenset({"rerank30", "rerank10", "cite_one", "cite_bundle"})
V7_ALL_TASKS = DECIDER_TASKS

RELEVANCE_LABELS = frozenset({"core", "related", "peripheral", "none"})
ANALYSIS_MAX_REASON_CHARS = 240
NEG_REASON_TEMPLATE = (
    "Unlikely core citation: no clear methodological or problem-setting link to Paper A."
)
LABEL_POLICY_CITE_AUX = "cite_aux_v7"
LABEL_POLICY_RERANK = "rerank_v7"
