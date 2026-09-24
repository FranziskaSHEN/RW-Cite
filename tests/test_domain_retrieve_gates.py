from __future__ import annotations

import numpy as np

from rwcite.graph.domain_retrieve import (
    apply_dual_gate,
    apply_year_window,
    arxiv_year,
    default_tau,
    merge_prototype_scores,
    normalize_queries_arg,
    resolve_min_arxiv_year,
)


def test_arxiv_year_modern_and_legacy():
    assert arxiv_year("2101.00001") == 2021
    assert arxiv_year("9901.00001") == 1999
    assert arxiv_year("hep-th/9901001") is None


def test_resolve_min_arxiv_year_recent():
    assert resolve_min_arxiv_year({}, recent_years=10, as_of_year=2026) == 2016
    assert resolve_min_arxiv_year({"min_arxiv_year": 2018}, recent_years=10) == 2018
    assert resolve_min_arxiv_year({}, recent_years=0) is None


def test_apply_year_window_drops_old_and_unknown():
    ids = ["2101.00001", "1501.00001", "hep-th/9901001"]
    scores = np.array([0.9, 0.8, 0.7])
    kept, sc, meta = apply_year_window(ids, scores, min_year=2016)
    assert kept == ["2101.00001"]
    assert meta["n_dropped_old"] == 1
    assert meta["n_dropped_unknown"] == 1
    assert sc.shape == (1,)


def test_default_tau_and_dual_gate():
    scores = [0.9, 0.8, 0.7, 0.2]
    tau = default_tau(scores, head_k=3)
    assert abs(tau - (0.8 - 0.05)) < 1e-9
    assert abs(default_tau(scores, head_k=3, margin=0.08) - (0.8 - 0.08)) < 1e-9
    ids = ["a", "b", "c", "d"]
    # 0.7 < tau(0.75) → dropped by score gate; rank still limited by n_max
    ranked, meta = apply_dual_gate(ids, np.array(scores), n_max=3, tau=tau)
    assert [x[0] for x in ranked] == ["a", "b"]
    assert meta["n_after_gate"] == 2
    assert meta["tau_margin"] == 0.05
    # wider margin admits borderline score 0.73 that default margin drops
    scores2 = [0.9, 0.8, 0.73, 0.2]
    ranked_def, _ = apply_dual_gate(
        ids, np.array(scores2), n_max=4, head_k=3, tau_margin=0.05
    )
    ranked_m, meta_m = apply_dual_gate(
        ids, np.array(scores2), n_max=4, head_k=3, tau_margin=0.08
    )
    assert [x[0] for x in ranked_def] == ["a", "b"]  # tau=0.75
    assert [x[0] for x in ranked_m] == ["a", "b", "c"]  # tau=0.72
    assert abs(meta_m["tau"] - 0.72) < 1e-9
    assert meta_m["tau_margin"] == 0.08
    ranked2, _ = apply_dual_gate(ids, np.array(scores), n_max=3, apply_tau=False)
    assert [x[0] for x in ranked2] == ["a", "b", "c"]


def test_merge_prototype_scores_takes_max():
    q = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=float)
    c = np.array([[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]], dtype=float)
    best = merge_prototype_scores(q, c)
    assert best.shape == (3,)
    assert best[0] > 0.99
    assert best[1] > 0.99


def test_normalize_queries_arg():
    assert normalize_queries_arg("a|b", ["c"]) == ["c", "a", "b"]
    cfg = {"retriever": {"queries": ["q1", "q2"]}}
    assert normalize_queries_arg(None, None, cfg) == ["q1", "q2"]
