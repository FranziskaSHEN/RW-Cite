"""Unit tests for eval_pool_ranker timing aggregation."""

from __future__ import annotations

from rwcite.cli.eval_pool_ranker import _summary_from_details, _timing_agg


def test_timing_agg_mean_and_order():
    details = [
        {
            "timings": {
                "universe_s": 1.0,
                "ce_score_s": 2.0,
                "total_s": 4.0,
            }
        },
        {
            "timings": {
                "universe_s": 3.0,
                "ce_score_s": 4.0,
                "total_s": 8.0,
                "struct_shortlist_s": 0.5,
            }
        },
    ]
    agg = _timing_agg(details)
    assert list(agg)[0] == "universe_s"
    assert "struct_shortlist_s" in agg
    assert abs(agg["universe_s"]["mean_s"] - 2.0) < 1e-9
    assert abs(agg["ce_score_s"]["sum_s"] - 6.0) < 1e-9
    assert agg["total_s"]["n"] == 2


def test_summary_includes_timings():
    details = [
        {
            "universe_recall": 1.0,
            "pool_recall@30": 0.5,
            "pool_over_u": 0.5,
            "hits_at_10": 5,
            "hits_at_30": 8,
            "hits_at_50": 9,
            "timings": {"total_s": 1.5, "universe_s": 0.2},
        }
    ]
    summary = _summary_from_details(details, pool_n=30, gate_thr=8)
    assert "timings" in summary
    assert abs(summary["timings"]["total_s"]["mean_s"] - 1.5) < 1e-9
