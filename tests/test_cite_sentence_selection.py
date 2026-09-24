"""CE candidate text is a function of graph edge order, and must stay that way.

Sorting the citers here (an earlier attempt at making the text order-free) moved
EWM test431 from 6.85 to 1.10 mean hits@10 with the deployed CE: the trainer walks
the same predecessor iterator, so any reordering has to be paired with a retrain.
These tests pin the contract the CE was trained against.
"""

from __future__ import annotations

import networkx as nx

from rwcite.ranker.rr_ranker_ce_sent import collect_cite_sentences


def _graph(edges: list[tuple[str, str]]) -> nx.DiGraph:
    g = nx.DiGraph()
    for citer, citee in edges:
        g.add_edge(citer, citee, sentence=f"{citer} cites it because of X~\\cite{{k}}.")
    return g


def test_follows_graph_edge_order():
    citers = [f"20{i:02d}.0000{j}" for i in range(1, 4) for j in range(1, 4)]
    forward = collect_cite_sentences(
        _graph([(c, "2101.00001") for c in citers]), "2101.00001", max_sents=3
    )
    reverse = collect_cite_sentences(
        _graph([(c, "2101.00001") for c in reversed(citers)]), "2101.00001", max_sents=3
    )
    assert forward and reverse
    assert forward != reverse
    assert citers[0] in forward[0]
    assert citers[-1] in reverse[0]


def test_respects_max_sents_and_excludes_the_query():
    g = _graph([(f"2001.0000{j}", "2101.00001") for j in range(1, 8)])
    got = collect_cite_sentences(
        g, "2101.00001", exclude_citer="2001.00003", max_sents=3
    )
    assert len(got) == 3
    assert not any("2001.00003" in s for s in got)


def test_missing_citee_and_sentenceless_edges_are_empty():
    g = _graph([("2001.00001", "2101.00001")])
    g.add_edge("2001.00002", "2101.00002", sentence="")
    assert collect_cite_sentences(g, "2101.09999", max_sents=3) == []
    assert collect_cite_sentences(g, "2101.00002", max_sents=3) == []
