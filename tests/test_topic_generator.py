"""Smoke tests for topic generator helpers."""

from rwcite.retrieve.retriever.topic_generator import (
    generate_topics_heuristic,
    generate_topics_batch,
)


def test_heuristic_topics_shape():
    levels = generate_topics_heuristic(
        "Transmon qubit readout",
        "We study superconducting qubit fidelity.",
        "quant-ph cond-mat.mes-hall",
    )
    assert set(levels) == {"Level 1", "Level 2", "Level 3"}
    for key in levels:
        assert len(levels[key]) == 3


def test_batch_heuristic():
    papers = [
        {
            "id": "2501.00001",
            "title": "Test paper",
            "abstract": "Abstract text about robots.",
            "categories": "cs.RO",
        }
    ]
    out = generate_topics_batch(papers, backend="heuristic", workers=1)
    assert len(out) == 1
    assert out[0]["paper_id"] == "2501.00001"
    assert len(out[0]["Level 1"]) == 3
