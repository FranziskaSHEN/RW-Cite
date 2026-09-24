"""Shared fixtures for RW-Cite unit tests."""

from __future__ import annotations

import json
from pathlib import Path

import networkx as nx
import pytest


@pytest.fixture
def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture
def tiny_metadata_jsonl(tmp_path: Path) -> Path:
    path = tmp_path / "arxiv-metadata-oai-snapshot.json"
    rows = [
        {
            "id": "2101.00001",
            "title": "First Paper About Quantum Computing",
            "abstract": "Abstract one.",
            "update_date": "2021-01-02",
        },
        {
            "id": "2205.12345",
            "title": "Second Paper On Superconducting Qubits",
            "abstract": "Abstract two.",
            "update_date": "2022-05-10",
        },
        {
            "id": "2301.00099",
            "title": "Third Survey Of Error Correction",
            "abstract": "Abstract three.",
            "update_date": "2023-01-15",
        },
    ]
    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
        encoding="utf-8",
    )
    return path


@pytest.fixture
def tiny_digraph() -> nx.DiGraph:
    """10 sources with outdeg>=2 plus citees; usable for time_elig10 with elig_k=2."""
    g = nx.DiGraph()
    for i in range(10):
        src = f"2{i:02d}01.0000{i}"
        g.add_node(src, title=f"Source {i}", abstract="")
        for j in range(2):
            tgt = f"1801.{10000 + i * 10 + j}"
            g.add_node(tgt, title=f"Tgt {i}-{j}", abstract="")
            g.add_edge(src, tgt, sentence=f"cites {tgt}.")
    return g
