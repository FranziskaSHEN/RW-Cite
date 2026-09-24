from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

CLI_MODULES = [
    "rwcite.cli.fetch_metadata",
    "rwcite.cli.update_corpus",
    "rwcite.cli.graph_pipeline",
    "rwcite.cli.dump_split",
    "rwcite.cli.build_rr_jsonl",
    "rwcite.cli.train_ce_sent",
    "rwcite.cli.train_citelink",
    "rwcite.cli.eval_pool_ranker",
    "rwcite.cli.gate_citelink",
    "rwcite.cli.validate_graph",
    "rwcite.cli.diag_retrieval",
    "rwcite.cli.rewrite_sentences",
    "rwcite.cli.extract_graph",
    "rwcite.cli.sweep_blend",
    "rwcite.cli.dump_ce_citelink_scores",
    "rwcite.cli.dump_rr_l0_pools",
    "rwcite.cli.build_rr_adapter_v6",
    "rwcite.cli.train_rr_adapter",
    "rwcite.cli.eval_rr_adapter_cite",
    "rwcite.cli.summarize_relevance_assessment",
]


@pytest.mark.parametrize("mod", CLI_MODULES)
def test_cli_help(mod: str):
    proc = subprocess.run(
        [sys.executable, "-m", mod, "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "usage:" in proc.stdout.lower() or "Usage" in proc.stdout


def test_script_wrappers_exist(repo_root: Path):
    for name in (
        "run_domain_graph_pipeline.py",
        "fetch_arxiv_metadata.py",
        "dump_domain_split.py",
        "gate_rr_citelink_quality.py",
    ):
        assert (repo_root / "scripts" / name).is_file()
