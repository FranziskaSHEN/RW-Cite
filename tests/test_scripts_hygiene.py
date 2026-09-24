from __future__ import annotations

import subprocess
from pathlib import Path

import pytest


def test_shell_scripts_lf_and_syntax(repo_root: Path):
    sh_files = sorted((repo_root / "scripts").glob("*.sh"))
    assert sh_files, "expected shell scripts"
    for path in sh_files:
        data = path.read_bytes()
        assert b"\r" not in data, f"CRLF in {path.name}"
        proc = subprocess.run(
            ["bash", "-n", str(path)],
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode == 0, f"{path.name}: {proc.stderr}"


def test_env_and_c2s_defaults(repo_root: Path):
    c2s = (repo_root / "scripts" / "run_domain_rr_c2s.sh").read_text(encoding="utf-8")
    assert 'BASE="${BASE:-models/base/scibert_scivocab_uncased}"' in c2s
    assert 'DATA_TAG="${DATA_TAG:-sclean}"' in c2s
    assert 'SKIP_BLEND="${SKIP_BLEND:-1}"' in c2s
    assert "rr-pool-ranker-ce-v1" not in c2s
    assert "arxiv_metadata.db" not in c2s

    boot = (repo_root / "scripts" / "bootstrap_assets.sh").read_text(encoding="utf-8")
    assert "arxiv_metadata.db" not in boot
    assert "rr-pool-ranker-ce-v1" not in boot
    assert "scibert_scivocab_uncased" in boot

    ranker = (repo_root / "scripts" / "run_domain_rr_ranker.sh").read_text(
        encoding="utf-8"
    )
    assert 'BASE="${BASE:-models/base/scibert_scivocab_uncased}"' in ranker


def test_no_legacy_product_names_in_package(repo_root: Path):
    banned = ("litbench", "authorsearch", "AuthorSearch", "LitBench", "as_native")
    hits: list[str] = []
    for path in (repo_root / "rwcite").rglob("*.py"):
        text = path.read_text(encoding="utf-8", errors="replace").lower()
        for b in banned:
            if b.lower() in text:
                hits.append(f"{path.relative_to(repo_root)}:{b}")
    # allow none
    assert hits == []


def test_no_private_infrastructure_paths_in_release_sources(repo_root: Path):
    """Keep workstation, cluster, and institutional endpoints out of releases."""
    banned = (
        "/share/home/",
        "/share2/",
        "c:\\users\\",
        "llm.bnuzh.edu.cn",
    )
    roots = (
        repo_root / "rwcite",
        repo_root / "scripts",
        repo_root / "experiments",
        repo_root / "configs",
        repo_root / "docs",
    )
    candidates = [repo_root / "README.md", repo_root / "LICENSE"]
    for root in roots:
        if root.is_dir():
            candidates.extend(path for path in root.rglob("*") if path.is_file())

    hits: list[str] = []
    for path in candidates:
        if "__pycache__" in path.parts:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="strict").lower()
        except (UnicodeDecodeError, OSError):
            continue
        for fragment in banned:
            if fragment in text:
                hits.append(f"{path.relative_to(repo_root)}:{fragment}")
    assert hits == []


@pytest.mark.parametrize(
    "name",
    [
        "scripts/run_domain_graph_build.sh",
        "scripts/run_domain_rr_c2s.sh",
        "scripts/run_domain_rr_adapter_v6.sh",
        "scripts/run_metadata_fetch.sh",
        "scripts/bootstrap_assets.sh",
        "configs/env_domain_rr.sh",
    ],
)
def test_release_entrypoints_exist(repo_root: Path, name: str):
    assert (repo_root / name).is_file()
