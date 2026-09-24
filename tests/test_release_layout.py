from __future__ import annotations

from pathlib import Path


def test_public_release_destinations_are_separate(repo_root: Path) -> None:
    github = repo_root / "releases" / "github"
    huggingface = repo_root / "releases" / "huggingface"

    assert (github / "README.md").is_file()
    assert (huggingface / "README.md").is_file()
    assert (huggingface / "corpus" / "README.md").is_file()
    assert (huggingface / "domains" / "README.md").is_file()


def test_release_builder_covers_code_corpus_and_domains(repo_root: Path) -> None:
    script = (repo_root / "scripts" / "prepare_public_releases.sh").read_text(
        encoding="utf-8"
    )
    assert '"$ROOT/releases/github"' in script
    assert '"$ROOT/releases/huggingface/corpus"' in script
    assert '"$ROOT/releases/huggingface/domains"' in script
    assert "pack_github_release.sh" in script
    assert "pack_datasets_release.sh" in script
    assert "pack_domain_release.sh" in script
    assert "stage_prepacked_domains.py" in script
    assert "--all-paper-domains" in script
    assert "BUILD_CORPUS=0" in script
    assert "--with-corpus" in script


def test_code_pack_excludes_generated_data_payloads(repo_root: Path) -> None:
    script = (repo_root / "scripts" / "pack_github_release.sh").read_text(
        encoding="utf-8"
    )
    assert "releases/huggingface/corpus/RW-Cite-datasets-*/" in script
    assert "releases/huggingface/domains/RW-Cite-domain-*/" in script
    assert "--exclude='datasets/'" in script
    assert "--exclude='models/'" in script
    assert "--exclude='outputs/'" in script


def test_domain_pack_uses_exact_paper_domain_inventory(repo_root: Path) -> None:
    script = (repo_root / "scripts" / "pack_domain_release.sh").read_text(
        encoding="utf-8"
    )
    assert "PAPER_DOMAINS=(ewm gw driving cosmo radio wsi exo sqc fno sce)" in script

    prepacked = (repo_root / "scripts" / "stage_prepacked_domains.py").read_text(
        encoding="utf-8"
    )
    for domain in ("ewm", "gw", "driving", "cosmo", "radio", "wsi", "exo", "sqc", "fno", "sce"):
        assert f'"{domain}"' in prepacked
