from __future__ import annotations

from pathlib import Path

from rwcite.runtime.config import (
    enabled_domains,
    load_app_settings,
    load_domains_settings,
    resolve_domain_path,
)
from rwcite.runtime.deps import get_metadata_store


def test_load_domains_sqc(repo_root: Path, monkeypatch):
    monkeypatch.setenv("RWCITE_ROOT", str(repo_root))
    load_app_settings.cache_clear()
    load_domains_settings.cache_clear()
    ds = load_domains_settings()
    assert "sqc" in ds.domains
    assert ds.domains["sqc"].domain_config.endswith("config_sqc.yaml")
    assert "arxiv_metadata.db" not in ds.shared.metadata
    assert ds.shared.metadata.endswith("arxiv-metadata-oai-snapshot.json")
    assert not hasattr(ds.shared, "metadata_db")


def test_app_settings_metadata_jsonl(repo_root: Path, monkeypatch):
    monkeypatch.setenv("RWCITE_ROOT", str(repo_root))
    load_app_settings.cache_clear()
    load_domains_settings.cache_clear()
    app = load_app_settings()
    assert app.metadata.name == "arxiv-metadata-oai-snapshot.json"
    assert get_metadata_store() is None


def test_resolve_domain_paths(repo_root: Path, monkeypatch):
    monkeypatch.setenv("RWCITE_ROOT", str(repo_root))
    load_app_settings.cache_clear()
    load_domains_settings.cache_clear()
    gexf = resolve_domain_path("sqc", "gexf")
    assert gexf.name.startswith("test_graph_rr")
    assert gexf.suffix == ".gexf"
    cfg = resolve_domain_path("sqc", "domain_config")
    assert cfg.name == "config_sqc.yaml"


def test_enabled_domains_match_paper_suite(repo_root: Path, monkeypatch):
    monkeypatch.setenv("RWCITE_ROOT", str(repo_root))
    load_app_settings.cache_clear()
    load_domains_settings.cache_clear()
    assert set(enabled_domains()) == {
        "ewm",
        "gw",
        "driving",
        "cosmo",
        "radio",
        "wsi",
        "exo",
        "sqc",
        "fno",
        "sce",
    }
