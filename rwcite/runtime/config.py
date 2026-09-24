from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel


RWCITE_ROOT = Path(
    os.environ.get("RWCITE_ROOT", Path(__file__).resolve().parents[2])
).resolve()

class AppSettings(BaseModel):
    data_root: Path
    metadata: Path = Path("datasets/arxiv-metadata-oai-snapshot.json")
    default_domain: str = "sqc"
    cuda_visible_devices: str = "0"
    max_concurrent_gpu_jobs: int = 1
    gexf_cache_max_domains: int = 2


class DomainConfig(BaseModel):
    label: str
    domain_config: str
    gexf: str
    adapter: str = ""
    retrieval_nodes: str
    papers_dir: str
    enabled: bool = True
    adapter_rr: str | None = None

class SharedConfig(BaseModel):
    embedder: str
    topics_cache: str
    embeddings_cache: str
    metadata: str = "datasets/arxiv-metadata-oai-snapshot.json"
    domain_embeddings_cache: str = "datasets/domain_embeddings"


class DomainsSettings(BaseModel):
    domains: dict[str, DomainConfig]
    shared: SharedConfig


@lru_cache
def load_app_settings() -> AppSettings:
    path = RWCITE_ROOT / "configs" / "app.yaml"
    if path.exists():
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
    else:
        raw = {}

    env_root = os.environ.get("RWCITE_DATA_ROOT") or os.environ.get("RWCITE_ROOT")
    if env_root:
        raw["data_root"] = env_root
    elif "data_root" not in raw:
        raw["data_root"] = str(RWCITE_ROOT)
    raw.pop("metadata_db", None)
    raw.pop("python_port", None)
    raw.pop("nextjs_port", None)
    raw.pop("cors_origins", None)
    raw.pop("corpus_update", None)
    raw.pop("retriever", None)
    raw["data_root"] = Path(raw["data_root"]).resolve()

    domains_raw = yaml.safe_load(
        (RWCITE_ROOT / "configs" / "domains.yaml").read_text(encoding="utf-8")
    )
    md = raw.get("metadata") or (domains_raw.get("shared") or {}).get(
        "metadata", "datasets/arxiv-metadata-oai-snapshot.json"
    )
    md_path = Path(md)
    if md_path.is_absolute():
        raw["metadata"] = md_path
    else:
        raw["metadata"] = raw["data_root"] / md_path
    return AppSettings(**raw)


@lru_cache
def load_domains_settings() -> DomainsSettings:
    path = RWCITE_ROOT / "configs" / "domains.yaml"
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    shared = raw.get("shared") or {}
    if isinstance(shared, dict):
        shared.pop("metadata_db", None)
        shared.pop("base_model", None)
        shared.pop("instruct_model", None)
        shared.pop("alpaca_template", None)
    return DomainsSettings(**raw)


def enabled_domains() -> dict[str, DomainConfig]:
    return {
        did: cfg
        for did, cfg in load_domains_settings().domains.items()
        if cfg.enabled
    }


def data_path(*parts: str) -> Path:
    root = load_app_settings().data_root
    return root.joinpath(*parts)


def resolve_domain_path(domain_id: str, field: str) -> Path:
    domain = load_domains_settings().domains[domain_id]
    rel = getattr(domain, field)
    return data_path(rel)
