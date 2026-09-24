from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

import networkx as nx

from rwcite.runtime.config import load_domains_settings, data_path, resolve_domain_path


def _gexf_override(domain_id: str) -> Path | None:
    """Optional GEXF path: RR_GEXF_OVERRIDE or RR_GEXF_OVERRIDE_<DOMAIN>."""
    specific = (os.environ.get(f"RR_GEXF_OVERRIDE_{domain_id.upper()}") or "").strip()
    generic = (os.environ.get("RR_GEXF_OVERRIDE") or "").strip()
    raw = specific or generic
    if not raw:
        return None
    p = Path(raw)
    if not p.is_absolute():
        from rwcite.runtime.config import RWCITE_ROOT

        p = Path(RWCITE_ROOT) / p
    return p if p.is_file() else None


class DomainResource:
    def __init__(self, domain_id: str):
        self.domain_id = domain_id
        cfg = load_domains_settings().domains[domain_id]
        self.label = cfg.label
        override = _gexf_override(domain_id)
        self.gexf_path = override or resolve_domain_path(domain_id, "gexf")
        if override is not None:
            print(f"[domain_cache] {domain_id} GEXF override → {self.gexf_path}", flush=True)
        self.papers_dir = resolve_domain_path(domain_id, "papers_dir")
        self.domain_config_path = data_path(cfg.domain_config)
        self.adapter_path = resolve_domain_path(domain_id, "adapter")
        self.retrieval_nodes_path = resolve_domain_path(domain_id, "retrieval_nodes")
        self.graph: nx.DiGraph | None = None
        self.node_ids: list[str] = []
        self.retrieval_seeds: set[str] = set()

    def load(self) -> None:
        if self.graph is not None:
            return
        if not self.gexf_path.exists():
            # Domain registered but pipeline not finished yet.
            self.graph = nx.DiGraph()
            self.node_ids = []
            self.retrieval_seeds = set()
            if self.retrieval_nodes_path.exists():
                with open(self.retrieval_nodes_path, encoding="utf-8") as f:
                    self.retrieval_seeds = set(json.load(f).keys())
            return
        self.graph = nx.read_gexf(
            self.gexf_path, node_type=None, relabel=False, version="1.2draft"
        )
        self.node_ids = list(self.graph.nodes())
        if self.retrieval_nodes_path.exists():
            with open(self.retrieval_nodes_path, encoding="utf-8") as f:
                seeds = json.load(f)
            self.retrieval_seeds = set(seeds.keys())
        else:
            self.retrieval_seeds = set()


class DomainResourceCache:
    def __init__(self, max_domains: int = 2):
        self.max_domains = max_domains
        self._cache: dict[str, DomainResource] = {}
        self._order: list[str] = []

    def get(self, domain_id: str) -> DomainResource:
        if domain_id not in load_domains_settings().domains:
            raise KeyError(f"Unknown domain: {domain_id}")
        if domain_id not in self._cache:
            if len(self._cache) >= self.max_domains and self._order:
                evict = self._order.pop(0)
                self._cache.pop(evict, None)
            resource = DomainResource(domain_id)
            resource.load()
            self._cache[domain_id] = resource
            self._order.append(domain_id)
        else:
            self._order.remove(domain_id)
            self._order.append(domain_id)
        return self._cache[domain_id]

    def invalidate(self, domain_id: str | None = None) -> None:
        if domain_id is None:
            self._cache.clear()
            self._order.clear()
            return
        self._cache.pop(domain_id, None)
        if domain_id in self._order:
            self._order.remove(domain_id)

    def stats(self, domain_id: str) -> dict:
        res = self.get(domain_id)
        g = res.graph
        assert g is not None
        latex_count = sum(
            1
            for pid in res.node_ids
            if (res.papers_dir / pid / "final_cleaned.tex").exists()
        )
        return {
            "node_count": g.number_of_nodes(),
            "edge_count": g.number_of_edges(),
            "latex_count": latex_count,
            "retrieval_seed_count": len(res.retrieval_seeds),
            "ready": res.gexf_path.exists(),
        }
