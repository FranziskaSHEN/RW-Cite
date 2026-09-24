from __future__ import annotations

import re

from rwcite.runtime.services.domain_cache import DomainResourceCache
from rwcite.runtime.services.metadata_store import MetadataStore

# Modern (YYMM.NNNNN) and legacy (e.g. hep-th/9901001, cs.AI/0001001) arXiv IDs
PAPER_ID_RE = re.compile(r"^(?:\d{4}\.\d{4,5}|[a-z\-]+(?:\.[A-Z]{2})?/\d{7})$")


class MetadataService:
    """Paper metadata from GEXF attrs; optional SQLite store (product-only, not Release)."""

    def __init__(self, cache: DomainResourceCache, store: MetadataStore | None = None):
        self.cache = cache
        self.store = store

    def get_paper(self, domain_id: str, paper_id: str) -> dict | None:
        if not PAPER_ID_RE.match(paper_id):
            return None
        res = self.cache.get(domain_id)
        if paper_id in res.graph.nodes:
            return self._merge_graph_and_sqlite(domain_id, paper_id, res)
        return self.get_paper_metadata(paper_id, domain_id=domain_id)

    def get_paper_metadata(self, paper_id: str, domain_id: str | None = None) -> dict | None:
        if not PAPER_ID_RE.match(paper_id):
            return None
        if not self.store:
            return None
        arxiv = self.store.get(paper_id)
        if not arxiv:
            return None
        in_degree = out_degree = None
        in_domain = False
        has_latex = False
        if domain_id:
            try:
                res = self.cache.get(domain_id)
                if paper_id in res.graph.nodes:
                    in_domain = True
                    in_degree = res.graph.in_degree(paper_id)
                    out_degree = res.graph.out_degree(paper_id)
                    latex_path = res.papers_dir / paper_id / "final_cleaned.tex"
                    has_latex = latex_path.exists()
            except KeyError:
                pass
        return {
            "paper_id": paper_id,
            "domain": domain_id or "",
            "title": arxiv.get("title") or "",
            "abstract": arxiv.get("abstract") or "",
            "label": paper_id,
            "introduction": "",
            "related": "",
            "in_degree": in_degree,
            "out_degree": out_degree,
            "is_retrieval_seed": False,
            "has_latex": has_latex,
            "in_domain": in_domain,
            "authors": arxiv.get("authors") or "",
            "categories": arxiv.get("categories") or "",
            "comments": arxiv.get("comments") or "",
            "journal_ref": arxiv.get("journal_ref") or "",
            "doi": arxiv.get("doi") or "",
            "update_date": arxiv.get("update_date") or "",
            "topics_l1": arxiv.get("topics_l1") or "",
            "topics_l2": arxiv.get("topics_l2") or "",
            "topics_l3": arxiv.get("topics_l3") or "",
            "metadata_source": "sqlite",
            "metadata_record_source": arxiv.get("source") or "",
        }

    def _merge_graph_and_sqlite(self, domain_id: str, paper_id: str, res) -> dict:
        attrs = dict(res.graph.nodes[paper_id])
        latex_path = res.papers_dir / paper_id / "final_cleaned.tex"
        intro = attrs.get("introduction") or attrs.get("intro") or ""
        related = attrs.get("related") or ""

        arxiv = (self.store.get(paper_id) if self.store else None) or {}
        title = attrs.get("title") or arxiv.get("title") or ""
        abstract = attrs.get("abstract") or arxiv.get("abstract") or ""
        meta_source = "sqlite+gexf" if arxiv else "gexf"

        return {
            "paper_id": paper_id,
            "domain": domain_id,
            "title": title,
            "abstract": abstract,
            "label": attrs.get("label", paper_id),
            "introduction": intro,
            "related": related,
            "in_degree": res.graph.in_degree(paper_id),
            "out_degree": res.graph.out_degree(paper_id),
            "is_retrieval_seed": paper_id in res.retrieval_seeds,
            "has_latex": latex_path.exists(),
            "in_domain": True,
            "authors": arxiv.get("authors") or "",
            "categories": arxiv.get("categories") or "",
            "comments": arxiv.get("comments") or "",
            "journal_ref": arxiv.get("journal_ref") or "",
            "doi": arxiv.get("doi") or "",
            "update_date": arxiv.get("update_date") or "",
            "topics_l1": arxiv.get("topics_l1") or "",
            "topics_l2": arxiv.get("topics_l2") or "",
            "topics_l3": arxiv.get("topics_l3") or "",
            "metadata_source": meta_source,
            "metadata_record_source": arxiv.get("source") or "",
        }
