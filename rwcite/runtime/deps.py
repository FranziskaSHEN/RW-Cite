from __future__ import annotations

import contextlib
import os
import sys
from functools import lru_cache

from rwcite.runtime.config import RWCITE_ROOT, load_app_settings


def ensure_package_imports() -> None:
    """No-op: retrieve lives under ``rwcite.retrieve`` after package install."""
    return


@contextlib.contextmanager
def data_cwd():
    """Temporarily chdir into data_root for modules that use relative config paths."""
    settings = load_app_settings()
    prev = os.getcwd()
    try:
        os.chdir(settings.data_root)
        yield settings.data_root
    finally:
        os.chdir(prev)


@lru_cache
def get_domain_cache():
    from rwcite.runtime.services.domain_cache import DomainResourceCache

    settings = load_app_settings()
    return DomainResourceCache(max_domains=settings.gexf_cache_max_domains)


@lru_cache
def get_retriever_service():
    from rwcite.runtime.services.domain_retriever_service import DomainRetrieverService

    # Release path uses domain embeds + BGE; SQLite MetadataStore is optional/product-only.
    return DomainRetrieverService(get_domain_cache(), None)


@lru_cache
def get_metadata_store():
    """Release 1.0 does not require arxiv_metadata.db; return None."""
    return None


@lru_cache
def get_metadata_service():
    from rwcite.runtime.services.metadata_service import MetadataService

    return MetadataService(get_domain_cache(), None)
