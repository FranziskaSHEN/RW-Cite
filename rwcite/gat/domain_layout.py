"""Minimal per-domain paths for GAT, cross-encoder, and query admission.

Prefer ``RR_GEXF_OVERRIDE`` when set. ewm protocol fingerprints stay in
``protocol.py`` for contract_check; other domains resolve under their lane.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from rwcite.gat import protocol as P
from rwcite.graph.domain_paths import domain_layout, working_dir_for_domain


@dataclass(frozen=True)
class GatDomainPaths:
    domain: str
    lane: str  # relative working dir without trailing slash
    working_dir: str
    nofuture_gexf: Path
    full_o2_gexf: Path
    gat_mvp_dir: str  # relative to repo root
    eval_dir: str
    struct_model: Path
    struct_nofuture_model: Path
    train_jsonl: Path
    test_jsonl: Path
    split_dir: Path
    retrieval_nodes: Path
    ce_stage1: Path
    ce_pairs_nofuture: Path
    jsonl_dir: Path


def _split_counts(split_dir: Path) -> tuple[int, int, int]:
    meta = split_dir / "split_meta.json"
    n_train, n_test, n_front = 0, 0, 0
    if meta.is_file():
        sm = json.loads(meta.read_text(encoding="utf-8"))
        n_train = int(sm.get("n_train_sources") or 0)
        n_test = int(sm.get("n_test_sources") or 0)
        n_front = int(sm.get("n_frontier_sources") or 0)
    return n_train, n_test, n_front


def resolve_gat_domain(domain: str, root: Path | None = None) -> GatDomainPaths:
    root = P.root_path() if root is None else Path(root)
    dom = (domain or "ewm").strip().lower()
    if dom not in ("ewm", "sqc", "gw", "fno", "wsi", "cosmo", "qopt", "sce", "hep", "exo", "radio", "radseg", "driving"):
        raise KeyError(
            f"unsupported GAT domain: {dom!r} "
            f"(use ewm|sqc|gw|fno|wsi|cosmo|qopt|sce|hep|exo|radio|radseg|driving)"
        )

    if dom == "ewm":
        lane = P.WD
    else:
        lane = working_dir_for_domain(dom, root).rstrip("/")

    layout = domain_layout(lane + "/")
    o2 = root / f"{lane}/description/test_graph_rr.o2.gexf"
    nf = root / f"{lane}/description/test_graph_rr.o2.nofuture.gexf"
    if not nf.is_file():
        legacy = root / f"{lane}/description/test_graph_rr.nofuture.gexf"
        if legacy.is_file():
            nf = legacy

    struct_live = root / layout["struct_model"]
    struct_nf = root / f"{lane}/ranker/struct_nofuture/model.npz"
    # Prefer nofuture struct when present (fair train graph).
    struct = struct_nf if struct_nf.is_file() else struct_live

    jsonl_dir = root / layout["jsonl_dir"]
    return GatDomainPaths(
        domain=dom,
        lane=lane,
        working_dir=lane + "/",
        nofuture_gexf=nf,
        full_o2_gexf=o2,
        gat_mvp_dir=f"{lane}/ranker/gat_mvp",
        eval_dir=f"{lane}/ranker/eval",
        struct_model=struct,
        struct_nofuture_model=struct_nf,
        train_jsonl=jsonl_dir / "train.jsonl",
        test_jsonl=jsonl_dir / "test.jsonl",
        split_dir=root / layout["splits_dir"],
        retrieval_nodes=root / f"{lane}/retrieval_nodes.json",
        ce_stage1=root / f"{lane}/ranker/ce_sent/stage1_nofuture",
        ce_pairs_nofuture=root / f"{lane}/ranker/pool/ce_base_pairs_nofuture.jsonl",
        jsonl_dir=jsonl_dir,
    )


def effective_nofuture_gexf(paths: GatDomainPaths, root: Path | None = None) -> Path:
    root = P.root_path() if root is None else Path(root)
    env = os.environ.get("RR_GEXF_OVERRIDE") or ""
    if env.strip():
        p = Path(env)
        return p if p.is_absolute() else root / p
    return paths.nofuture_gexf


def shell_exports(domain: str, root: Path | None = None) -> str:
    """Bash ``eval``-able KEY=value lines for cookbook scripts."""
    root = P.root_path() if root is None else Path(root)
    p = resolve_gat_domain(domain, root)
    n_train, n_test, n_front = _split_counts(p.split_dir)
    # Always export this domain's nofuture path (do not echo a stale RR_GEXF_OVERRIDE).
    gexf = p.nofuture_gexf
    pack_gpus = {
        "ewm": "0,1",
        "sqc": "2,3",
        "gw": "4,5",
        "fno": "0,1",
        "wsi": "2,3",
        "cosmo": "3,4",
        "qopt": "0,1",
        "sce": "3,4",
        "hep": "3,4",
        "exo": "5,6",
        "radio": "3,4",
        "radseg": "5,6",
        "driving": "3,4",
    }[p.domain]
    ce_gpu = {
        "ewm": "0",
        "sqc": "2",
        "gw": "4",
        "fno": "0",
        "wsi": "2",
        "cosmo": "3",
        "qopt": "0",
        "sce": "3",
        "hep": "3",
        "exo": "5",
        "radio": "3",
        "radseg": "5",
        "driving": "3",
    }[p.domain]
    # Prefer relative paths from repo root for logs / overrides.
    def rel(path: Path) -> str:
        try:
            return str(path.relative_to(root))
        except ValueError:
            return str(path)

    lines = [
        f"DOMAIN={p.domain}",
        f"LANE={p.lane}",
        f"GEXF_O2={rel(p.full_o2_gexf)}",
        f"GEXF_NOFUTURE={rel(p.nofuture_gexf)}",
        f"RR_GEXF_OVERRIDE={rel(gexf)}",
        f"SPLIT_DIR={rel(p.split_dir)}",
        f"JSONL_DIR={rel(p.jsonl_dir)}",
        f"RET_NODES={rel(p.retrieval_nodes)}",
        f"GAT_MVP_DIR={p.gat_mvp_dir}",
        f"EVAL_DIR={p.eval_dir}",
        f"STRUCT_MODEL={rel(p.struct_model)}",
        f"STRUCT_NOFUTURE_MODEL={rel(p.struct_nofuture_model)}",
        f"CE_STAGE1={rel(p.ce_stage1)}",
        f"CE_PAIRS={rel(p.ce_pairs_nofuture)}",
        f"PACK_GPUS={pack_gpus}",
        f"CE_GPU={ce_gpu}",
        f"N_TRAIN={n_train}",
        f"N_TEST={n_test}",
        f"N_FRONTIER={n_front}",
        f"WINDOWS_TEST_TAG=test{n_test or 'XXX'}",
        f"WINDOWS_TRAIN_TAG=train{n_train or 'XXX'}",
        f"L0_CANONICAL_TAG={p.domain}_gat_l0_default_test{n_test or 'XXX'}",
        f"GAT_EVAL_TAG={p.domain}_gat_mvp_test{n_test or 'XXX'}",
    ]
    return "\n".join(lines) + "\n"
