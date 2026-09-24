#!/usr/bin/env python3
"""RW-Cite domain graph pipeline (unified for all domains).

Stages:
  0) Link shared Topics / embeddings / BGE if missing
  1) Dense retrieve (multi-prototype + N_max ∧ τ) → retrieval_nodes JSON
  2) Download + LaTeX clean (skip_if_downloaded; incremental)
  3–4) Native seed + expand with checkpoint resume
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

ROOT = Path(
    os.environ.get("RWCITE_ROOT", Path(__file__).resolve().parents[2])
).resolve()
# Optional external asset mount (shared arxiv topics / BGE / etc.)
ASSET_ROOT = Path(
    os.environ.get("RWCITE_ASSET_SOURCE")
    or os.environ.get("RWCITE_DATA_ROOT")
    or ROOT
).resolve()

os.chdir(ROOT)

import logging  # noqa: E402

logging.getLogger("bibtexparser").setLevel(logging.ERROR)

from rwcite.retrieve import arxiv_download as lp  # noqa: E402
from rwcite.retrieve.utils.utils import read_yaml_file  # noqa: E402
from rwcite.graph.domain_graph_resume import (  # noqa: E402
    pick_zip_dirs,
    run_native_resume_build,
)
from rwcite.graph.domain_retrieve import (  # noqa: E402
    normalize_queries_arg,
    retrieve_and_write,
)
from rwcite.graph.domain_paths import domain_build_paths

DEFAULT_QUERY = (
    "scientific literature search exploratory information retrieval "
    "knowledge graph natural language processing"
)
DEFAULT_CONFIG = "configs/config_sqc.yaml"
ARXIV_META = ROOT / "datasets" / "arxiv-metadata-oai-snapshot.json"

SHARED_LINKS = (
    ("datasets/arxiv-metadata-oai-snapshot.json", True),
    ("datasets/arxiv_topics", False),
    ("datasets/topic_level_embeds/AliMaatouk___ar_xiv-topics-embeddings", False),
    ("datasets/arxiv_topics_supplement.jsonl", True),
    ("datasets/topic_level_embeds/supplement_embeddings.parquet", True),
    ("models/base/bge-large-en-v1.5", False),
)


def _load_enrich_module():
    from rwcite.graph import extract_rr_graph as mod

    return mod


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="RW-Cite domain graph: retrieve → download → native seed/expand (resume)"
    )
    p.add_argument("--query", default="", help="Query; | for multi; or config.queries")
    p.add_argument("--queries", action="append", default=[], help="Extra prototypes")
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--num-retrievals", type=int, default=None, help="N_max override")
    p.add_argument("--score-tau", type=float, default=None)
    p.add_argument("--no-score-tau", action="store_true")
    p.add_argument(
        "--recent-years",
        type=int,
        default=None,
        help="Override config retriever.recent_years (e.g. 10). 0=disable year gate",
    )
    p.add_argument(
        "--min-arxiv-year",
        type=int,
        default=None,
        help="Override absolute min arXiv year (takes precedence over --recent-years)",
    )
    p.add_argument("--skip-retrieval", action="store_true")
    p.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip retrieve+download; build from existing papers + retrieval_nodes",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Use only first N retrieval ids (smoke). 0=all",
    )
    p.add_argument(
        "--work-dir",
        default="",
        help="Checkpoint dir (default: <domain>/description/graph_build_ckpt)",
    )
    p.add_argument(
        "--out-gexf",
        default="",
        help="Final GEXF (default: config gexf under description/)",
    )
    p.add_argument("--reset", action="store_true", help="Clear checkpoint in work-dir")
    p.add_argument("--ckpt-every", type=int, default=50)
    p.add_argument(
        "--zip-dir",
        default="",
        help="Force a single zip root; default=auto multi-root by coverage",
    )
    p.add_argument(
        "--seed-only",
        action="store_true",
        help="Stop after native seed (no expand)",
    )
    p.add_argument(
        "--skip-enrich",
        action="store_true",
        help="Alias of --seed-only",
    )
    p.add_argument(
        "--allow-download",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Seed/expand may download missing source tars when no local zip",
    )
    p.add_argument("--expand-rounds", type=int, default=2)
    p.add_argument("--expand-frac", type=float, default=0.2)
    p.add_argument("--max-expand-per-round", type=int, default=0)
    p.add_argument("--min-expand-per-round", type=int, default=0)
    p.add_argument(
        "--expand-sim-tau",
        type=float,
        default=None,
        help="Expand similarity gate; default 0=off (indeg-only). Set e.g. 0.65 to enable",
    )
    return p.parse_args()


def _link_or_copy(src: Path, dst: Path) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return "exists"
    try:
        os.link(src, dst)
        return "hardlink"
    except OSError:
        pass
    if src.is_dir():
        os.symlink(src, dst, target_is_directory=True)
        return "symlink"
    os.symlink(src, dst)
    return "symlink"


def ensure_shared_assets() -> None:
    print(f"[0/4] Ensuring shared corpus assets under {ROOT}")
    if ASSET_ROOT.resolve() == ROOT.resolve():
        print("  asset source == RWCITE_ROOT; skip external linking")
        return
    if not ASSET_ROOT.is_dir():
        print(f"  WARN: asset source missing: {ASSET_ROOT}")
        return
    for rel, is_file in SHARED_LINKS:
        src = ASSET_ROOT / rel
        dst = ROOT / rel
        if not src.exists():
            print(f"  SKIP missing in asset source: {rel}")
            continue
        if is_file and src.is_dir():
            print(f"  SKIP type mismatch: {rel}")
            continue
        mode = _link_or_copy(src, dst)
        print(f"  {mode}: {rel}")


def init_pipeline_globals(
    config_path: str,
    num_retrievals: int | None = None,
) -> dict:
    cfg_path = Path(config_path)
    if not cfg_path.is_absolute():
        cfg_path = ROOT / cfg_path
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    config = read_yaml_file(str(cfg_path))
    if num_retrievals is not None:
        config["retriever"]["num_retrievals"] = num_retrievals
    working_dir = config["data_downloading"]["download_directory"]
    domain_tag = Path(working_dir.rstrip("/")).name.replace("_retrieval", "")
    paths = domain_build_paths(working_dir)

    lp.config = config
    lp.config_path = str(cfg_path)
    lp.working_dir = working_dir
    lp.save_zip_directory = working_dir + "research_papers_zip/"
    lp.save_directory = working_dir + "research_papers/"
    lp.save_description = working_dir + "description/"
    lp.save_path = lp.save_description + "results.json"
    lp.save_graph = lp.save_description + "test_graph.json"
    lp.gexf_file = lp.save_description + config["data_downloading"]["gexf_file"]
    lp.retrieval_nodes_path = config.get(
        "retrieval_nodes_path",
        paths["retrieval_nodes"],
    )
    lp.failed_downloads_path = config.get(
        "failed_downloads_path",
        paths["failed_downloads"],
    )

    for path in (lp.save_zip_directory, lp.save_directory, lp.save_description):
        os.makedirs(path, exist_ok=True)
    Path(lp.retrieval_nodes_path).parent.mkdir(parents=True, exist_ok=True)

    random.seed(config["data_downloading"]["processing"]["random_seed"])
    return {"domain_tag": domain_tag, "config": config, "config_path": str(cfg_path)}


def count_local_downloads() -> tuple[int, int, int]:
    graph_count = 0
    if Path(lp.save_graph).exists():
        with open(lp.save_graph, encoding="utf-8") as f:
            graph_count = len(json.load(f))
    tar_count = len(list(Path(lp.save_zip_directory).glob("*.tar.gz")))
    dir_count = 0
    papers = Path(lp.save_directory)
    if papers.is_dir():
        dir_count = sum(
            1 for p in papers.iterdir() if p.is_dir() and any(p.iterdir())
        )
    return graph_count, tar_count, dir_count


def run_retrieval(
    queries: list[str],
    *,
    score_tau: float | None,
    no_score_tau: bool,
    recent_years: int | None = None,
    min_arxiv_year: int | None = None,
) -> list[str]:
    num = lp.config["retriever"]["num_retrievals"]
    ry = recent_years
    if ry is None:
        ry = (lp.config.get("retriever") or {}).get("recent_years")
    print(
        f"[1/4] Retrieve N_max={num} prototypes={len(queries)} "
        f"tau={'off' if no_score_tau else (score_tau if score_tau is not None else 'auto')} "
        f"recent_years={ry if min_arxiv_year is None else f'min_year={min_arxiv_year}'}"
    )
    for i, q in enumerate(queries):
        print(f"  q[{i}]: {q!r}")
    _ids, meta = retrieve_and_write(
        queries,
        lp.retrieval_nodes_path,
        lp.config_path,
        n_max=num,
        tau=score_tau,
        apply_tau=not no_score_tau,
        recent_years=0 if recent_years == 0 else recent_years,
        min_arxiv_year=min_arxiv_year,
    )
    print(f"  year_window: {meta.get('year_window')}")
    print(f"  gate: {meta.get('gate')}")

    removed = lp.filter_retrieval_nodes_file()
    if removed:
        print(f"  filtered {removed} previously failed paper ids")
    with open(lp.retrieval_nodes_path, encoding="utf-8") as f:
        paper_ids = list(json.load(f).keys())
    print(f"  retrieved {len(paper_ids)} → {lp.retrieval_nodes_path}")
    return paper_ids


def run_download(paper_ids: list[str]) -> None:
    graph_count, tar_count, dir_count = count_local_downloads()
    print(
        f"[2/4] Download/clean {len(paper_ids)} papers "
        f"(skip existing: graph={graph_count}, tar={tar_count}, dirs={dir_count})..."
    )
    if not ARXIV_META.exists():
        raise FileNotFoundError(
            f"Missing {ARXIV_META}. Sync/link arxiv metadata snapshot first."
        )
    result = lp.fetch_arxiv_papers(paper_ids)
    print(result)


def main() -> int:
    args = parse_args()
    print(f"RW-Cite root: {ROOT}")
    print(f"asset source: {ASSET_ROOT}")
    print(f"retrieve: rwcite.retrieve | root={ROOT}")

    ensure_shared_assets()
    meta = init_pipeline_globals(
        config_path=args.config,
        num_retrievals=args.num_retrievals,
    )
    domain_tag = meta["domain_tag"]
    config = meta["config"]

    queries = normalize_queries_arg(
        args.query or None,
        args.queries or None,
        config,
    )
    if not queries:
        queries = [DEFAULT_QUERY]

    expand_sim_tau = (
        float(args.expand_sim_tau) if args.expand_sim_tau is not None else 0.0
    )
    seed_only = bool(args.seed_only or args.skip_enrich)

    work_dir = (
        Path(args.work_dir)
        if args.work_dir
        else Path(lp.save_description) / "graph_build_ckpt"
    )
    if not work_dir.is_absolute():
        work_dir = ROOT / work_dir
    out_gexf = Path(args.out_gexf) if args.out_gexf else Path(lp.gexf_file)
    if not out_gexf.is_absolute():
        out_gexf = ROOT / out_gexf

    # --- retrieve / download ---
    paper_ids: list[str] = []
    if args.skip_download:
        print("[1-2/4] Skipping retrieval/download (--skip-download)")
        nodes_path = Path(lp.retrieval_nodes_path)
        if not nodes_path.exists():
            print(f"Missing {nodes_path}; need retrieval list for native seed build.")
            return 1
        with open(nodes_path, encoding="utf-8") as f:
            paper_ids = list(json.load(f).keys())
    else:
        if args.skip_retrieval:
            nodes_path = Path(lp.retrieval_nodes_path)
            if not nodes_path.exists():
                print(f"Missing {nodes_path}; cannot --skip-retrieval.")
                return 1
            with open(nodes_path, encoding="utf-8") as f:
                paper_ids = list(json.load(f).keys())
            print(f"[1/4] Reusing {len(paper_ids)} ids from {nodes_path}")
        else:
            paper_ids = run_retrieval(
                queries,
                score_tau=args.score_tau,
                no_score_tau=args.no_score_tau,
                recent_years=args.recent_years,
                min_arxiv_year=args.min_arxiv_year,
            )
        if not paper_ids:
            print("No papers retrieved; aborting.")
            return 1
        run_download(paper_ids)

    if args.limit and args.limit > 0:
        paper_ids = paper_ids[: args.limit]
        print(f"[limit] using first {len(paper_ids)} retrieval ids", flush=True)

    # --- build (native seed + optional expand) ---
    print("[3-4/4] Native seed + expand (resumable)...", flush=True)
    mod = _load_enrich_module()
    papers_dir = Path(lp.save_directory)
    if not papers_dir.is_absolute():
        papers_dir = ROOT / papers_dir
    papers_parent = papers_dir.parent
    explicit_zip = Path(args.zip_dir) if args.zip_dir else None
    if explicit_zip is not None and not explicit_zip.is_absolute():
        explicit_zip = ROOT / explicit_zip
    zip_dirs = pick_zip_dirs(
        papers_parent=papers_parent,
        asset_root=ASSET_ROOT if ASSET_ROOT.is_dir() else None,
        explicit=explicit_zip,
    )
    embedder = (config.get("retriever") or {}).get(
        "embedder", "models/base/bge-large-en-v1.5"
    )
    run_native_resume_build(
        mod,
        retrieval_ids=paper_ids,
        papers_dir=papers_dir,
        zip_dirs=zip_dirs,
        metadata_path=ARXIV_META,
        work_dir=work_dir,
        out_gexf=out_gexf,
        queries=queries,
        expand_rounds=0 if seed_only else args.expand_rounds,
        expand_frac=args.expand_frac,
        max_expand_per_round=args.max_expand_per_round,
        min_expand_per_round=args.min_expand_per_round,
        expand_sim_tau=expand_sim_tau,
        expand_embedder=embedder,
        allow_download=args.allow_download,
        ckpt_every=args.ckpt_every,
        reset=args.reset,
        seed_only=seed_only,
        run_tag=domain_tag,
    )
    print(
        f"Pipeline complete (native resume). out={out_gexf} ckpt={work_dir}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
