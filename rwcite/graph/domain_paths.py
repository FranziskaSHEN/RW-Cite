"""Paths for per-domain artifacts (under ``*_retrieval/``, not ``datasets/``)."""

from __future__ import annotations

from pathlib import Path

import yaml

# Fixed train/test protocol (test_admit_v1). Recorded in split_meta only.
DEFAULT_SPLIT_ID = "test_admit_v1"


def normalize_working_dir(download_directory: str) -> str:
    return download_directory.rstrip("/") + "/"


def working_dir_for_domain(domain_id: str, root: Path | str) -> str:
    """Return ``download_directory`` for a domain id from domains.yaml + config."""
    root = Path(root)
    dy = yaml.safe_load((root / "configs/domains.yaml").read_text(encoding="utf-8")) or {}
    block = (dy.get("domains") or {}).get(domain_id)
    if not block:
        raise KeyError(f"unknown domain: {domain_id}")
    cfg_path = root / block["domain_config"]
    ycfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    return normalize_working_dir(ycfg["data_downloading"]["download_directory"])


def domain_build_paths(download_directory: str) -> dict[str, str]:
    """Graph-build outputs."""
    root = normalize_working_dir(download_directory)
    desc = root + "description/"
    return {
        "root": root,
        "description": desc,
        "retrieval_nodes": root + "retrieval_nodes.json",
        "failed_downloads": root + "failed_downloads.json",
        "work_dir": desc + "rebuild_year10/",
        "graph_build_ckpt": desc + "graph_build_ckpt/",
    }


def domain_data_paths(download_directory: str) -> dict[str, str]:
    """Splits, jsonl, domain embeddings, retrieval diag (single default split)."""
    root = normalize_working_dir(download_directory)
    data = root + "data/"
    return {
        "data_root": data,
        "splits_dir": data + "splits/",
        "jsonl_dir": data + "reference_recommend/",
        "domain_embeddings": data + "domain_embeddings.parquet",
        "retrieval_diag": data + "retrieval_diag.json",
    }


def domain_ranker_paths(
    download_directory: str,
    *,
    base_tag: str = "",
) -> dict[str, str]:
    """Ranker model checkpoints."""
    root = normalize_working_dir(download_directory)
    ranker = root + "ranker/"
    ce_root = ranker + "ce_sent/"
    c2s_hn = ce_root + "c2s-hn/"
    if base_tag:
        c2s_hn = ce_root + f"c2s-hn-{base_tag}/"
    return {
        "ranker_root": ranker,
        "citelink_dir": ranker + "citelink/",
        "citelink_model": ranker + "citelink/model.npz",
        "ce_stage1": ce_root + "stage1/",
        "ce_c2s_hn": c2s_hn,
        "ce_ablation": ce_root.rstrip("/"),
    }


def domain_ranker_pool_paths(download_directory: str) -> dict[str, str]:
    """Training pairs / citelink cache (intermediate ranker artifacts)."""
    root = normalize_working_dir(download_directory)
    pool = root + "ranker/pool/"
    return {
        "pool_dir": pool,
        "citelink_cache": pool + "citelink_pairs.npz",
        "base_pairs": pool + "ce_base_pairs.jsonl",
        "hn_pairs": pool + "ce_hn_pairs.jsonl",
        "ablation_pairs": pool + "ce_pairs.jsonl",
    }


def domain_struct_ranker_paths(download_directory: str) -> dict[str, str]:
    """Domain-local structural shortlist ranker (10-feature linear RankNet)."""
    root = normalize_working_dir(download_directory)
    struct = root + "ranker/struct/"
    return {
        "struct_dir": struct,
        "struct_model": struct + "model.npz",
        "struct_feats": struct + "feats/",
        "struct_report": struct + "train_report.json",
    }


def domain_ranker_eval_paths(
    download_directory: str,
    *,
    base_tag: str = "",
) -> dict[str, str]:
    """Eval merged json, gates, training reports."""
    root = normalize_working_dir(download_directory)
    ev = root + "ranker/eval/"
    report = ev + "c2s_report.json"
    if base_tag:
        report = ev + f"c2s_report_{base_tag}.json"
    return {
        "eval_dir": ev,
        "citelink_gate": ev + "citelink_gate.json",
        "c2s_report": report,
        "retrain_report": ev + "retrain_report.json",
    }


def domain_layout(
    download_directory: str,
    *,
    base_tag: str = "",
) -> dict[str, str]:
    """All domain-local artifact paths."""
    out: dict[str, str] = {}
    out.update(domain_build_paths(download_directory))
    out.update(domain_data_paths(download_directory))
    out.update(domain_ranker_paths(download_directory, base_tag=base_tag))
    out.update(domain_ranker_pool_paths(download_directory))
    out.update(domain_struct_ranker_paths(download_directory))
    out.update(domain_ranker_eval_paths(download_directory, base_tag=base_tag))
    return out


def domain_root_from_config(config: dict) -> str:
    return normalize_working_dir(config["data_downloading"]["download_directory"])
