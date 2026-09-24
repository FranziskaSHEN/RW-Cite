#!/usr/bin/env python3
"""Domain retrieval diagnostics.

Modes:
  * Default: score existing retrieval lists vs domain embeddings × build queries.
  * ``--retrieve``: run WP3 retrieve (year → multi-prototype → N_max∧τ) first.
  * ``--retrieve-only``: WP3 retrieve + gate/year meta + zip coverage only.

Output default: per-domain ``<domain>_retrieval/data/retrieval_diag.json`` when a single
domain is selected; otherwise ``datasets/rr_pool_ranker/domain_retrieval_diag.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]


def _domain_keys_from_registry() -> dict[str, dict[str, Any]]:
    """Build diagnostic key map from configs/domains.yaml (no hard-coded domains)."""
    dy = yaml.safe_load((ROOT / "configs/domains.yaml").read_text(encoding="utf-8")) or {}
    out: dict[str, dict[str, Any]] = {}
    for did, block in (dy.get("domains") or {}).items():
        if not isinstance(block, dict) or not block.get("enabled", True):
            continue
        cfg = block.get("domain_config") or ""
        out[did] = {
            "config": cfg,
            "kw": (),  # optional keyword head/tail; empty = skip kw diagnostics
            "gexf": block.get("gexf", ""),
            "retrieval_nodes": block.get("retrieval_nodes", ""),
        }
    return out


DOMAIN_KEYS = _domain_keys_from_registry()


def _load_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_domains_yaml(path: Path) -> dict:
    raw = _load_yaml(path)
    return raw.get("domains") or {}


def _resolve_domain_paths(
    key: str,
    *,
    domains_cfg: dict,
) -> tuple[Path, Path, Path, dict, dict]:
    """Return (cfg_path, ret_path, zip_dir, ycfg, dcfg)."""
    meta = DOMAIN_KEYS[key]
    dcfg = domains_cfg.get(key) or {}
    cfg_path = ROOT / meta["config"]
    ycfg = _load_yaml(cfg_path) if cfg_path.is_file() else {}
    ret_rel = dcfg.get("retrieval_nodes") or ycfg.get("retrieval_nodes_path") or ""
    ret_path = ROOT / ret_rel
    dd = (ycfg.get("data_downloading") or {}).get("download_directory") or ""
    if dd:
        zip_dir = ROOT / dd.rstrip("/") / "research_papers_zip"
    else:
        papers = dcfg.get("papers_dir") or ""
        zip_dir = (ROOT / papers).parent / "research_papers_zip"
    return cfg_path, ret_path, zip_dir, ycfg, dcfg


def zip_has(zip_dir: Path, pid: str) -> bool:
    for cand in (pid, pid.replace("/", "_")):
        if (zip_dir / f"{cand}.tar.gz").exists():
            return True
        if (zip_dir / f"{cand}.zip").exists():
            return True
    return False


def zip_coverage(ret_ids: list[str], zip_dir: Path) -> dict[str, Any]:
    ret_set = set(ret_ids)
    cache = list(zip_dir.glob("*.tar.gz")) + list(zip_dir.glob("*.zip"))
    hit = sum(1 for pid in ret_set if zip_has(zip_dir, pid))
    try:
        zip_rel = str(zip_dir.relative_to(ROOT))
    except ValueError:
        zip_rel = str(zip_dir)
    return {
        "zip_dir": zip_rel,
        "zip_cache_total": len(cache),
        "retrieval_in_zip": hit,
        "retrieval_in_zip_frac": round(hit / max(1, len(ret_set)), 4),
    }


def run_wp3_retrieve(key: str, *, domains_cfg: dict) -> dict[str, Any]:
    """Run native WP3 retrieve for one domain; write retrieval_nodes JSON."""
    from rwcite.graph.domain_retrieve import retrieve_and_write

    cfg_path, ret_path, zip_dir, _ycfg, _dcfg = _resolve_domain_paths(
        key, domains_cfg=domains_cfg
    )
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Missing domain config: {cfg_path}")
    ret_path.parent.mkdir(parents=True, exist_ok=True)
    ids, meta = retrieve_and_write([], ret_path, cfg_path)
    out = {
        "domain": key,
        "config": str(cfg_path.relative_to(ROOT)),
        "retrieval_nodes": str(ret_path.relative_to(ROOT)),
        "n_retrieved": len(ids),
        "gate": meta.get("gate"),
        "year_window": meta.get("year_window"),
        "gate_order": meta.get("gate_order"),
        "n_prototypes": meta.get("n_prototypes"),
        "queries": meta.get("queries"),
        "zip": zip_coverage(ids, zip_dir),
    }
    print(
        f"  retrieve {key}: n={out['n_retrieved']} "
        f"tau={out['gate'].get('tau')} "
        f"zip={out['zip']['retrieval_in_zip']}/{out['n_retrieved']}",
        flush=True,
    )
    return out


def _load_retrieval_ids(ret_path: Path) -> list[str]:
    with open(ret_path, encoding="utf-8") as f:
        ret_raw = json.load(f)
    if isinstance(ret_raw, dict):
        return [str(k).strip() for k in ret_raw.keys()]
    return [str(x).strip() for x in ret_raw]


def _encode_query(embedder: str, text: str) -> np.ndarray:
    load_kwargs = {"local_files_only": True}
    try:
        tok = AutoTokenizer.from_pretrained(embedder, **load_kwargs)
        model = AutoModel.from_pretrained(embedder, **load_kwargs)
    except OSError:
        tok = AutoTokenizer.from_pretrained(embedder)
        model = AutoModel.from_pretrained(embedder)
    model = model.to(device="cuda", dtype=torch.float16)
    inputs = tok([text], return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        out = model(**inputs.to("cuda"))
        emb = out.last_hidden_state[:, 0, :].float().cpu().numpy()
    return emb


def _pct(xs: list[float]) -> dict[str, float]:
    if not xs:
        return {}
    a = np.asarray(xs, dtype=float)
    return {
        f"p{p}": float(np.percentile(a, p))
        for p in (0, 5, 10, 25, 50, 75, 90, 95, 100)
    }


def _bucket_means(scores: list[float]) -> dict[str, float | None]:
    def mean_slice(a: int, b: int | None = None) -> float | None:
        chunk = scores[a:b]
        return float(np.mean(chunk)) if chunk else None

    n = len(scores)
    return {
        "head_100": mean_slice(0, min(100, n)),
        "r100_500": mean_slice(100, min(500, n)) if n > 100 else None,
        "r500_2k": mean_slice(500, min(2000, n)) if n > 500 else None,
        "r2k_5k": mean_slice(2000, min(5000, n)) if n > 2000 else None,
        "r5k_end": mean_slice(5000, n) if n > 5000 else None,
        "tail_200": mean_slice(max(0, n - 200), n) if n else None,
    }


def _kw_rate(titles: list[str], kws: tuple[str, ...]) -> float:
    if not titles:
        return 0.0
    hit = 0
    for t in titles:
        tl = (t or "").lower()
        if any(k in tl for k in kws):
            hit += 1
    return hit / len(titles)


def _norm_id(pid: str) -> str:
    return str(pid).strip()


def diagnose_domain(
    key: str,
    *,
    domains_cfg: dict,
    emb_dir: Path,
    embedder: str,
    query_override: str | None = None,
) -> tuple[str, dict[str, Any]]:
    meta = DOMAIN_KEYS[key]
    cfg_path, ret_path, zip_dir, ycfg, dcfg = _resolve_domain_paths(
        key, domains_cfg=domains_cfg
    )
    ret_cfg = ycfg.get("retriever") or {}
    queries = ret_cfg.get("queries") or []
    if query_override:
        query = query_override
    elif queries:
        query = queries[0]
    else:
        query = str(ret_cfg.get("query") or "")
    if not query:
        raise ValueError(f"No query for domain {key}")

    if not ret_path.is_file():
        raise FileNotFoundError(
            f"Missing {ret_path}; run with --retrieve or --retrieve-only first."
        )

    ret_ids = _load_retrieval_ids(ret_path)
    ret_set = set(ret_ids)

    gexf_path = ROOT / (dcfg.get("gexf") or "")

    from rwcite.graph.domain_paths import domain_data_paths, domain_root_from_config

    emb_path = ROOT / domain_data_paths(domain_root_from_config(ycfg))[
        "domain_embeddings"
    ]
    if not emb_path.is_file():
        legacy = emb_dir / f"{key}_embeddings.parquet"
        if legacy.is_file():
            emb_path = legacy

    zip_stats = zip_coverage(ret_ids, zip_dir)

    score_result: dict[str, Any] = {}
    if emb_path.is_file():
        df = pd.read_parquet(emb_path)
        id_col = "paper_id" if "paper_id" in df.columns else "id"
        ids = [_norm_id(x) for x in df[id_col].tolist()]
        embs = np.stack(df["embedding"].tolist()).astype(np.float32)
        id_to_i = {pid: i for i, pid in enumerate(ids)}

        q_emb = _encode_query(embedder, query)
        all_scores = cosine_similarity(q_emb, embs)[0]

        scored_ret: list[tuple[str, float]] = []
        miss = 0
        for pid in ret_ids:
            i = id_to_i.get(pid)
            if i is None:
                miss += 1
                continue
            scored_ret.append((pid, float(all_scores[i])))
        scored_ret.sort(key=lambda x: -x[1])
        ret_scores = [s for _, s in scored_ret]

        title_map: dict[str, str] = {}
        g_nodes: set[str] = set()
        elig10: set[str] = set()
        if gexf_path.is_file():
            g = nx.read_gexf(str(gexf_path), node_type=None, relabel=False, version="1.2draft")
            g_nodes = {_norm_id(n) for n in g.nodes}
            for n in g.nodes:
                nid = _norm_id(n)
                title_map[nid] = str(g.nodes[n].get("title") or "")
                if g.out_degree(n) >= 10:
                    elig10.add(nid)

        ret_in_graph = len(ret_set & g_nodes) / len(ret_set) if ret_set else 0.0
        graph_non_ret = len(g_nodes - ret_set) / len(g_nodes) if g_nodes else 0.0
        elig10_in_ret = len(elig10 & ret_set) / len(elig10) if elig10 else 0.0

        non_ret_idx = [i for i, pid in enumerate(ids) if pid not in ret_set]
        non_ret_scores = [float(all_scores[i]) for i in non_ret_idx]

        head_med = float(np.median(ret_scores[:200])) if ret_scores else None
        tau_margin = float(ret_cfg.get("score_tau_margin", 0.05))
        knee = None
        if head_med is not None and ret_scores:
            thr = head_med - tau_margin
            for rank, sc in enumerate(ret_scores):
                if sc < thr:
                    knee = rank
                    break

        def titles_for(slice_ids: list[str]) -> list[str]:
            return [title_map.get(pid, "") for pid in slice_ids]

        n = len(scored_ret)
        head_ids = [p for p, _ in scored_ret[:200]]
        mid_ids = [p for p, _ in scored_ret[n // 3 : 2 * n // 3]] if n else []
        tail_ids = [p for p, _ in scored_ret[max(0, n - 200) :]]
        non_sample = [ids[i] for i in non_ret_idx[:200]]

        kws = meta["kw"]
        score_result = {
            "ret_n": len(ret_ids),
            "score_cover": len(scored_ret),
            "score_miss": miss,
            "score_pct": _pct(ret_scores),
            "buckets": _bucket_means(ret_scores),
            "knee_rank_headmed_minus_margin": knee,
            "score_tau_margin": tau_margin,
            "head_med200": head_med,
            "non_ret_scored": len(non_ret_scores),
            "non_ret_score_pct": _pct(non_ret_scores),
            "kw_rate_head200": _kw_rate(titles_for(head_ids), kws),
            "kw_rate_mid": _kw_rate(titles_for(mid_ids), kws),
            "kw_rate_tail200": _kw_rate(titles_for(tail_ids), kws),
            "kw_rate_nonret_sample": _kw_rate(titles_for(non_sample), kws),
            "ret_in_graph": ret_in_graph,
            "graph_nodes": len(g_nodes),
            "graph_non_ret_frac": graph_non_ret,
            "elig10": len(elig10),
            "elig10_in_retrieval": elig10_in_ret,
            "gexf": str(gexf_path.relative_to(ROOT)) if gexf_path.is_file() else "",
        }
    else:
        score_result = {
            "ret_n": len(ret_ids),
            "score_cover": 0,
            "score_miss": len(ret_ids),
            "note": f"missing domain embeddings: {emb_path.relative_to(ROOT)}",
            "gexf": str(gexf_path.relative_to(ROOT)) if gexf_path.is_file() else "",
        }

    result = {
        **score_result,
        "retrieval_nodes": str(ret_path.relative_to(ROOT)),
        "n_max_config": ret_cfg.get("num_retrievals"),
        "apply_score_tau_config": ret_cfg.get("apply_score_tau"),
        "recent_years_config": ret_cfg.get("recent_years"),
        "demo": bool(ret_cfg.get("demo", False)),
        "n_prototypes_config": len(queries) if queries else 1,
        "zip": zip_stats,
    }
    return query, result


def main() -> int:
    ap = argparse.ArgumentParser(description="Domain retrieval diagnostics")
    ap.add_argument("--domains-yaml", default="configs/domains.yaml")
    ap.add_argument(
        "--emb-dir",
        default="datasets/domain_embeddings",
        help="Legacy fallback dir for domain embeddings if not under domain data/",
    )
    ap.add_argument(
        "--out",
        default="",
        help="Output JSON (default: domain data/retrieval_diag.json for one domain)",
    )
    ap.add_argument(
        "--domains",
        default="",
        help="Comma-separated domain keys (default: all enabled in domains.yaml)",
    )
    ap.add_argument("--embedder", default="models/base/bge-large-en-v1.5")
    ap.add_argument(
        "--retrieve",
        action="store_true",
        help="Run WP3 retrieve (year→multi-prototype→N_max∧τ) before diagnostics",
    )
    ap.add_argument(
        "--retrieve-only",
        action="store_true",
        help="Only WP3 retrieve + gate/year/zip stats (skip embedding score diag)",
    )
    args = ap.parse_args()

    domains_cfg = _load_domains_yaml(ROOT / args.domains_yaml)
    emb_dir = ROOT / args.emb_dir
    keys = [k.strip() for k in args.domains.split(",") if k.strip()]
    if not keys:
        keys = list(DOMAIN_KEYS.keys())

    out_path = Path(args.out) if args.out else None
    if out_path is None:
        if len(keys) == 1:
            from rwcite.graph.domain_paths import domain_data_paths, working_dir_for_domain

            wd = working_dir_for_domain(keys[0], ROOT)
            out_path = ROOT / domain_data_paths(wd)["retrieval_diag"]
        else:
            out_path = ROOT / "datasets/rr_pool_ranker/domain_retrieval_diag.json"

    out_retrieve: dict[str, Any] = {}
    if args.retrieve or args.retrieve_only:
        for key in keys:
            if key not in DOMAIN_KEYS:
                print(f"skip unknown domain {key}", flush=True)
                continue
            print(f"wp3 retrieve {key} ...", flush=True)
            out_retrieve[key] = run_wp3_retrieve(key, domains_cfg=domains_cfg)

    if args.retrieve_only:
        payload = {
            "retrieve": out_retrieve,
            "protocol": {
                "gate_order": ["year_window", "N_max_and_tau"],
                "note": "WP3 retrieve only; no embedding score diagnostics.",
            },
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"wrote {out_path}", flush=True)
        return 0

    out_queries: dict[str, str] = {}
    out_results: dict[str, Any] = {}
    for key in keys:
        if key not in DOMAIN_KEYS:
            print(f"skip unknown domain {key}", flush=True)
            continue
        print(f"diag {key} ...", flush=True)
        try:
            q, res = diagnose_domain(
                key,
                domains_cfg=domains_cfg,
                emb_dir=emb_dir,
                embedder=args.embedder,
            )
        except FileNotFoundError as exc:
            print(f"  skip {key}: {exc}", flush=True)
            continue
        out_queries[key] = q
        out_results[key] = res
        if "ret_in_graph" in res:
            print(
                f"  ret_n={res['ret_n']} ret_in_graph={res['ret_in_graph']:.3f} "
                f"elig10_in_ret={res['elig10_in_retrieval']:.3f} "
                f"zip={res['zip']['retrieval_in_zip']}/{res['ret_n']}",
                flush=True,
            )
        else:
            print(
                f"  ret_n={res['ret_n']} zip={res['zip']['retrieval_in_zip']}/{res['ret_n']}",
                flush=True,
            )

    payload: dict[str, Any] = {
        "queries": out_queries,
        "results": out_results,
        "protocol": {
            "score_space": "domain_embeddings × primary build query (BGE)",
            "note": "Not identical to full-corpus topic retrieve space; for head/tail & overlap.",
            "fields": [
                "ret_in_graph",
                "elig10",
                "elig10_in_retrieval",
                "graph_non_ret_frac",
                "zip",
            ],
        },
    }
    if out_retrieve:
        payload["retrieve"] = out_retrieve
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
