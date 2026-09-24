"""Dump fused Top-N pools for citation-sentence adapter training and evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import torch

from rwcite.adapter import POOL_SOURCE
from rwcite.adapter.paths import AdapterV6Paths, ensure_adapter_dirs, resolve_adapter_v6
from rwcite.gat.domain_layout import effective_nofuture_gexf
from rwcite.gat.evaluate import evaluate
from rwcite.gat.fuse_ce import (
    _fuse_scores,
    _gat_scores_from_eval,
    assert_ce_cache_matches,
    cache_ce_scores,
)
from rwcite.gat.l0_recipe import DEFAULT_ALPHA, DEFAULT_RRF_K, DEFAULT_RRF_W
from rwcite.ranker.reference_recommend import normalize_arxiv_id


def _load_id_map(gat_mvp: Path) -> list[str]:
    meta = json.loads((gat_mvp / "id_map.json").read_text(encoding="utf-8"))
    return list(meta["ids"])


def _split_n(paths: AdapterV6Paths) -> tuple[int, int]:
    meta = paths.gat.split_dir / "split_meta.json"
    n_train = n_test = 0
    if meta.is_file():
        sm = json.loads(meta.read_text(encoding="utf-8"))
        n_train = int(sm.get("n_train_sources") or 0)
        n_test = int(sm.get("n_test_sources") or 0)
    return n_train, n_test


def _windows_path(paths: AdapterV6Paths, *, split: str) -> Path:
    n_train, n_test = _split_n(paths)
    gat_mvp = paths.root / paths.gat.gat_mvp_dir
    if split == "train":
        p = gat_mvp / f"windows_train{n_train}.npz"
        if not p.is_file():
            raise FileNotFoundError(f"missing train windows: {p}")
        return p
    p = gat_mvp / f"windows_test{n_test}.npz"
    if not p.is_file():
        raise FileNotFoundError(f"missing test windows: {p}")
    return p


def _ce_cache_path(paths: AdapterV6Paths, *, split: str, n: int) -> Path:
    gat_mvp = paths.root / paths.gat.gat_mvp_dir
    return gat_mvp / f"ce_scores_{split}{n}_sent_s1.npz"


def _gat_scores_path(paths: AdapterV6Paths, *, split: str, n: int) -> Path:
    eval_dir = paths.root / paths.gat.eval_dir
    return eval_dir / f"{paths.domain}_gat_mvp_{split}{n}_e4_scores_merged.json"


def ensure_ce_cache(
    paths: AdapterV6Paths,
    *,
    split: str,
    windows_path: Path,
    force: bool = False,
    batch_size: int = 64,
) -> Path:
    n_train, n_test = _split_n(paths)
    n = n_train if split == "train" else n_test
    ce_npz = _ce_cache_path(paths, split=split, n=n)
    ce_path = paths.gat.ce_stage1
    gexf = effective_nofuture_gexf(paths.gat, paths.root)
    if ce_npz.is_file() and not force:
        try:
            assert_ce_cache_matches(ce_npz, ce_path=ce_path, force=False)
            return ce_npz
        except RuntimeError:
            pass
    print(f"caching CE scores → {ce_npz}", flush=True)
    cache_ce_scores(
        root=paths.root,
        windows_path=windows_path,
        out_npz=ce_npz,
        ce_path=ce_path,
        gexf=gexf,
        mode="ce_sent",
        short=True,
        max_sents=3,
        batch_size=int(batch_size),
    )
    return ce_npz


def ensure_gat_scores(
    paths: AdapterV6Paths,
    *,
    split: str,
    windows_path: Path,
    force: bool = False,
    device: str | None = None,
) -> Path:
    n_train, n_test = _split_n(paths)
    n = n_train if split == "train" else n_test
    out_json = _gat_scores_path(paths, split=split, n=n)
    if out_json.is_file() and not force:
        data = json.loads(out_json.read_text(encoding="utf-8"))
        if _gat_scores_from_eval(out_json):
            return out_json
        # file exists but no scores — rebuild
    gat_mvp = paths.root / paths.gat.gat_mvp_dir
    ckpt = gat_mvp / "ckpt_e4.pt"
    if not ckpt.is_file():
        raise FileNotFoundError(f"missing E4 ckpt: {ckpt}")
    dev = torch.device(
        device
        or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"dumping GAT E4 scores → {out_json} device={dev}", flush=True)
    result = evaluate(
        out_dir=gat_mvp,
        windows_path=windows_path,
        ckpt_path=ckpt,
        ablation="E4",
        device=dev,
        top_n=50,
        fair=True,
        dump_scores=True,
    )
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return out_json


def _node_title_abs(
    g: nx.DiGraph, pid: str, node_by: dict[str, Any] | None = None
) -> tuple[str, str]:
    node = pid
    if node_by is not None:
        node = node_by.get(normalize_arxiv_id(pid), pid)
    if node not in g:
        return pid, ""
    attrs = g.nodes[node]
    title = (attrs.get("title") or pid).strip()
    abstract = (attrs.get("abstract") or "").strip()
    return title, abstract


def fuse_top_pools(
    *,
    paths: AdapterV6Paths,
    windows_path: Path,
    ce_npz: Path,
    gat_eval_json: Path,
    o2_gexf: Path,
    pool_n: int = 50,
    alpha: float = DEFAULT_ALPHA,
    rrf_k: int = DEFAULT_RRF_K,
    rrf_w: float = DEFAULT_RRF_W,
) -> list[dict[str, Any]]:
    windows = np.load(windows_path, allow_pickle=True)
    ce = np.load(ce_npz, allow_pickle=True)
    ce_scores = ce["ce_scores"]
    ce_by_q: dict[str, int] = {}
    for i, qid in enumerate(ce["query_ids"].tolist()):
        raw = str(qid)
        ce_by_q[raw] = i
        ce_by_q[normalize_arxiv_id(raw)] = i
    gat_raw = _gat_scores_from_eval(gat_eval_json)
    if not gat_raw:
        raise RuntimeError(f"{gat_eval_json} has no per-query scores")
    gat_by_q: dict[str, dict[str, Any]] = {}
    for raw, row in gat_raw.items():
        gat_by_q[raw] = row
        gat_by_q[normalize_arxiv_id(raw)] = row
    ids = _load_id_map(paths.root / paths.gat.gat_mvp_dir)
    g = nx.read_gexf(str(o2_gexf), node_type=None, relabel=False, version="1.2draft")
    node_by = {normalize_arxiv_id(str(n)): n for n in g.nodes}

    rows: list[dict[str, Any]] = []
    n = int(windows["query_ids"].shape[0])
    for i in range(n):
        raw_qid = str(windows["query_ids"][i])
        qid = normalize_arxiv_id(raw_qid)
        nc = int(windows["n_cands"][i])
        gat_row = gat_by_q.get(qid) or gat_by_q.get(raw_qid)
        ci = ce_by_q.get(qid)
        if ci is None:
            ci = ce_by_q.get(raw_qid)
        if gat_row is None or ci is None or nc < 1:
            continue
        sg = np.asarray(gat_row["scores"][:nc], dtype=np.float32)
        sc = np.asarray(ce_scores[ci, :nc], dtype=np.float32)
        m_full = min(sg.size, sc.size, nc)
        sg, sc = sg[:m_full], sc[:m_full]
        blend = _fuse_scores(
            sg=sg,
            sc=sc,
            alpha=float(alpha),
            fuse_mode="l0_rrf",
            rrf_k=int(rrf_k),
            rrf_w=float(rrf_w),
        )
        order = blend.argsort()[::-1][: max(1, int(pool_n))]
        cand_globals = windows["cand_idx"][i, :m_full]
        pool: list[dict[str, Any]] = []
        for j in order:
            gi = int(cand_globals[int(j)])
            if gi < 0 or gi >= len(ids):
                continue
            pid = normalize_arxiv_id(str(ids[gi]))
            title, _abs = _node_title_abs(g, pid, node_by)
            pool.append(
                {
                    "id": pid,
                    "title": title,
                    "score": float(blend[int(j)]),
                }
            )
        q_title, q_abs = _node_title_abs(g, qid, node_by)
        rows.append(
            {
                "source_id": qid,
                "title": q_title,
                "abstract": q_abs[:800],
                "pool": pool,
                "n_pool": len(pool),
                "pool_source": POOL_SOURCE,
                "alpha": float(alpha),
                "rrf_k": int(rrf_k),
                "rrf_w": float(rrf_w),
            }
        )
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def dump_split_pools(
    paths: AdapterV6Paths,
    *,
    split: str,
    pool_n: int = 50,
    force_ce: bool = False,
    force_gat: bool = False,
    device: str | None = None,
) -> dict[str, Any]:
    ensure_adapter_dirs(paths)
    windows_path = _windows_path(paths, split=split)
    ce_npz = ensure_ce_cache(
        paths, split=split, windows_path=windows_path, force=force_ce
    )
    gat_json = ensure_gat_scores(
        paths,
        split=split,
        windows_path=windows_path,
        force=force_gat,
        device=device,
    )
    o2 = paths.gat.full_o2_gexf
    if not o2.is_file():
        raise FileNotFoundError(f"missing full O2 gexf: {o2}")
    rows = fuse_top_pools(
        paths=paths,
        windows_path=windows_path,
        ce_npz=ce_npz,
        gat_eval_json=gat_json,
        o2_gexf=o2,
        pool_n=pool_n,
    )
    out = paths.pools_train if split == "train" else paths.pools_test
    write_jsonl(out, rows)
    meta = {
        "split": split,
        "n_rows": len(rows),
        "pool_n": int(pool_n),
        "pool_source": POOL_SOURCE,
        "out": str(out),
        "windows": str(windows_path),
        "ce_cache": str(ce_npz),
        "gat_scores": str(gat_json),
        "gexf_o2": str(o2),
    }
    print(json.dumps(meta, indent=2), flush=True)
    return meta


def dump_domain_pools(
    domain: str,
    *,
    root: Path | None = None,
    pool_n: int = 50,
    splits: tuple[str, ...] = ("train", "test"),
    force_ce: bool = False,
    force_gat: bool = False,
    device: str | None = None,
) -> dict[str, Any]:
    paths = resolve_adapter_v6(domain, root)
    report: dict[str, Any] = {"domain": paths.domain, "splits": {}}
    for split in splits:
        report["splits"][split] = dump_split_pools(
            paths,
            split=split,
            pool_n=pool_n,
            force_ce=force_ce,
            force_gat=force_gat,
            device=device,
        )
    return report
