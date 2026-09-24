"""Build citation-sentence SFT data from fused pools and the full graph."""

from __future__ import annotations

import json
import random
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import networkx as nx

from rwcite.adapter import POOL_SOURCE
from rwcite.adapter.paths import AdapterV6Paths, ensure_adapter_dirs, resolve_adapter_v6
from rwcite.latex.cite_sentence_clean import clean_cite_sentence
from rwcite.ranker.reference_recommend import normalize_arxiv_id

_TEMPLATE_PREFIX = "we build on "
_ALLOWED_TASKS = frozenset({"cite_bundle", "cite_one"})


def _norm_tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", (text or "").lower()))


def _jaccard(a: str, b: str) -> float:
    ta, tb = _norm_tokens(a), _norm_tokens(b)
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / max(1, len(ta | tb))


def _is_template_sentence(sent: str) -> bool:
    s = (sent or "").strip().lower()
    return s.startswith(_TEMPLATE_PREFIX) or len(s) < 15


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _load_id_list(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        return {normalize_arxiv_id(str(x)) for x in raw if x}
    if isinstance(raw, dict):
        for key in ("source_ids", "ids", "train_sources", "test_sources"):
            if key in raw and isinstance(raw[key], list):
                return {normalize_arxiv_id(str(x)) for x in raw[key] if x}
    return set()


def assert_no_leak(
    train_ids: set[str],
    *,
    test_jsonl: Path,
    split_dir: Path,
) -> dict[str, Any]:
    test_ids = {
        normalize_arxiv_id(str(r.get("source_id") or ""))
        for r in _load_jsonl(test_jsonl)
        if r.get("source_id")
    }
    frozen = _load_id_list(split_dir / "test_sources.json")
    inter = train_ids & test_ids
    leak_frozen = train_ids & frozen
    report = {
        "n_train": len(train_ids),
        "n_test_jsonl": len(test_ids),
        "n_frozen_test": len(frozen),
        "leak_train_test": len(inter),
        "leak_train_frozen": len(leak_frozen),
    }
    if inter or leak_frozen:
        raise SystemExit(
            "ERROR: train/test leakage "
            f"inter={len(inter)} frozen={len(leak_frozen)} "
            f"sample={list(inter | leak_frozen)[:5]}"
        )
    return report


def _full_gold(g: nx.DiGraph, source: str) -> tuple[list[dict], list[str]]:
    outs = list(g.successors(source))
    gold_ids = [normalize_arxiv_id(str(t)) for t in outs]
    gold: list[dict] = []
    for tid_raw in outs:
        tid = normalize_arxiv_id(str(tid_raw))
        edge = g.edges[source, tid_raw]
        sent = str(edge.get("sentence") or edge.get("label") or "").strip()
        quality = str(edge.get("sentence_quality") or "").strip()
        cres = clean_cite_sentence(sent) if sent else None
        if cres is not None and cres.text:
            sent = cres.text
            if not cres.ok:
                quality = quality or "weak"
            elif not quality:
                quality = "ok"
        title = (g.nodes[tid_raw].get("title") or tid).strip()
        if len(sent) < 15:
            sent = f"We build on {title}."
            quality = "template"
        is_tmpl = _is_template_sentence(sent)
        is_garbage = bool(cres is not None and not cres.ok and not is_tmpl) or quality == "bad"
        gold.append(
            {
                "id": tid,
                "title": title,
                "abstract": (g.nodes[tid_raw].get("abstract") or "")[:500],
                "sentence": sent,
                "is_template": is_tmpl,
                "is_garbage": is_garbage,
            }
        )
    return gold, gold_ids


def greedy_diverse_selected(
    gold_by_id: dict[str, dict],
    in_pool_gold_ids: list[str],
    *,
    k: int,
    jaccard_thr: float = 0.8,
) -> list[dict]:
    items = [gold_by_id[i] for i in in_pool_gold_ids if i in gold_by_id]
    clean = [g for g in items if not g.get("is_garbage")]
    if clean:
        items = clean
    non_tmpl = [g for g in items if not g.get("is_template")]
    tmpl = [g for g in items if g.get("is_template")]
    ordered = non_tmpl + tmpl
    picked: list[dict] = []
    n_tmpl = 0
    for g in ordered:
        if len(picked) >= k:
            break
        if g.get("is_template") and n_tmpl >= 1:
            continue
        sent = g.get("sentence") or ""
        if any(_jaccard(sent, p.get("sentence") or "") >= jaccard_thr for p in picked):
            continue
        picked.append(
            {
                "id": g["id"],
                "title": g["title"],
                "abstract": (g.get("abstract") or "")[:500],
                "sentence": sent,
            }
        )
        if g.get("is_template"):
            n_tmpl += 1
    if not picked and ordered:
        g0 = ordered[0]
        picked.append(
            {
                "id": g0["id"],
                "title": g0["title"],
                "abstract": (g0.get("abstract") or "")[:500],
                "sentence": g0.get("sentence") or "",
            }
        )
    return picked


def expand_tasks(
    *,
    source_id: str,
    title: str,
    abstract: str,
    gold: list[dict],
    gold_ids: list[str],
    pool_ids: list[str],
    n_pool: int,
    jaccard_thr: float,
    rng: random.Random,
) -> list[dict]:
    gold_set = set(gold_ids)
    by_gold = {g["id"]: g for g in gold}
    in_pool = [pid for pid in pool_ids if pid in gold_set]
    if not in_pool:
        return []
    selected = greedy_diverse_selected(
        by_gold, in_pool, k=min(10, len(in_pool)), jaccard_thr=jaccard_thr
    )
    if not selected:
        return []
    cite_ids = [s["id"] for s in selected]
    k_eff = len(selected)
    shared = {
        "source_id": source_id,
        "title": title,
        "abstract": abstract[:800],
        "gold": gold,
        "gold_ids": gold_ids,
        "in_pool_gold_ids": in_pool,
        "full_gold": True,
        "label_policy": "cite_only_v6",
        "pool_source": POOL_SOURCE,
        "n_pool": n_pool,
        "paper_abs_chars": 800,
        "analysis_task_map": {"cite_bundle": "bundle", "cite_one": "cite_only"},
    }
    rows: list[dict] = []
    for _ in range(2):
        rows.append(
            {
                **shared,
                "task": "cite_bundle",
                "k": k_eff,
                "selected": [dict(s) for s in selected],
                "assistant_ids": list(cite_ids),
            }
        )
    for _ in range(2):
        s = dict(rng.choice(selected))
        rows.append(
            {
                **shared,
                "task": "cite_one",
                "k": 1,
                "selected": [s],
                "assistant_ids": [s["id"]],
                "paper": s,
            }
        )
    return rows


def build_cite_eval_rows(
    pool_rows: list[dict],
    g: nx.DiGraph,
    *,
    pool_n: int,
    node_by: dict[str, Any] | None = None,
) -> list[dict]:
    if node_by is None:
        node_by = {normalize_arxiv_id(str(n)): n for n in g.nodes}
    out: list[dict] = []
    for pr in pool_rows:
        sid = normalize_arxiv_id(str(pr.get("source_id") or ""))
        node = node_by.get(sid)
        if not sid or not node:
            continue
        gold, gold_ids = _full_gold(g, node)
        pool = list(pr.get("pool") or [])[:pool_n]
        pool_ids = [normalize_arxiv_id(str(p.get("id") or "")) for p in pool if p.get("id")]
        title = (pr.get("title") or g.nodes[node].get("title") or sid).strip()
        abstract = (
            pr.get("abstract") or g.nodes[node].get("abstract") or ""
        ).strip()[:800]
        out.append(
            {
                "task": "cite_eval",
                "source_id": sid,
                "title": title,
                "abstract": abstract,
                "gold": gold,
                "gold_ids": gold_ids,
                "pool_ids": pool_ids,
                "n_pool": len(pool_ids),
                "pool_source": POOL_SOURCE,
                "label_policy": "cite_only_v6",
            }
        )
    return out


def build_domain_v6(
    domain: str,
    *,
    root: Path | None = None,
    pool_n: int = 50,
    jaccard_thr: float = 0.8,
    seed: int = 42,
) -> dict[str, Any]:
    paths = resolve_adapter_v6(domain, root)
    ensure_adapter_dirs(paths)
    if not paths.pools_train.is_file():
        raise FileNotFoundError(
            f"missing {paths.pools_train}; run dump_rr_l0_pools first"
        )
    o2 = paths.gat.full_o2_gexf
    if not o2.is_file():
        raise FileNotFoundError(f"missing O2 gexf: {o2}")

    pool_rows = _load_jsonl(paths.pools_train)
    train_ids = {
        normalize_arxiv_id(str(r.get("source_id") or ""))
        for r in pool_rows
        if r.get("source_id")
    }
    leak = assert_no_leak(
        train_ids,
        test_jsonl=paths.gat.test_jsonl,
        split_dir=paths.gat.split_dir,
    )
    frozen_train = _load_id_list(paths.gat.split_dir / "train_sources.json")
    if frozen_train:
        extra = train_ids - frozen_train
        if extra:
            raise SystemExit(
                f"ERROR: pool sources not in train_sources.json n={len(extra)} "
                f"sample={list(extra)[:5]}"
            )

    g = nx.read_gexf(str(o2), node_type=None, relabel=False, version="1.2draft")
    # normalize node keys for membership: build alias map
    node_by = {normalize_arxiv_id(str(n)): n for n in g.nodes}

    rng = random.Random(seed)
    train_rows: list[dict] = []
    n_skipped = 0
    for pr in pool_rows:
        sid = normalize_arxiv_id(str(pr.get("source_id") or ""))
        node = node_by.get(sid)
        if not node:
            n_skipped += 1
            continue
        gold, gold_ids = _full_gold(g, node)
        pool = list(pr.get("pool") or [])[:pool_n]
        pool_ids = [
            normalize_arxiv_id(str(p.get("id") or "")) for p in pool if p.get("id")
        ]
        title = (pr.get("title") or g.nodes[node].get("title") or sid).strip()
        abstract = (
            pr.get("abstract") or g.nodes[node].get("abstract") or ""
        ).strip()
        expanded = expand_tasks(
            source_id=sid,
            title=title,
            abstract=abstract,
            gold=gold,
            gold_ids=gold_ids,
            pool_ids=pool_ids,
            n_pool=len(pool_ids),
            jaccard_thr=jaccard_thr,
            rng=rng,
        )
        if not expanded:
            n_skipped += 1
            continue
        train_rows.extend(expanded)

    rng.shuffle(train_rows)
    bad = [r for r in train_rows if r.get("task") not in _ALLOWED_TASKS]
    if bad:
        raise SystemExit(f"ERROR: forbidden tasks in train rows: {bad[0].get('task')}")

    with paths.train_jsonl.open("w", encoding="utf-8") as f:
        for r in train_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    cite_eval: list[dict] = []
    if paths.pools_test.is_file():
        cite_eval = build_cite_eval_rows(
            _load_jsonl(paths.pools_test), g, pool_n=pool_n, node_by=node_by
        )
        with paths.cite_eval_jsonl.open("w", encoding="utf-8") as f:
            for r in cite_eval:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    task_counts = dict(Counter(r["task"] for r in train_rows))
    meta = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "domain": paths.domain,
        "pool_source": POOL_SOURCE,
        "n_pool_sources": len(pool_rows),
        "n_train_rows": len(train_rows),
        "n_skipped_sources": n_skipped,
        "n_cite_eval": len(cite_eval),
        "task_counts": task_counts,
        "leak": leak,
        "train_jsonl": str(paths.train_jsonl),
        "cite_eval_jsonl": str(paths.cite_eval_jsonl) if cite_eval else "",
        "gexf_o2": str(o2),
        "pools_train": str(paths.pools_train),
    }
    paths.meta_json.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2), flush=True)
    return meta
