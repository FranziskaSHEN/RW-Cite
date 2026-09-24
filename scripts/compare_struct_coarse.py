#!/usr/bin/env python3
"""Compare struct coarse-rank recipes on the RW graph (recall@k vs gold)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rwcite.ranker.reference_recommend import normalize_arxiv_id  # noqa: E402
from rwcite.ranker.rr_pool_build import _rank_with_mlp  # noqa: E402
from rwcite.ranker.rr_ranker import RankerModel, heuristic_score, load_ranker, rank_universe  # noqa: E402
from rwcite.ranker.rr_ranker_mlp import load_mlp_ranker  # noqa: E402
from rwcite.ranker.rr_universe import build_rr_universe  # noqa: E402
from rwcite.runtime.deps import get_domain_cache, get_retriever_service  # noqa: E402


def _load_sources(path: Path, max_n: int) -> list[dict]:
    seen: set[str] = set()
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            sid = normalize_arxiv_id(str(r.get("source_id") or ""))
            if not sid or sid in seen:
                continue
            seen.add(sid)
            rows.append(r)
    return rows[:max_n]


def _gold(row: dict, graph) -> set[str]:
    sid = normalize_arxiv_id(str(row.get("source_id") or ""))
    gold = {normalize_arxiv_id(g["id"]) for g in row.get("gold") or [] if g.get("id")}
    if graph is not None and sid in graph:
        gold |= {normalize_arxiv_id(str(t)) for t in graph.successors(sid)}
    gold.discard(sid)
    return {g for g in gold if g}


def _recall_at(ids: list[str], gold: set[str], k: int) -> float:
    if not gold:
        return float("nan")
    return len(gold & set(ids[:k])) / len(gold)


def _rank_heuristic(title, abstract, pack, graph, win):
    return rank_universe(
        title=title,
        abstract=abstract,
        universe_pack=pack,
        graph=graph,
        n=win,
        emb_sims=None,
        model=None,
    )


def _rank_linear(title, abstract, pack, graph, win, model: RankerModel | None):
    return rank_universe(
        title=title,
        abstract=abstract,
        universe_pack=pack,
        graph=graph,
        n=win,
        emb_sims=None,
        model=model,
    )


def _rank_logistic_emb(title, abstract, pack, graph, win, model: RankerModel, encode):
    """Non-CE path: emb_sim on struct top-500, then logistic."""
    universe = list(pack.get("universe") or [])
    coarse = rank_universe(
        title=title,
        abstract=abstract,
        universe_pack=pack,
        graph=graph,
        n=min(500, len(universe)),
        emb_sims=None,
        model=None,
    )
    ids = [c["id"] for c in coarse]
    by = {c["id"]: c for c in universe}
    qv = encode([f"{title}\n{abstract}".strip()[:800]])[0]
    texts = [f"{by[i].get('title') or ''}\n{by[i].get('abstract') or ''}"[:800] for i in ids if i in by]
    emb_sims: dict[str, float] = {}
    if texts:
        pv = encode(texts)
        emb_sims = {i: float(v @ qv) for i, v in zip(ids, pv)}
    return rank_universe(
        title=title,
        abstract=abstract,
        universe_pack=pack,
        graph=graph,
        n=win,
        emb_sims=emb_sims,
        model=model,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="ewm_rw")
    ap.add_argument(
        "--sources-jsonl",
        default="embodied_world_model_retrieval_rw/data/reference_recommend_p3/dev400.jsonl",
    )
    ap.add_argument("--max-sources", type=int, default=200)
    ap.add_argument("--window", type=int, default=400)
    ap.add_argument("--ks", default="50,200,400,800")
    ap.add_argument(
        "--shared-ranker-root",
        type=Path,
        default=None,
        help=(
            "Optional directory containing model.npz and mlp.npz from a "
            "deprecated shared ranker. Omit it to compare only the current "
            "domain-local ranker and heuristic."
        ),
    )
    args = ap.parse_args()

    ks = [int(k) for k in args.ks.split(",") if k.strip()]
    win = args.window
    rows = _load_sources(ROOT / args.sources_jsonl, args.max_sources)
    graph = get_domain_cache().get(args.domain).graph
    retriever = get_retriever_service()

    def search_fn(query: str, limit: int):
        hits = retriever.search(
            scope="domain",
            domain_id=args.domain,
            query=query,
            offset=0,
            limit=limit,
            sort="relevance",
        )
        return list(hits.get("results") or [])

    def encode(texts: list[str]) -> np.ndarray:
        import torch

        retriever._load_embedder()
        tok, emb = retriever._tokenizer, retriever._embedder
        device = next(emb.parameters()).device
        outs = []
        for i in range(0, len(texts), 32):
            batch = texts[i : i + 32]
            inputs = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=256)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                v = emb(**inputs).last_hidden_state[:, 0, :].float()
                v = torch.nn.functional.normalize(v, dim=-1)
            outs.append(v.cpu().numpy())
        return np.concatenate(outs, 0)

    rw = RankerModel.load(ROOT / "models/rr-pool-ranker-ewm-rw-p3/model.npz")

    methods: dict[str, callable] = {
        "heuristic": lambda t, a, p, g: _rank_heuristic(t, a, p, g, win),
        "rw_ranknet_cepath": lambda t, a, p, g: _rank_linear(t, a, p, g, win, rw),
    }
    if args.shared_ranker_root is not None:
        shared_root = args.shared_ranker_root.expanduser().resolve()
        shared_model_path = shared_root / "model.npz"
        shared_mlp_path = shared_root / "mlp.npz"
        if not shared_model_path.is_file():
            ap.error(f"shared ranker model is missing: {shared_model_path}")
        shared = RankerModel.load(shared_model_path)
        methods["shared_logistic_cepath"] = (
            lambda t, a, p, g: _rank_linear(t, a, p, g, win, shared)
        )
        methods["shared_logistic_embpath"] = (
            lambda t, a, p, g: _rank_logistic_emb(t, a, p, g, win, shared, encode)
        )
        if shared_mlp_path.is_file():
            mlp = load_mlp_ranker(str(shared_mlp_path))
            if mlp is not None:
                methods["shared_mlp_fullU"] = lambda t, a, p, g: _rank_with_mlp(
                    t, a, p, g, encode, mlp, win, full_u=True
                )
                methods["shared_mlp_prefilter800"] = lambda t, a, p, g: _rank_with_mlp(
                    t, a, p, g, encode, mlp, win, full_u=False
                )

    acc = {m: {f"recall@{k}": [] for k in ks} for m in methods}
    acc["heuristic"]["jaccard_vs_heuristic@400"] = [1.0]  # self

    for qi, row in enumerate(rows):
        title = row.get("title") or ""
        abstract = row.get("abstract") or ""
        sid = normalize_arxiv_id(str(row.get("source_id") or ""))
        gold = _gold(row, graph)
        if not gold:
            continue
        pack = build_rr_universe(
            title,
            abstract,
            graph,
            search_fn=search_fn,
            exclude_id=sid,
            multi_query=True,
            prf=True,
        )
        if not pack.get("universe"):
            continue

        ranked: dict[str, list[str]] = {}
        for name, fn in methods.items():
            top = fn(title, abstract, pack, graph)
            ids = [c["id"] for c in top]
            ranked[name] = ids
            for k in ks:
                acc[name][f"recall@{k}"].append(_recall_at(ids, gold, k))

        base = set(ranked["heuristic"][:win])
        for name, ids in ranked.items():
            if name == "heuristic":
                continue
            key = f"jaccard_vs_heuristic@{win}"
            if key not in acc[name]:
                acc[name][key] = []
            other = set(ids[:win])
            acc[name][key].append(len(base & other) / max(1, len(base | other)))

        if (qi + 1) % 25 == 0:
            print(f"  [{qi + 1}/{len(rows)}]", flush=True)

    report = {"n_sources": len(rows), "window": win, "methods": {}}
    for name, buckets in acc.items():
        report["methods"][name] = {
            k: float(np.nanmean(v)) if v else float("nan") for k, v in buckets.items()
        }
        if "recall@400" in buckets:
            report["methods"][name]["recall@400_delta_vs_heuristic"] = (
                report["methods"][name]["recall@400"]
                - report["methods"]["heuristic"]["recall@400"]
            )

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
