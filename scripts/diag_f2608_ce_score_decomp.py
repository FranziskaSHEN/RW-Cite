#!/usr/bin/env python3
"""Quantify cross-graph CE collapse on ewm 2608 frontier (shared ewm struct window).

For each query: build U → ewm-struct shortlist W=400 → score all W with ewm CE and AS CE.
Reports gold vs non-gold score gaps, gold rank retention, Spearman of two CE orders.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

os.environ.setdefault("RR_STRUCT_COARSE", "linear")
os.environ.setdefault("RR_CE_SENT_SHORT", "1")
os.environ.setdefault("RR_CE_SENT_WINDOW", "400")
os.environ.setdefault("RR_CE_SENT_MAX_SENTS", "3")
os.environ.setdefault("RR_CE_SENT_CITELINK_BLEND", "0")
os.environ["RR_RANKER_PATH"] = str(
    (ROOT / "embodied_world_model_retrieval/ranker/struct/model.npz").resolve()
)

from rwcite.ranker.reference_recommend import normalize_arxiv_id  # noqa: E402
from rwcite.ranker.rr_ranker_ce_sent import (  # noqa: E402
    build_ce_sent_shortlist,
    cand_text_with_sents,
    load_ce_sent_ranker,
)
from rwcite.ranker.rr_universe import build_rr_universe  # noqa: E402
from rwcite.runtime.deps import get_domain_cache, get_retriever_service  # noqa: E402


def _load_rows(path: Path) -> list[dict]:
    seen: set[str] = set()
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("task") not in (None, "", "bundle"):
                continue
            sid = normalize_arxiv_id(str(r.get("source_id") or ""))
            if not sid or sid in seen:
                continue
            seen.add(sid)
            rows.append(r)
    return rows


def main() -> None:
    max_q = int(os.environ.get("MAX_Q", "196"))
    jsonl = Path(
        "embodied_world_model_retrieval/data/reference_recommend/test_frontier_2608.jsonl"
    )
    out = Path(
        os.environ.get(
            "OUT",
            "embodied_world_model_retrieval/ranker/eval/f2608_ce_score_decomp.json",
        )
    )
    rows = _load_rows(jsonl)[:max_q]

    graph = get_domain_cache().get("ewm").graph
    retriever = get_retriever_service()

    def search_fn(query: str, limit: int):
        hits = retriever.search(
            scope="domain",
            domain_id="ewm",
            query=query,
            offset=0,
            limit=limit,
            sort="relevance",
        )
        return list(hits.get("results") or [])

    ce_ewm = load_ce_sent_ranker(
        str(ROOT / "embodied_world_model_retrieval/ranker/ce_sent/c2s-hn-scibert_scivocab_uncased")
    )
    ce_as = load_ce_sent_ranker(
        str(
            ROOT
            / "embodied_world_model_retrieval_as/ranker/ce_sent/c2s-hn-scibert_scivocab_uncased.lin"
        )
    )
    assert ce_ewm and ce_as

    per: list[dict] = []
    for i, row in enumerate(rows):
        sid = normalize_arxiv_id(str(row.get("source_id") or ""))
        title = row.get("title") or ""
        abstract = row.get("abstract") or ""
        gold = {
            normalize_arxiv_id(g["id"])
            for g in (row.get("gold") or [])
            if g.get("id")
        }
        pack = build_rr_universe(
            title,
            abstract,
            graph,
            search_fn=search_fn,
            exclude_id=sid,
            multi_query=True,
            prf=True,
        )
        coarse = build_ce_sent_shortlist(
            title=title,
            abstract=abstract,
            universe_pack=pack,
            graph=graph,
            window=400,
            coarse_mode="struct",
            encode=None,
            retriever=retriever,
            query_id=sid,
        )
        if not coarse:
            continue
        ids = [c["id"] for c in coarse]
        is_gold = np.array([normalize_arxiv_id(x) in gold for x in ids], dtype=bool)
        q = f"{title}\n{abstract}".strip()[:800]
        texts = [
            cand_text_with_sents(
                c, graph=graph, exclude_citer=sid, max_sents=3, short=True
            )
            for c in coarse
        ]
        bs = int(os.environ.get("CE_BATCH", "32"))
        s_ewm = np.asarray(ce_ewm.score_pairs(q, texts, batch_size=bs), dtype=np.float64)
        s_as = np.asarray(ce_as.score_pairs(q, texts, batch_size=bs), dtype=np.float64)

        def ranks(scores: np.ndarray) -> np.ndarray:
            # rank 1 = best
            order = np.argsort(-scores, kind="stable")
            r = np.empty(len(scores), dtype=np.int32)
            r[order] = np.arange(1, len(scores) + 1)
            return r

        r_ewm = ranks(s_ewm)
        r_as = ranks(s_as)
        g_idx = np.nonzero(is_gold)[0]
        if g_idx.size == 0:
            continue

        # Spearman between CE score vectors on the shared window
        def _spearman(a: np.ndarray, b: np.ndarray) -> float:
            ra = np.argsort(np.argsort(-a))
            rb = np.argsort(np.argsort(-b))
            ra = ra - ra.mean()
            rb = rb - rb.mean()
            denom = float(np.sqrt((ra * ra).sum() * (rb * rb).sum()))
            return float((ra * rb).sum() / denom) if denom > 1e-12 else 0.0

        sp = _spearman(s_ewm, s_as)

        def gap(scores: np.ndarray) -> float:
            g = scores[is_gold]
            n = scores[~is_gold]
            if g.size == 0 or n.size == 0:
                return float("nan")
            return float(g.mean() - n.mean())

        def auc_proxy(scores: np.ndarray) -> float:
            # P(score_gold > score_neg)
            g = scores[is_gold]
            n = scores[~is_gold]
            if g.size == 0 or n.size == 0:
                return float("nan")
            # sample if huge
            if n.size > 2000:
                rng = np.random.default_rng(0)
                n = n[rng.choice(n.size, 2000, replace=False)]
            wins = sum(float((gi > n).mean()) for gi in g) / g.size
            return float(wins)

        rec = {
            "source_id": sid,
            "n_gold_total": len(gold),
            "n_gold_in_W": int(is_gold.sum()),
            "W": len(ids),
            "spearman_ce": sp,
            "gap_ewm": gap(s_ewm),
            "gap_as": gap(s_as),
            "auc_ewm": auc_proxy(s_ewm),
            "auc_as": auc_proxy(s_as),
            "mean_rank_gold_ewm": float(r_ewm[g_idx].mean()),
            "mean_rank_gold_as": float(r_as[g_idx].mean()),
            "hits10_ewm": int((r_ewm[g_idx] <= 10).sum()),
            "hits10_as": int((r_as[g_idx] <= 10).sum()),
            "hits50_ewm": int((r_ewm[g_idx] <= 50).sum()),
            "hits50_as": int((r_as[g_idx] <= 50).sum()),
            "gold_in_ewm50_surv_as50": int(
                ((r_ewm[g_idx] <= 50) & (r_as[g_idx] <= 50)).sum()
            ),
            "gold_in_ewm50_lost_as50": int(
                ((r_ewm[g_idx] <= 50) & (r_as[g_idx] > 50)).sum()
            ),
        }
        per.append(rec)
        if (i + 1) % 10 == 0 or i == 0:
            print(
                f"[{i+1}/{len(rows)}] {sid} sp={sp:.3f} "
                f"gap_ewm={rec['gap_ewm']:.3f} gap_as={rec['gap_as']:.3f} "
                f"@10 {rec['hits10_ewm']}→{rec['hits10_as']}",
                flush=True,
            )

    def mean(key: str) -> float:
        xs = [p[key] for p in per if p.get(key) == p.get(key)]
        return float(np.mean(xs)) if xs else float("nan")

    summary = {
        "n_queries": len(per),
        "mean_spearman_ce": mean("spearman_ce"),
        "mean_gap_ewm": mean("gap_ewm"),
        "mean_gap_as": mean("gap_as"),
        "mean_auc_ewm": mean("auc_ewm"),
        "mean_auc_as": mean("auc_as"),
        "mean_rank_gold_ewm": mean("mean_rank_gold_ewm"),
        "mean_rank_gold_as": mean("mean_rank_gold_as"),
        "mean_hits10_ewm": mean("hits10_ewm"),
        "mean_hits10_as": mean("hits10_as"),
        "mean_hits50_ewm": mean("hits50_ewm"),
        "mean_hits50_as": mean("hits50_as"),
        "sum_gold_ewm50": int(sum(p["hits50_ewm"] for p in per)),
        "sum_gold_lost_as50": int(sum(p["gold_in_ewm50_lost_as50"] for p in per)),
        "retention_top50_gold": float(
            sum(p["gold_in_ewm50_surv_as50"] for p in per)
            / max(1, sum(p["hits50_ewm"] for p in per))
        ),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "per_query": per}, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
