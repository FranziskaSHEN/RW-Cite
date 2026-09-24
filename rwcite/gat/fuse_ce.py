"""Cross-encoder/GAT score-and-rank fusion on frozen structural Top-400 windows.

The score-fusion component is ``α z(s_GAT) + (1-α) z(s_CE)``.  The
reported score-and-rank fusion adds standardized reciprocal-rank evidence with
offset ``k=20`` and weight ``w=0.4``.  Historical command-line mode names are
retained only so frozen experiment artifacts remain reproducible.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np

from rwcite.gat import BASELINE_FULL381_HITS_AT_10
from rwcite.gat import protocol as P
from rwcite.gat.compare import compare
from rwcite.gat.ranking_metrics import hits_at_k as _hits_at_k
from rwcite.gat.ranking_metrics import mean_ir_fields, query_ir_metrics


def _zscore(arr: np.ndarray) -> np.ndarray:
    """NaN-aware z-score (matches train._zscore_1d)."""
    if arr.size == 0:
        return arr
    x = np.asarray(arr, dtype=np.float32)
    mu = float(np.nanmean(x))
    sd = float(np.nanstd(x))
    if not np.isfinite(sd) or sd < 1e-6:
        return np.zeros_like(x, dtype=np.float32)
    return (x - mu) / sd


def assert_ce_cache_matches(
    ce_npz: Path,
    *,
    ce_path: Path,
    force: bool = False,
) -> None:
    """Refuse silent reuse of a cache scored with a different CE checkpoint."""
    if force or not ce_npz.is_file():
        return
    try:
        meta = np.load(ce_npz, allow_pickle=True)
    except OSError as e:
        raise RuntimeError(f"unreadable CE cache {ce_npz}: {e}") from e
    if "ce_path" not in meta.files:
        raise RuntimeError(
            f"CE cache {ce_npz} missing ce_path meta; pass --force-cache to rebuild"
        )
    cached = meta["ce_path"]
    if isinstance(cached, np.ndarray):
        if cached.size == 1:
            cached = cached.reshape(-1)[0]
        else:
            cached = str(cached.tolist())
    # Unwrap accidental list/tuple wrappers from merge scripts.
    if isinstance(cached, (list, tuple)) and len(cached) == 1:
        cached = cached[0]
    cached_s = str(cached).strip()
    if cached_s.startswith("[") and cached_s.endswith("]") and "'" in cached_s:
        # e.g. "['/path/to/ce']" from str(np.array([...]))
        inner = cached_s.strip("[]").strip().strip("'\"")
        if inner:
            cached_s = inner
    want_s = str(ce_path)

    def _norm(p: str) -> str:
        try:
            return str(Path(p).resolve())
        except OSError:
            return p

    def _ce_key(p: str) -> str:
        parts = Path(p).parts
        if "ce_sent" in parts:
            i = parts.index("ce_sent")
            return "/".join(parts[i:])
        return Path(p).name

    if _norm(cached_s) == _norm(want_s) or _ce_key(cached_s) == _ce_key(want_s):
        return
    raise RuntimeError(
        f"CE cache {ce_npz} was scored with ce_path={cached_s!r}, "
        f"but selected CE is {want_s}. Pass --force-cache or "
        f"--ce-cache matching this checkpoint (default L0: "
        f"ce_scores_test381_sent_s1.npz)."
    )


def _node_text(g: nx.Graph, pid: str) -> str:
    d = g.nodes.get(pid) or {}
    title = str(d.get("title") or "").strip()
    abstract = str(d.get("abstract") or "").strip()
    return f"{title} [SEP] {abstract}".strip()


def _ranks_desc(scores: np.ndarray) -> np.ndarray:
    """0-based ranks; higher score → rank 0."""
    order = np.asarray(scores, dtype=np.float64).argsort()[::-1]
    ranks = np.empty(order.size, dtype=np.int64)
    ranks[order] = np.arange(order.size, dtype=np.int64)
    return ranks


def _rrf_from_scores(
    sc_ce: np.ndarray,
    sc_gat: np.ndarray,
    *,
    rrf_k: int,
) -> np.ndarray:
    """Reciprocal rank fusion of two score vectors (higher is better)."""
    k = max(1, int(rrf_k))
    rc = _ranks_desc(sc_ce)
    rg = _ranks_desc(sc_gat)
    return (1.0 / (k + rc) + 1.0 / (k + rg)).astype(np.float32)


def _fuse_scores(
    *,
    sg: np.ndarray,
    sc: np.ndarray,
    alpha: float,
    fuse_mode: str = "l0",
    rrf_k: int = 20,
    rrf_w: float = 0.4,
) -> np.ndarray:
    """Build fused ranking scores on an aligned candidate slice.

    CE sentinels / NaNs are masked before z-score (same as L0); for RRF ranks
    those positions are pushed to the bottom via -inf.
    """
    bad = ~np.isfinite(sc) | (sc <= -1e8)
    sc_clean = np.where(bad, np.nan, sc)
    zg = _zscore(sg)
    zc = _zscore(sc_clean)
    zc = np.where(bad | ~np.isfinite(zc), zg, zc)
    l0 = float(alpha) * zg + (1.0 - float(alpha)) * zc
    mode = (fuse_mode or "l0").lower().strip()
    if mode in ("", "l0", "blend", "zscore"):
        return l0.astype(np.float32)
    if mode in ("l0_rrf", "l0_plus", "rrf_hybrid"):
        sc_rank = np.where(bad, -np.inf, sc.astype(np.float64))
        rrf = _rrf_from_scores(sc_rank, sg, rrf_k=rrf_k)
        w = float(rrf_w)
        return (w * _zscore(rrf) + (1.0 - w) * l0).astype(np.float32)
    raise ValueError(f"unknown fuse_mode {fuse_mode!r}; use l0|l0_rrf")


def cache_ce_scores(
    *,
    root: Path,
    windows_path: Path,
    out_npz: Path,
    ce_path: Path,
    gexf: Path,
    batch_size: int = 32,
    max_queries: int = 0,
    start_query: int = 0,
    mode: str = "ce_sent",
    short: bool = True,
    max_sents: int = 3,
) -> dict[str, Any]:
    """Score each window candidate with frozen CE.

    mode:
      - ce_sent: title/abs + incoming cite-sentences (E1 / RELEASE_2.0 C₂s SHORT)
      - title_abs: title+abstract only (legacy weak cache)
    """
    from rwcite.ranker.rr_ranker_ce_sent import (
        cand_text_with_sents,
        load_ce_sent_ranker,
    )
    from rwcite.ranker.rr_ranker_ce import _query_text

    import os

    os.environ["RR_RANKER_CE_SENT_PATH"] = str(ce_path)
    if short:
        os.environ["RR_CE_SENT_SHORT"] = "1"
    else:
        os.environ["RR_CE_SENT_SHORT"] = "0"
    os.environ["RR_CE_SENT_MAX_SENTS"] = str(int(max_sents))
    # Align with RELEASE_2.0 / E1 eval (default code path is 384 — wrong for gate)
    if not (os.environ.get("RR_CE_SENT_MAX_LEN") or "").strip():
        os.environ["RR_CE_SENT_MAX_LEN"] = "256"
    # force reload so max_length picks up env
    import rwcite.ranker.rr_ranker_ce_sent as _ces

    _ces._CE_SENT_CACHE = None
    _ces._CE_SENT_PATH = None
    ce = load_ce_sent_ranker(str(ce_path))
    if ce is None:
        raise RuntimeError(f"failed to load CE from {ce_path}")
    print(
        f"CE loaded max_length={ce.max_length} short={short} max_sents={max_sents}",
        flush=True,
    )

    print(f"loading gexf {gexf} … mode={mode} short={short} max_sents={max_sents}", flush=True)
    g = nx.read_gexf(gexf)
    windows = np.load(windows_path, allow_pickle=True)
    ids = json.loads((windows_path.parent / "id_map.json").read_text(encoding="utf-8"))[
        "ids"
    ]
    n_all = int(windows["query_ids"].shape[0])
    start = max(0, int(start_query))
    end = n_all
    if max_queries > 0:
        end = min(n_all, start + int(max_queries))
    else:
        end = n_all
    n = max(0, end - start)

    w = int(windows["cand_idx"].shape[1])
    scores = np.full((n, w), np.nan, dtype=np.float32)
    qids: list[str] = []
    t0 = time.perf_counter()
    for local_i, i in enumerate(range(start, end)):
        qid = str(windows["query_ids"][i])
        qids.append(qid)
        nc = int(windows["n_cands"][i])
        q_node = g.nodes.get(qid) or {}
        q_txt = _query_text(
            str(q_node.get("title") or ""), str(q_node.get("abstract") or "")
        )
        texts: list[str] = []
        for j in range(nc):
            gi = int(windows["cand_idx"][i, j])
            if gi < 0:
                texts.append("")
                continue
            pid = ids[gi]
            if mode == "ce_sent":
                c_node = g.nodes.get(pid) or {}
                texts.append(
                    cand_text_with_sents(
                        {
                            "id": pid,
                            "title": str(c_node.get("title") or ""),
                            "abstract": str(c_node.get("abstract") or ""),
                        },
                        graph=g,
                        exclude_citer=qid,
                        max_sents=max_sents,
                        short=short,
                    )
                )
            else:
                texts.append(_node_text(g, pid))
        if not any(t.strip() for t in texts):
            continue
        scored = ce.score_pairs(q_txt, texts, batch_size=batch_size)
        for j, sc in enumerate(scored):
            if j < nc and texts[j].strip():
                scores[local_i, j] = float(sc)
            else:
                scores[local_i, j] = -1e9
        if (local_i + 1) % 20 == 0:
            print(
                f"ce cache {start+local_i+1}/{n_all} (shard {local_i+1}/{n}) "
                f"elapsed={time.perf_counter()-t0:.1f}s",
                flush=True,
            )

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    mode_tag = f"ce_sent_short{int(short)}_ms{max_sents}" if mode == "ce_sent" else "title_abs"
    np.savez_compressed(
        out_npz,
        query_ids=np.asarray(qids, dtype=object),
        ce_scores=scores,
        windows=str(windows_path),
        ce_path=str(ce_path),
        gexf=str(gexf),
        mode=mode_tag,
        short=np.asarray([int(short)]),
        max_sents=np.asarray([int(max_sents)]),
        max_length=np.asarray([int(getattr(ce, "max_length", 256))]),
    )
    meta = {
        "n": n,
        "path": str(out_npz),
        "ce_path": str(ce_path),
        "elapsed_s": float(time.perf_counter() - t0),
        "mode": mode_tag,
        "short": short,
        "max_sents": max_sents,
        "max_length": int(getattr(ce, "max_length", 256)),
        "start_query": start,
        "end_query": end,
    }
    print(json.dumps(meta, indent=2), flush=True)
    return meta


def _gat_scores_from_eval(eval_json: Path) -> dict[str, dict[str, Any]]:
    data = json.loads(eval_json.read_text(encoding="utf-8"))
    out: dict[str, dict[str, Any]] = {}
    for row in data.get("per_query") or []:
        qid = str(row.get("source_id") or "")
        if not qid or "scores" not in row:
            continue
        out[qid] = {
            "scores": np.asarray(row["scores"], dtype=np.float32),
            "cand_ids": list(row.get("cand_ids") or []),
            "gold_hint": int(row.get("n_gold_in_window") or 0),
        }
    return out


def _shortlist_indices(
    *,
    mode: str,
    m_cap: int,
    sg: np.ndarray,
    sc: np.ndarray,
) -> np.ndarray:
    """Return positions into the full window for cascade shortlist.

    struct: keep leading M (windows already struct-ranked).
    gat / ce: top-M by that score (descending).
    """
    n = int(sg.size)
    m = max(1, min(int(m_cap), n)) if m_cap and m_cap > 0 else n
    mode = (mode or "none").lower().strip()
    if mode in ("", "none", "full", "all"):
        return np.arange(n, dtype=np.int64)
    if mode == "struct":
        return np.arange(m, dtype=np.int64)
    if mode == "gat":
        return sg.argsort()[::-1][:m].astype(np.int64)
    if mode == "ce":
        return sc.argsort()[::-1][:m].astype(np.int64)
    raise ValueError(f"unknown shortlist mode {mode!r}; use none|struct|gat|ce")


def blend_sweep(
    *,
    root: Path,
    windows_path: Path,
    ce_npz: Path,
    gat_eval_json: Path,
    alphas: list[float],
    out_prefix: str,
    shortlist: str = "none",
    shortlist_m: int = 0,
    bar_hits_at_10: float | None = None,
    fuse_mode: str = "l0",
    rrf_k: int = 20,
    rrf_w: float = 0.4,
) -> dict[str, Any]:
    windows = np.load(windows_path, allow_pickle=True)
    ce = np.load(ce_npz, allow_pickle=True)
    ce_scores = ce["ce_scores"]
    ce_qids = [str(x) for x in ce["query_ids"].tolist()]
    ce_by_q = {qid: i for i, qid in enumerate(ce_qids)}
    gat_by_q = _gat_scores_from_eval(gat_eval_json)
    if not gat_by_q:
        raise RuntimeError(
            f"{gat_eval_json} has no per-query scores; re-run evaluate --dump-scores"
        )

    ids = json.loads((windows_path.parent / "id_map.json").read_text(encoding="utf-8"))[
        "ids"
    ]
    n = int(windows["query_ids"].shape[0])
    bar = float(bar_hits_at_10) if bar_hits_at_10 is not None else float(BASELINE_FULL381_HITS_AT_10)
    research_bar = float(P.L0_DEFAULT_HITS_AT_10)
    fuse_mode_n = (fuse_mode or "l0").lower().strip()
    if fuse_mode_n in ("l0_plus", "rrf_hybrid"):
        fuse_mode_n = "l0_rrf"
    cascade = (shortlist or "none").lower() not in ("", "none", "full", "all")
    if fuse_mode_n == "l0_rrf":
        protocol = (
            "ce_gat_l0_rrf_hybrid_cascade" if cascade else "ce_gat_l0_rrf_hybrid"
        )
    else:
        protocol = (
            "ce_gat_zscore_blend_L0_cascade" if cascade else "ce_gat_zscore_blend_L0"
        )
    results: dict[str, Any] = {
        "alphas": {},
        "bar": bar,
        "research_bar_l0_3_0": research_bar,
        "shortlist": shortlist,
        "shortlist_m": int(shortlist_m),
        "fuse_mode": fuse_mode_n,
        "rrf_k": int(rrf_k),
        "rrf_w": float(rrf_w),
        "protocol": protocol,
    }
    # windows live under …/ranker/gat_mvp/; eval JSON under sibling …/ranker/eval/
    eval_dir = windows_path.parent.parent / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    best_alpha = None
    best_h10 = -1.0
    for alpha in alphas:
        gold_in_sl = 0.0
        n_gold_all = 0.0
        per: list[dict[str, Any]] = []
        for i in range(n):
            qid = str(windows["query_ids"][i])
            nc = int(windows["n_cands"][i])
            gold_full = windows["gold_mask"][i, :nc].astype(np.float32)
            n_gold_all += float(windows["n_gold"][i]) if "n_gold" in windows.files else float(gold_full.sum())
            gat_row = gat_by_q.get(qid)
            ci = ce_by_q.get(qid)
            if gat_row is None or ci is None:
                per.append({"source_id": qid, "hits_at_10": 0, "skip": True})
                continue
            sg = np.asarray(gat_row["scores"][:nc], dtype=np.float32)
            sc = np.asarray(ce_scores[ci, :nc], dtype=np.float32)
            # align length
            m_full = min(sg.size, sc.size, nc)
            sg, sc, gold_full = sg[:m_full], sc[:m_full], gold_full[:m_full]
            idx = _shortlist_indices(
                mode=shortlist, m_cap=shortlist_m, sg=sg, sc=sc
            )
            sg_s, sc_s, gold = sg[idx], sc[idx], gold_full[idx]
            gold_in_sl += float(gold.sum())
            blend = _fuse_scores(
                sg=sg_s,
                sc=sc_s,
                alpha=float(alpha),
                fuse_mode=fuse_mode_n,
                rrf_k=rrf_k,
                rrf_w=rrf_w,
            )
            ir = query_ir_metrics(blend, gold)
            order = blend.argsort()[::-1][:30]
            cand_globals = windows["cand_idx"][i, :m_full][idx]
            top_ids = [
                ids[int(cand_globals[j])] for j in order if int(cand_globals[j]) >= 0
            ]
            row: dict[str, Any] = {
                "source_id": qid,
                "n_gold_in_window": int(gold_full.sum()),
                "n_gold_in_shortlist": int(gold.sum()),
                "shortlist_m": int(idx.size),
                "top_ids": top_ids,
            }
            row.update(ir)
            per.append(row)
        means = mean_ir_fields(per)
        mean10 = means["mean_hits_at_10"]
        mean30 = means["mean_hits_at_30"]
        sl_recall = float(gold_in_sl / max(n_gold_all, 1.0))
        summary = {
            "n": n,
            "mean_hits_at_10": mean10,
            "mean_hits_at_30": mean30,
            "mean_ndcg_at_10": means["mean_ndcg_at_10"],
            "mean_mrr_at_10": means["mean_mrr_at_10"],
            "mean_recall_at_30": means["mean_recall_at_30"],
            "mean_recall_at_100": means["mean_recall_at_100"],
            "mean_recall_at_400": means["mean_recall_at_400"],
            "recall_note": (
                "R@k = |Top-k ∩ G_window| / |G_window|; "
                "R@400 window-capped (pool=400); R@1000 not defined"
            ),
            "alpha_gat": float(alpha),
            "bar_hits_at_10": bar,
            "research_bar_l0_3_0": research_bar,
            "protocol": results["protocol"],
            "fuse_mode": fuse_mode_n,
            "rrf_k": int(rrf_k),
            "rrf_w": float(rrf_w),
            "shortlist": shortlist,
            "shortlist_m": int(shortlist_m),
            "shortlist_gold_recall": sl_recall,
            "gat_eval": str(gat_eval_json),
            "ce_cache": str(ce_npz),
        }
        if fuse_mode_n == "l0_rrf":
            tag = (
                f"{out_prefix}_a{str(alpha).replace('.', 'p')}"
                f"_k{int(rrf_k)}_w{str(rrf_w).replace('.', 'p')}"
            )
        else:
            tag = f"{out_prefix}_a{str(alpha).replace('.', 'p')}"
        out_path = eval_dir / f"{tag}_merged.json"
        out_path.write_text(
            json.dumps({"summary": summary, "per_query": per}, indent=2),
            encoding="utf-8",
        )
        cmp = compare(
            root / P.BASELINE_FULL381_MERGED,
            out_path,
            bar=bar,
        )
        results["alphas"][str(alpha)] = {
            "summary": summary,
            "compare": cmp,
            "path": str(out_path),
        }
        print(
            f"fuse={fuse_mode_n} shortlist={shortlist} M={shortlist_m} "
            f"alpha={alpha:.2f} rrf_k={rrf_k} rrf_w={rrf_w:.2f} "
            f"@10={mean10:.4f} @30={mean30:.4f} "
            f"nDCG@10={means['mean_ndcg_at_10']:.4f} "
            f"MRR@10={means['mean_mrr_at_10']:.4f} "
            f"R@100={means['mean_recall_at_100']:.4f} "
            f"R@400={means['mean_recall_at_400']:.4f} "
            f"sl_recall={sl_recall:.4f} "
            f"promote={cmp.get('promote_ok')}",
            flush=True,
        )
        if mean10 > best_h10:
            best_h10 = mean10
            best_alpha = alpha

    results["best_alpha"] = best_alpha
    results["best_hits_at_10"] = best_h10
    alpha0 = (results["alphas"].get("0.0") or results["alphas"].get("0") or {}).get(
        "summary", {}
    ).get("mean_hits_at_10")
    results["alpha0_hits_at_10"] = alpha0
    results["alpha0_near_e1"] = bool(
        alpha0 is not None and abs(float(alpha0) - BASELINE_FULL381_HITS_AT_10) <= 0.15
    )
    results["beats_e4"] = bool(best_h10 > 3.4803149606299213)
    results["beats_l0_3_0"] = bool(best_h10 > research_bar)
    results["target_band_ge_3_8"] = bool(best_h10 >= 3.8)
    results["target_band_ge_4_2"] = bool(best_h10 >= 4.2)
    # complementarity: interior alpha wins → fusion valuable
    results["fusion_valuable"] = bool(
        best_alpha is not None
        and 0.0 < float(best_alpha) < 1.0
        and best_h10 > BASELINE_FULL381_HITS_AT_10
        and results["beats_e4"]
    )
    if best_alpha is not None:
        best_row = results["alphas"].get(str(best_alpha)) or {}
        results["best_path"] = best_row.get("path")
        results["best_shortlist_gold_recall"] = (best_row.get("summary") or {}).get(
            "shortlist_gold_recall"
        )
    return results


def _expand_alphas(coarse: list[float], *, fine_step: float = 0.05) -> list[float]:
    """Coarse grid then fine neighborhood around provisional best."""
    xs = sorted(set(float(a) for a in coarse))
    # provisional best will be refined after first pass by caller; here just densify [0,1]
    fine = [round(i * fine_step, 4) for i in range(0, int(1.0 / fine_step) + 1)]
    return sorted(set(xs + fine))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="CE↔GAT L0 score blend")
    ap.add_argument("--step", choices=["cache_ce", "blend", "all"], default="all")
    ap.add_argument("--domain", default="ewm")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--windows", default=None)
    ap.add_argument("--ce-cache", default=None)
    ap.add_argument("--gat-eval", default=None, help="GAT eval JSON with --dump-scores")
    ap.add_argument(
        "--ckpt",
        default=None,
        help="GAT ckpt for dump-scores when gat-eval lacks scores (default out-dir/ckpt_e4.pt)",
    )
    ap.add_argument(
        "--ce-path",
        default=None,
        help="nofuture CE dir (default: …/c2s-hn-…_nofuture or promoted)",
    )
    ap.add_argument("--gexf", default=None)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-queries", type=int, default=0)
    ap.add_argument("--start-query", type=int, default=0, help="shard start index into windows")
    ap.add_argument(
        "--mode",
        choices=["ce_sent", "title_abs"],
        default="ce_sent",
        help="ce_sent = E1-aligned cite-sentence CE; title_abs = legacy weak cache",
    )
    ap.add_argument("--short", type=int, default=1, help="RR_CE_SENT_SHORT (1=production)")
    ap.add_argument("--max-sents", type=int, default=3)
    ap.add_argument("--force-cache", action="store_true")
    ap.add_argument("--fine-alpha", action="store_true", help="also sweep α step 0.05")
    ap.add_argument("--alphas", default="0,0.25,0.5,0.75,1")
    ap.add_argument("--out-prefix", default=None)
    ap.add_argument("--report-name", default=None)
    ap.add_argument(
        "--shortlist",
        default="none",
        choices=["none", "full", "struct", "gat", "ce"],
        help="Cascade L0'': shortlist before z-score blend (none=full window)",
    )
    ap.add_argument(
        "--shortlist-m",
        type=int,
        default=0,
        help="Shortlist size M (0=all when shortlist=none)",
    )
    ap.add_argument(
        "--research-bar",
        type=float,
        default=None,
        help="Optional compare bar (default charter E1 bar; cascade uses L0 3.924 via report)",
    )
    ap.add_argument(
        "--fuse-mode",
        default="l0",
        choices=["l0", "l0_rrf", "l0_plus"],
        help="l0=z-score blend; l0_rrf/l0_plus=w*z(RRF)+(1-w)*L0",
    )
    ap.add_argument("--rrf-k", type=int, default=20, help="RRF constant k (L0+)")
    ap.add_argument("--rrf-w", type=float, default=0.4, help="weight on z(RRF) in L0+")
    args = ap.parse_args(argv)

    root = P.root_path()
    out_dir = Path(args.out_dir) if args.out_dir else root / P.GAT_MVP_CKPT_DIR
    if not out_dir.is_absolute():
        out_dir = root / out_dir
    windows_path = Path(args.windows) if args.windows else out_dir / "windows_test381.npz"
    default_cache = (
        out_dir / "ce_scores_test381_sent_s1.npz"
        if args.mode == "ce_sent"
        else out_dir / "ce_scores_test381.npz"
    )
    ce_npz = Path(args.ce_cache) if args.ce_cache else default_cache
    gexf = Path(args.gexf) if args.gexf else root / P.NOFUTURE_GEXF
    ce_path = Path(args.ce_path) if args.ce_path else None
    if ce_path is None:
        cand = [
            root
            / "embodied_world_model_retrieval/ranker/ce_sent/stage1_nofuture",
            root
            / "embodied_world_model_retrieval/ranker/ce_sent/c2s-hn-scibert_scivocab_uncased_nofuture",
            root
            / "embodied_world_model_retrieval/ranker/ce_sent/c2s-hn-scibert_scivocab_uncased",
        ]
        ce_path = next((p for p in cand if (p / "config.json").is_file()), cand[0])

    out_prefix = args.out_prefix or (
        "ewm_gat_l0_rrf_test381"
        if args.fuse_mode in ("l0_rrf", "l0_plus")
        else (
            "ewm_gat_ce_sent_s1_blend_test381"
            if args.mode == "ce_sent"
            else "ewm_gat_ce_blend_test381"
        )
    )
    report_name = args.report_name or (
        "l0_rrf_blend_report.json"
        if args.fuse_mode in ("l0_rrf", "l0_plus")
        else (
            "ce_sent_s1_blend_report.json"
            if args.mode == "ce_sent"
            else "ce_blend_report.json"
        )
    )

    if args.step in ("cache_ce", "all"):
        if not args.force_cache and ce_npz.is_file():
            assert_ce_cache_matches(ce_npz, ce_path=ce_path, force=False)
        if args.force_cache or not ce_npz.is_file():
            cache_ce_scores(
                root=root,
                windows_path=windows_path,
                out_npz=ce_npz,
                ce_path=ce_path,
                gexf=gexf,
                batch_size=args.batch_size,
                max_queries=args.max_queries,
                start_query=args.start_query,
                mode=args.mode,
                short=bool(args.short),
                max_sents=args.max_sents,
            )
        else:
            print(f"reuse ce cache {ce_npz} (ce_path ok)", flush=True)

    if args.step in ("blend", "all"):
        assert_ce_cache_matches(ce_npz, ce_path=ce_path, force=args.force_cache)
        if not ce_npz.is_file():
            raise SystemExit(f"missing CE cache {ce_npz}; run --step cache_ce first")
        scores_path = (
            root
            / "embodied_world_model_retrieval/ranker/eval/ewm_gat_mvp_test381_e4_scores_merged.json"
        )
        gat_eval = Path(args.gat_eval) if args.gat_eval else scores_path
        if not gat_eval.is_absolute():
            gat_eval = root / gat_eval
        need_dump = True
        if gat_eval.is_file():
            pq = (json.loads(gat_eval.read_text()).get("per_query") or [{}])[0]
            need_dump = "scores" not in pq
        if need_dump:
            from rwcite.gat.evaluate import evaluate
            import torch

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            ckpt = Path(args.ckpt) if args.ckpt else out_dir / "ckpt_e4.pt"
            if not ckpt.is_absolute():
                ckpt = root / ckpt
            if not ckpt.is_file():
                raise SystemExit(f"missing {ckpt} for dump-scores")
            print(f"re-eval dump-scores from {ckpt}", flush=True)
            result = evaluate(
                out_dir=out_dir,
                windows_path=windows_path,
                ckpt_path=ckpt,
                ablation="E4",
                device=device,
                fair=True,
                dump_scores=True,
            )
            if args.gat_eval:
                gat_eval = Path(args.gat_eval)
                if not gat_eval.is_absolute():
                    gat_eval = root / gat_eval
            else:
                gat_eval = scores_path
            gat_eval.parent.mkdir(parents=True, exist_ok=True)
            gat_eval.write_text(json.dumps(result, indent=2), encoding="utf-8")
            print(f"wrote {gat_eval}", flush=True)

        alphas = [float(x) for x in args.alphas.split(",") if x.strip()]
        if args.fine_alpha:
            alphas = _expand_alphas(alphas, fine_step=0.05)
        report = blend_sweep(
            root=root,
            windows_path=windows_path,
            ce_npz=ce_npz,
            gat_eval_json=gat_eval,
            alphas=alphas,
            out_prefix=out_prefix,
            shortlist=args.shortlist,
            shortlist_m=args.shortlist_m,
            bar_hits_at_10=args.research_bar,
            fuse_mode=args.fuse_mode,
            rrf_k=args.rrf_k,
            rrf_w=args.rrf_w,
        )
        report["ce_mode"] = args.mode
        report["ce_cache"] = str(ce_npz)
        report_path = out_dir / report_name
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(
            json.dumps(
                {
                    k: report[k]
                    for k in (
                        "fuse_mode",
                        "rrf_k",
                        "rrf_w",
                        "best_alpha",
                        "best_hits_at_10",
                        "alpha0_hits_at_10",
                        "alpha0_near_e1",
                        "beats_e4",
                        "beats_l0_3_0",
                        "target_band_ge_3_8",
                        "fusion_valuable",
                        "shortlist",
                        "shortlist_m",
                        "best_path",
                    )
                    if k in report
                },
                indent=2,
            )
        )
        print(f"wrote {report_path}", flush=True)
        if not report.get("alpha0_near_e1") and (args.shortlist or "none") in (
            "none",
            "full",
        ):
            print(
                f"WARN: α=0 @10={report.get('alpha0_hits_at_10')} not near E1=2.735 "
                "(±0.15) — check CE-sent cache alignment",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
