"""Citation-aware cross-encoder over query and candidate evidence.

Activate with RR_RANKER_CE_SENT_PATH=/path/to/checkpoint_dir
Historical citation-link blending remains available for artifact reproduction;
the paper framework uses the structural window and citation-aware CE directly.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from rwcite.ranker.reference_recommend import normalize_arxiv_id
from rwcite.ranker.rr_ranker import rank_universe
from rwcite.ranker.rr_ranker_ce import CERanker, _query_text
from rwcite.ranker.rr_ranker_cite_link import (
    CiteLinkMLP,
    citelink_rank_full_universe,
    score_cands_citelink,
)
from rwcite.ranker.rr_struct_coarse import rank_struct_coarse

_CE_SENT_CACHE: CERanker | None = None
_CE_SENT_PATH: str | None = None


def rr_ce_sent_params() -> dict[str, Any]:
    def _i(name: str, default: int) -> int:
        raw = (os.environ.get(name) or "").strip()
        if not raw:
            return default
        try:
            return max(1, int(raw))
        except ValueError:
            return default

    def _f(name: str, default: float) -> float:
        raw = (os.environ.get(name) or "").strip()
        if not raw:
            return default
        try:
            return float(raw)
        except ValueError:
            return default

    coarse = (os.environ.get("RR_CE_SENT_COARSE") or "struct").strip().lower()
    if coarse not in ("struct", "citelink", "union"):
        coarse = "struct"
    win = _i("RR_CE_SENT_WINDOW", 400)
    # SCORE_M: CE only top-M of shortlist; 0 or unset → score all window
    score_m_raw = (os.environ.get("RR_CE_SENT_SCORE_M") or "").strip()
    if score_m_raw:
        try:
            score_m = max(1, int(score_m_raw))
        except ValueError:
            score_m = win
    else:
        score_m = win
    union_cap = _i("RR_CE_SENT_UNION_CAP", max(win, 500))
    return {
        "path": (os.environ.get("RR_RANKER_CE_SENT_PATH") or "").strip(),
        "window": win,
        "score_m": score_m,
        "coarse": coarse,
        "union_cap": union_cap,
        "batch_size": _i("RR_CE_SENT_BATCH", 32),
        "max_length": _i("RR_CE_SENT_MAX_LEN", 384),
        "max_sents": _i("RR_CE_SENT_MAX_SENTS", 4),
        "citelink_blend": _f("RR_CE_SENT_CITELINK_BLEND", 0.25),
        "struct_blend": _f("RR_CE_SENT_STRUCT_BLEND", 0.05),
        "short_cand": (os.environ.get("RR_CE_SENT_SHORT") or "").strip().lower()
        in ("1", "true", "yes"),
    }


def _merge_union(
    struct_list: list[dict[str, Any]],
    cl_list: list[dict[str, Any]],
    *,
    cap: int,
) -> list[dict[str, Any]]:
    """Prefer citelink order, then fill from struct; dedupe to cap."""
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for c in cl_list + struct_list:
        pid = c.get("id") or ""
        if not pid or pid in seen:
            continue
        seen.add(pid)
        out.append(
            {
                "id": pid,
                "title": c.get("title") or "",
                "abstract": c.get("abstract") or "",
                "source": c.get("source") or "union",
                "score": float(c.get("score") or 0.0),
            }
        )
        if len(out) >= cap:
            break
    return out


def build_ce_sent_shortlist(
    *,
    title: str,
    abstract: str,
    universe_pack: dict[str, Any],
    graph: Any,
    window: int,
    coarse_mode: str = "struct",
    union_cap: int | None = None,
    citelink: CiteLinkMLP | None = None,
    cand_embs: dict[str, np.ndarray] | None = None,
    q_emb: np.ndarray | None = None,
    encode: Any = None,
    retriever: Any = None,
    query_id: str = "",
) -> list[dict[str, Any]]:
    """Build CE shortlist: struct / citelink-fullU / union (R1)."""
    u_n = len(universe_pack.get("universe") or [])
    win = min(int(window), u_n) if u_n else 0
    if win <= 0:
        return []
    struct = rank_struct_coarse(
        title=title,
        abstract=abstract,
        universe_pack=universe_pack,
        graph=graph,
        n=win,
        encode=encode,
        retriever=retriever,
        query_id=query_id,
    )
    mode = (coarse_mode or "struct").strip().lower()
    if mode == "struct" or citelink is None or q_emb is None or cand_embs is None:
        return struct
    cl = citelink_rank_full_universe(
        universe_pack=universe_pack,
        graph=graph,
        model=citelink,
        q_emb=q_emb,
        cand_embs=cand_embs,
        n=win,
    )
    if mode == "citelink":
        return cl
    # union
    cap = int(union_cap if union_cap is not None else max(win, 500))
    return _merge_union(struct, cl, cap=cap)


def collect_cite_sentence_pool(
    graph: Any,
    citee_id: str,
    *,
    exclude_citer: str = "",
    citer_filter: Callable[[str], bool] | None = None,
) -> list[tuple[str, str]]:
    """Collect non-empty incoming citation records in graph order."""
    citee = normalize_arxiv_id(citee_id)
    excl = normalize_arxiv_id(exclude_citer) if exclude_citer else ""
    if graph is None or not citee or citee not in getattr(graph, "nodes", {}):
        return []
    records: list[tuple[str, str]] = []
    try:
        # Preserve graph order because training and inference serialize the same
        # bounded evidence representation.
        preds = graph.predecessors(citee)
    except Exception:  # noqa: BLE001
        return []
    for pred in preds:
        cr = normalize_arxiv_id(str(pred))
        if excl and cr == excl:
            continue
        if citer_filter is not None and not citer_filter(cr):
            continue
        try:
            data = graph.edges[pred, citee]
        except Exception:  # noqa: BLE001
            continue
        sent = str((data or {}).get("sentence") or "").strip()
        if sent:
            records.append((cr, sent))
    return records


def select_cite_sentence_records(
    records: list[tuple[str, str]], *, max_sents: int
) -> list[tuple[str, str]]:
    """Apply the trained stride cap to an already constructed record pool."""
    if not records or max_sents <= 0:
        return []
    selected = list(records)
    if len(selected) > max_sents:
        step = max(1, len(selected) // max_sents)
        selected = selected[::step][:max_sents]
    return selected


def collect_cite_sentences(
    graph: Any,
    citee_id: str,
    *,
    exclude_citer: str = "",
    max_sents: int = 4,
    citer_filter: Callable[[str], bool] | None = None,
) -> list[str]:
    """Collect bounded incoming citation sentences, excluding the query citer."""
    records = collect_cite_sentence_pool(
        graph,
        citee_id,
        exclude_citer=exclude_citer,
        citer_filter=citer_filter,
    )
    return [
        sentence
        for _, sentence in select_cite_sentence_records(records, max_sents=max_sents)
    ]


def cand_text_with_sents(
    c: dict[str, Any],
    *,
    graph: Any = None,
    exclude_citer: str = "",
    max_sents: int = 4,
    short: bool | None = None,
) -> str:
    """Build candidate side text. short=True (v4b): title + Cite sentences, tiny abs."""
    if short is None:
        short = bool(rr_ce_sent_params().get("short_cand"))
    title = (c.get("title") or "").strip()
    abstract = (c.get("abstract") or "").strip()
    sents = collect_cite_sentences(
        graph, c.get("id") or "", exclude_citer=exclude_citer, max_sents=max_sents
    )
    if short:
        # Prefer cite context; keep a short abs fallback if no sentences
        if sents:
            block = "\n".join(f"Cite: {s[:220]}" for s in sents)
            return f"{title}\n{block}".strip()[:900]
        return f"{title}\n{abstract[:180]}".strip()
    base = f"{title}\n{abstract}".strip()[:900]
    if not sents:
        return base
    block = "\n".join(f"Cite: {s[:280]}" for s in sents)
    return f"{base}\n{block}".strip()[:1600]


def cand_text_from_sentences(
    c: dict[str, Any],
    sentences: list[str],
    *,
    short: bool,
) -> str:
    """Serialize a frozen evidence set exactly as the cross-encoder expects."""
    title = (c.get("title") or "").strip()
    abstract = (c.get("abstract") or "").strip()
    sents = list(sentences)
    if short:
        if sents:
            block = "\n".join(f"Cite: {s[:220]}" for s in sents)
            return f"{title}\n{block}".strip()[:900]
        return f"{title}\n{abstract[:180]}".strip()
    base = f"{title}\n{abstract}".strip()[:900]
    if not sents:
        return base
    block = "\n".join(f"Cite: {s[:280]}" for s in sents)
    return f"{base}\n{block}".strip()[:1600]


def dump_window_ce_citelink_scores(
    *,
    title: str,
    abstract: str,
    universe_pack: dict[str, Any],
    graph: Any,
    ce: CERanker,
    exclude_id: str = "",
    citelink: CiteLinkMLP | None = None,
    cand_embs: dict[str, np.ndarray] | None = None,
    q_emb: np.ndarray | None = None,
    window: int | None = None,
    batch_size: int | None = None,
    max_sents: int | None = None,
    coarse_mode: str | None = None,
    score_m: int | None = None,
    encode: Any = None,
    retriever: Any = None,
) -> dict[str, Any]:
    """Score CE shortlist once; return raw ce/citelink scores for offline blend sweep.

    Always scores citelink on the shortlist when model+embs are provided (even if
    online blend=0), so β can be re-applied without re-running CE.
    """
    params = rr_ce_sent_params()
    win = int(window if window is not None else params["window"])
    bs = int(batch_size if batch_size is not None else params["batch_size"])
    ms = int(max_sents if max_sents is not None else params["max_sents"])
    mode = (coarse_mode if coarse_mode is not None else params["coarse"]).strip().lower()
    m_ce = int(score_m if score_m is not None else params["score_m"])

    coarse = build_ce_sent_shortlist(
        title=title,
        abstract=abstract,
        universe_pack=universe_pack,
        graph=graph,
        window=win,
        coarse_mode=mode,
        union_cap=int(params["union_cap"]),
        citelink=citelink,
        cand_embs=cand_embs,
        q_emb=q_emb,
        encode=encode,
        retriever=retriever,
        query_id=exclude_id,
    )
    u_ids = [c["id"] for c in (universe_pack.get("universe") or []) if c.get("id")]
    u_size = int((universe_pack.get("meta") or {}).get("universe_size") or len(u_ids))
    if not coarse:
        return {
            "universe_ids": u_ids,
            "universe_size": u_size,
            "window": win,
            "cands": [],
        }

    cl_scores = np.zeros(len(coarse), dtype=np.float32)
    if citelink is not None and cand_embs is not None and q_emb is not None:
        scored = score_cands_citelink(
            cands=coarse,
            graph=graph,
            model=citelink,
            q_emb=q_emb,
            cand_embs=cand_embs,
            universe_pack=universe_pack,
        )
        by_id = {c["id"]: float(sc) for sc, c in scored}
        for i, c in enumerate(coarse):
            cl_scores[i] = by_id.get(c["id"], 0.0)

    # Cascade priority: citelink when present, else struct order (matches online)
    if citelink is not None and cand_embs is not None and q_emb is not None:
        cheap = cl_scores.copy()
    else:
        cheap = np.array(
            [1.0 - float(i) / float(max(len(coarse), 1)) for i in range(len(coarse))],
            dtype=np.float32,
        )

    m_ce = max(1, min(m_ce, len(coarse)))
    cheap_order = np.argsort(-cheap)
    ce_idx = [int(i) for i in cheap_order[:m_ce]]
    q = _query_text(title, abstract)
    ce_scores = np.full(len(coarse), np.nan, dtype=np.float32)
    if ce_idx:
        texts = [
            cand_text_with_sents(
                coarse[i],
                graph=graph,
                exclude_citer=exclude_id,
                max_sents=ms,
                short=bool(params.get("short_cand")),
            )
            for i in ce_idx
        ]
        scored_ce = np.asarray(ce.score_pairs(q, texts, batch_size=bs), dtype=np.float32)
        for j, i in enumerate(ce_idx):
            ce_scores[i] = float(scored_ce[j])

    cands: list[dict[str, Any]] = []
    for i, c in enumerate(coarse):
        row: dict[str, Any] = {
            "id": c["id"],
            "struct_idx": int(i),
            "citelink_score": float(cl_scores[i]),
            "ce_scored": bool(not np.isnan(ce_scores[i])),
        }
        if row["ce_scored"]:
            row["ce_score"] = float(ce_scores[i])
        cands.append(row)

    return {
        "universe_ids": u_ids,
        "universe_size": u_size,
        "window": int(len(coarse)),
        "score_m": int(m_ce),
        "cands": cands,
    }


def load_ce_sent_ranker(path: str | None = None) -> CERanker | None:
    """Load v4 CE checkpoint dir (needs config.json)."""
    global _CE_SENT_CACHE, _CE_SENT_PATH
    params = rr_ce_sent_params()
    p = (path if path is not None else params["path"]).strip()
    if not p:
        return None
    resolved = Path(p)
    if not resolved.is_absolute():
        for cand in (Path(p), Path.cwd() / p, Path(__file__).resolve().parents[3] / p):
            if (cand / "config.json").is_file():
                resolved = cand.resolve()
                break
    if not (resolved / "config.json").is_file():
        return None
    key = str(resolved)
    if _CE_SENT_CACHE is not None and _CE_SENT_PATH == key:
        return _CE_SENT_CACHE
    try:
        ce = CERanker.from_pretrained(
            resolved, max_length=int(params["max_length"])
        )
    except Exception:  # noqa: BLE001
        return None
    _CE_SENT_CACHE = ce
    _CE_SENT_PATH = key
    return _CE_SENT_CACHE


def _zscore(arr: np.ndarray) -> np.ndarray:
    if arr.size == 0:
        return arr
    mu = float(arr.mean())
    sd = float(arr.std())
    if sd < 1e-6:
        return np.zeros_like(arr)
    return (arr - mu) / sd


def rank_universe_ce_sent(
    *,
    title: str,
    abstract: str,
    universe_pack: dict[str, Any],
    graph: Any,
    ce: CERanker,
    n: int = 30,
    window: int | None = None,
    exclude_id: str = "",
    citelink: CiteLinkMLP | None = None,
    cand_embs: dict[str, np.ndarray] | None = None,
    q_emb: np.ndarray | None = None,
    citelink_blend: float | None = None,
    batch_size: int | None = None,
    max_sents: int | None = None,
    coarse_mode: str | None = None,
    score_m: int | None = None,
    shortlist: list[dict[str, Any]] | None = None,
    encode: Any = None,
    retriever: Any = None,
    timings_out: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    t_rank0 = time.perf_counter()
    params = rr_ce_sent_params()
    win = int(window if window is not None else params["window"])
    bs = int(batch_size if batch_size is not None else params["batch_size"])
    ms = int(max_sents if max_sents is not None else params["max_sents"])
    blend = float(
        citelink_blend
        if citelink_blend is not None
        else params["citelink_blend"]
    )
    struct_b = float(params["struct_blend"])
    mode = (coarse_mode if coarse_mode is not None else params["coarse"]).strip().lower()
    m_ce = int(score_m if score_m is not None else params["score_m"])

    t0 = time.perf_counter()
    if shortlist is not None:
        coarse = list(shortlist)
    else:
        coarse = build_ce_sent_shortlist(
            title=title,
            abstract=abstract,
            universe_pack=universe_pack,
            graph=graph,
            window=win,
            coarse_mode=mode,
            union_cap=int(params["union_cap"]),
            citelink=citelink,
            cand_embs=cand_embs,
            q_emb=q_emb,
            encode=encode,
            retriever=retriever,
            query_id=exclude_id,
        )
    t_shortlist = time.perf_counter() - t0
    if not coarse:
        if timings_out is not None:
            timings_out.clear()
            timings_out.update(
                {
                    "struct_shortlist_s": float(t_shortlist),
                    "citelink_cheap_s": 0.0,
                    "cand_text_s": 0.0,
                    "ce_score_s": 0.0,
                    "blend_rank_s": 0.0,
                    "rank_total_s": float(time.perf_counter() - t_rank0),
                }
            )
        return []

    # Cheap scores for cascade priority + pad
    t0 = time.perf_counter()
    cheap = np.zeros(len(coarse), dtype=np.float32)
    if citelink is not None and cand_embs is not None and q_emb is not None:
        scored = score_cands_citelink(
            cands=coarse,
            graph=graph,
            model=citelink,
            q_emb=q_emb,
            cand_embs=cand_embs,
            universe_pack=universe_pack,
        )
        by_id = {c["id"]: float(sc) for sc, c in scored}
        for i, c in enumerate(coarse):
            cheap[i] = by_id.get(c["id"], 0.0)
    else:
        cheap = np.array(
            [1.0 - float(i) / float(max(len(coarse), 1)) for i in range(len(coarse))],
            dtype=np.float32,
        )
    t_cheap = time.perf_counter() - t0

    # CE cascade: score only top-M by cheap order
    m_ce = max(1, min(m_ce, len(coarse)))
    cheap_order = np.argsort(-cheap)
    ce_idx = set(int(i) for i in cheap_order[:m_ce])
    q = _query_text(title, abstract)
    ce_scores = np.full(len(coarse), -1e9, dtype=np.float32)
    t_cand_text = 0.0
    t_ce = 0.0
    if ce_idx:
        idx_list = sorted(ce_idx)
        t0 = time.perf_counter()
        texts = [
            cand_text_with_sents(
                coarse[i],
                graph=graph,
                exclude_citer=exclude_id,
                max_sents=ms,
                short=bool(params.get("short_cand")),
            )
            for i in idx_list
        ]
        t_cand_text = time.perf_counter() - t0
        t0 = time.perf_counter()
        scored_ce = np.asarray(
            ce.score_pairs(q, texts, batch_size=bs), dtype=np.float32
        )
        t_ce = time.perf_counter() - t0
        for j, i in enumerate(idx_list):
            ce_scores[i] = float(scored_ce[j])

    t0 = time.perf_counter()
    cl_scores = cheap.copy()  # reuse cite-link cheap when available
    if blend > 0 and citelink is None:
        cl_scores = np.zeros(len(coarse), dtype=np.float32)

    struct = np.array(
        [1.0 - float(i) / float(max(len(coarse), 1)) for i in range(len(coarse))],
        dtype=np.float32,
    )

    # Blend only among CE-scored; pad rest by cheap below floor
    mask = np.array([i in ce_idx for i in range(len(coarse))], dtype=bool)
    final = np.zeros(len(coarse), dtype=np.float32)
    if mask.any():
        zce = _zscore(ce_scores[mask])
        zcl = _zscore(cl_scores[mask]) if blend > 0 else np.zeros_like(zce)
        zst = struct[mask]
        final[mask] = (1.0 - blend) * zce + blend * zcl + struct_b * zst
        floor = float(final[mask].min()) - 1.0
    else:
        floor = 0.0
    if (~mask).any():
        z_cheap = _zscore(cheap[~mask])
        final[~mask] = floor - 1.0 + 0.01 * z_cheap

    order = np.argsort(-final)
    out: list[dict[str, Any]] = []
    for i in order[:n]:
        c = coarse[int(i)]
        out.append(
            {
                "id": c["id"],
                "title": c.get("title") or "",
                "abstract": c.get("abstract") or "",
                "source": "ce_sent",
                "score": float(final[int(i)]),
                "ce_score": float(ce_scores[int(i)]),
                "citelink_score": float(cl_scores[int(i)]),
                "ce_scored": bool(int(i) in ce_idx),
            }
        )
    t_blend = time.perf_counter() - t0
    if timings_out is not None:
        timings_out.clear()
        timings_out.update(
            {
                "struct_shortlist_s": float(t_shortlist),
                "citelink_cheap_s": float(t_cheap),
                "cand_text_s": float(t_cand_text),
                "ce_score_s": float(t_ce),
                "blend_rank_s": float(t_blend),
                "rank_total_s": float(time.perf_counter() - t_rank0),
            }
        )
    return out
