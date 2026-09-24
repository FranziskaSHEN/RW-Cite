"""v3e: rank structure window by cite-sentence ↔ query embedding alignment.

For each candidate citee, take incoming edge sentences (exclude query as citer),
score max cosine(query, sentence_emb). Optionally blend with cite-link logit.

Activate:
  RR_RANKER_SENT_PATH=/path/to/cite_sentences.npz
  Optional: RR_RANKER_CITELINK_PATH for blend (RR_SENT_CITELINK_BLEND, default 0.70)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

from rwcite.ranker.reference_recommend import normalize_arxiv_id
from rwcite.ranker.rr_ranker import rank_universe
from rwcite.ranker.rr_ranker_cite_link import (
    CiteLinkMLP,
    _cand_extra,
    pack_pair,
)


def rr_sent_params() -> dict[str, Any]:
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

    return {
        "path": (os.environ.get("RR_RANKER_SENT_PATH") or "").strip(),
        "window": _i("RR_SENT_WINDOW", 400),
        "max_sents": _i("RR_SENT_MAX_PER_CITEE", 16),
        "citelink_blend": _f("RR_SENT_CITELINK_BLEND", 0.70),
    }


def _resolve_path(path: str) -> Path | None:
    if not path:
        return None
    cands = [Path(path)]
    if not Path(path).is_absolute():
        repo = Path(__file__).resolve().parents[3]
        cands.append(repo / path)
        cands.append(Path.cwd() / path)
    for c in cands:
        if c.is_file():
            return c.resolve()
    return None


class CiteSentenceIndex:
    """citee → (citer_ids, emb_matrix rows)."""

    def __init__(self, citees: np.ndarray, citers: np.ndarray, emb: np.ndarray):
        self.emb = emb.astype(np.float32)
        self._by: dict[str, tuple[list[str], list[int]]] = {}
        for i, (ce, cr) in enumerate(zip(citees.tolist(), citers.tolist())):
            ce = normalize_arxiv_id(str(ce))
            cr = normalize_arxiv_id(str(cr))
            if not ce:
                continue
            if ce not in self._by:
                self._by[ce] = ([], [])
            self._by[ce][0].append(cr)
            self._by[ce][1].append(i)

    def max_sim(
        self,
        q: np.ndarray,
        citee: str,
        *,
        exclude_citer: str = "",
        max_sents: int = 16,
    ) -> float:
        citee = normalize_arxiv_id(citee)
        pack = self._by.get(citee)
        if not pack:
            return 0.0
        citers, idxs = pack
        excl = normalize_arxiv_id(exclude_citer) if exclude_citer else ""
        chosen: list[int] = [
            ix for cr, ix in zip(citers, idxs) if not (excl and cr == excl)
        ]
        if not chosen:
            return 0.0
        if len(chosen) > max_sents:
            # stride sample for coverage among many citers
            step = max(1, len(chosen) // max_sents)
            chosen = chosen[::step][:max_sents]
        S = self.emb[chosen]
        # q already normalized
        sims = S @ q
        return float(np.max(sims))


_INDEX: CiteSentenceIndex | None = None
_INDEX_PATH: str | None = None


def load_sent_index(path: str | None = None) -> CiteSentenceIndex | None:
    global _INDEX, _INDEX_PATH
    p = path if path is not None else rr_sent_params()["path"]
    resolved = _resolve_path(p) if p else None
    if resolved is None:
        return None
    key = str(resolved)
    if _INDEX is not None and _INDEX_PATH == key:
        return _INDEX
    data = np.load(resolved, allow_pickle=True)
    _INDEX = CiteSentenceIndex(data["citees"], data["citers"], data["emb"])
    _INDEX_PATH = key
    return _INDEX


def _zscore(arr: np.ndarray) -> np.ndarray:
    if arr.size == 0:
        return arr
    mu = float(arr.mean())
    sd = float(arr.std())
    if sd < 1e-6:
        return np.zeros_like(arr)
    return (arr - mu) / sd


def rank_universe_sent(
    *,
    title: str,
    abstract: str,
    universe_pack: dict[str, Any],
    graph: Any,
    sent_index: CiteSentenceIndex,
    q_emb: np.ndarray,
    n: int = 30,
    window: int | None = None,
    exclude_id: str = "",
    citelink: CiteLinkMLP | None = None,
    cand_embs: dict[str, np.ndarray] | None = None,
    citelink_blend: float | None = None,
) -> list[dict[str, Any]]:
    params = rr_sent_params()
    win = int(window if window is not None else params["window"])
    max_sents = int(params["max_sents"])
    blend = float(
        citelink_blend
        if citelink_blend is not None
        else params["citelink_blend"]
    )
    coarse = rank_universe(
        title=title,
        abstract=abstract,
        universe_pack=universe_pack,
        graph=graph,
        n=min(win, len(universe_pack.get("universe") or [])),
        emb_sims=None,
        model=None,
    )
    if not coarse:
        return []

    q = np.asarray(q_emb, dtype=np.float32).ravel()
    qn = float(np.linalg.norm(q))
    if qn > 1e-9:
        q = q / qn

    sent_scores = np.array(
        [
            sent_index.max_sim(
                q, c["id"], exclude_citer=exclude_id, max_sents=max_sents
            )
            for c in coarse
        ],
        dtype=np.float32,
    )

    cl_scores = np.zeros(len(coarse), dtype=np.float32)
    if citelink is not None and cand_embs and blend > 0:
        node_by = universe_pack.get("node_by") or {}
        rows = []
        keep_idx = []
        for i, c in enumerate(coarse):
            ce = cand_embs.get(c["id"])
            if ce is None:
                continue
            v = np.asarray(ce, dtype=np.float32).ravel()
            vn = float(np.linalg.norm(v))
            if vn > 1e-9:
                v = v / vn
            extra = _cand_extra(graph, node_by, c["id"]) if citelink.extra_dim else None
            if citelink.extra_dim and (extra is None or extra.size != citelink.extra_dim):
                extra = np.zeros((citelink.extra_dim,), dtype=np.float32)
            rows.append(pack_pair(q, v, extra))
            keep_idx.append(i)
        if rows:
            logits = citelink.score_batch(np.stack(rows))
            for j, i in enumerate(keep_idx):
                cl_scores[i] = float(logits[j])

    # mild structure residual
    struct = np.array(
        [1.0 - float(i) / float(max(win, 1)) for i in range(len(coarse))],
        dtype=np.float32,
    )

    zs = _zscore(sent_scores)
    zc = _zscore(cl_scores) if blend > 0 else np.zeros_like(zs)
    # final: primarily sentence align, blend citelink, tiny struct
    final = (1.0 - blend) * zs + blend * zc + 0.05 * struct

    order = np.argsort(-final)
    out: list[dict[str, Any]] = []
    for i in order[:n]:
        c = coarse[int(i)]
        out.append(
            {
                "id": c["id"],
                "title": c.get("title") or "",
                "abstract": c.get("abstract") or "",
                "source": "sent",
                "score": float(final[int(i)]),
                "sent_sim": float(sent_scores[int(i)]),
            }
        )
    return out
