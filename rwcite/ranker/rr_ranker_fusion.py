"""v3d listwise fusion: LightGBM on [citelink_logit, struct_rank, emb_sim, ...].

Activate via RR_RANKER_FUSION_PATH=/path/to/model.txt
Requires RR_RANKER_CITELINK_PATH for the frozen cite-link scorer.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from rwcite.ranker.rr_ranker import rank_universe
from rwcite.ranker.rr_ranker_cite_link import (
    CiteLinkMLP,
    _cand_extra,
    pack_pair,
    rr_citelink_params,
)

FUSION_FEATURE_NAMES = [
    "citelink_logit",
    "struct_rank_n",
    "emb_sim",
    "log_indeg",
    "log_outdeg",
    "citelink_x_emb",
]


def rr_fusion_params() -> dict[str, Any]:
    def _i(name: str, default: int) -> int:
        raw = (os.environ.get(name) or "").strip()
        if not raw:
            return default
        try:
            return max(1, int(raw))
        except ValueError:
            return default

    return {
        "path": (os.environ.get("RR_RANKER_FUSION_PATH") or "").strip(),
        "window": _i("RR_FUSION_WINDOW", 400),
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


_FUSION_CACHE: Any = None
_FUSION_PATH: str | None = None


def load_fusion_ranker(path: str | None = None):
    global _FUSION_CACHE, _FUSION_PATH
    p = path if path is not None else rr_fusion_params()["path"]
    resolved = _resolve_path(p) if p else None
    if resolved is None:
        return None
    key = str(resolved)
    if _FUSION_CACHE is not None and _FUSION_PATH == key:
        return _FUSION_CACHE
    import lightgbm as lgb

    booster = lgb.Booster(model_file=str(resolved))
    _FUSION_CACHE = booster
    _FUSION_PATH = key
    return booster


def extract_fusion_features(
    *,
    citelink_logit: float,
    struct_rank: int,
    window: int,
    emb_sim: float,
    log_indeg: float,
    log_outdeg: float,
) -> np.ndarray:
    win = max(1, int(window))
    return np.array(
        [
            float(citelink_logit),
            float(struct_rank) / float(win),
            float(emb_sim),
            float(log_indeg),
            float(log_outdeg),
            float(citelink_logit) * float(emb_sim),
        ],
        dtype=np.float32,
    )


def score_window_citelink(
    *,
    citelink: CiteLinkMLP,
    q_emb: np.ndarray,
    coarse: list[dict[str, Any]],
    cand_embs: dict[str, np.ndarray],
    graph: Any,
    node_by: dict[str, Any],
) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray]:
    """Return (kept_cands, logits, emb_sims) aligned."""
    q = np.asarray(q_emb, dtype=np.float32).ravel()
    qn = float(np.linalg.norm(q))
    if qn > 1e-9:
        q = q / qn
    kept: list[dict[str, Any]] = []
    rows: list[np.ndarray] = []
    sims: list[float] = []
    for c in coarse:
        pid = c["id"]
        ce = cand_embs.get(pid)
        if ce is None:
            continue
        v = np.asarray(ce, dtype=np.float32).ravel()
        vn = float(np.linalg.norm(v))
        if vn > 1e-9:
            v = v / vn
        extra = _cand_extra(graph, node_by, pid) if citelink.extra_dim else None
        if citelink.extra_dim and (extra is None or extra.size != citelink.extra_dim):
            extra = np.zeros((citelink.extra_dim,), dtype=np.float32)
        rows.append(pack_pair(q, v, extra))
        kept.append(c)
        sims.append(float(np.dot(q, v)))
    if not rows:
        return [], np.zeros((0,), np.float32), np.zeros((0,), np.float32)
    logits = citelink.score_batch(np.stack(rows))
    return kept, logits, np.asarray(sims, dtype=np.float32)


def rank_universe_fusion(
    *,
    title: str,
    abstract: str,
    universe_pack: dict[str, Any],
    graph: Any,
    citelink: CiteLinkMLP,
    fusion_booster: Any,
    q_emb: np.ndarray,
    cand_embs: dict[str, np.ndarray],
    n: int = 30,
    window: int | None = None,
) -> list[dict[str, Any]]:
    params = rr_fusion_params()
    win = int(window if window is not None else params["window"])
    # align citelink window env if unset
    cl_win = int(rr_citelink_params()["window"])
    win = min(win, cl_win) if window is None else win
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
    node_by = universe_pack.get("node_by") or {}
    kept, logits, sims = score_window_citelink(
        citelink=citelink,
        q_emb=q_emb,
        coarse=coarse,
        cand_embs=cand_embs,
        graph=graph,
        node_by=node_by,
    )
    if not kept:
        return []
    # map struct rank from original coarse order
    rank_by = {c["id"]: i for i, c in enumerate(coarse)}
    feats = []
    for i, c in enumerate(kept):
        extra = _cand_extra(graph, node_by, c["id"])
        feats.append(
            extract_fusion_features(
                citelink_logit=float(logits[i]),
                struct_rank=int(rank_by.get(c["id"], i)),
                window=win,
                emb_sim=float(sims[i]),
                log_indeg=float(extra[0]),
                log_outdeg=float(extra[1]),
            )
        )
    X = np.stack(feats)
    scores = fusion_booster.predict(X)
    order = np.argsort(-scores)
    out: list[dict[str, Any]] = []
    for i in order[:n]:
        c = kept[int(i)]
        out.append(
            {
                "id": c["id"],
                "title": c.get("title") or "",
                "abstract": c.get("abstract") or "",
                "source": "fusion",
                "score": float(scores[int(i)]),
            }
        )
    return out


def save_fusion_meta(out_dir: Path, extra: dict | None = None) -> None:
    meta = {
        "kind": "rr_citelink_fusion_lambdarank_v3d",
        "feature_names": FUSION_FEATURE_NAMES,
        **(extra or {}),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
