"""Cite-link pool reranker: P(cite | emb_q, emb_c) from graph edges.

Activate via RR_RANKER_CITELINK_PATH=/path/to/model.npz
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Callable

import numpy as np

from rwcite.ranker.rr_ranker import rank_universe


def rr_citelink_params() -> dict[str, Any]:
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
        "path": (os.environ.get("RR_RANKER_CITELINK_PATH") or "").strip(),
        "window": _i("RR_CITELINK_WINDOW", 400),
        "struct_blend": _f("RR_CITELINK_STRUCT_BLEND", 0.35),
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


def pack_pair(u: np.ndarray, v: np.ndarray, extra: np.ndarray | None = None) -> np.ndarray:
    u = u.astype(np.float32).ravel()
    v = v.astype(np.float32).ravel()
    parts = [u, v, u * v, np.abs(u - v)]
    if extra is not None:
        parts.append(extra.astype(np.float32).ravel())
    return np.concatenate(parts, axis=0)


class CiteLinkMLP:
    def __init__(
        self,
        w1: np.ndarray,
        b1: np.ndarray,
        w2: np.ndarray,
        b2: float,
        mu: np.ndarray,
        sd: np.ndarray,
        *,
        emb_dim: int,
        extra_dim: int,
        hidden: int,
    ):
        self.w1 = w1.astype(np.float32)
        self.b1 = b1.astype(np.float32)
        self.w2 = w2.astype(np.float32).ravel()
        self.b2 = float(b2)
        self.mu = mu.astype(np.float32)
        self.sd = np.maximum(sd.astype(np.float32), 1e-6)
        self.emb_dim = int(emb_dim)
        self.extra_dim = int(extra_dim)
        self.hidden = int(hidden)

    def score_vec(self, x: np.ndarray) -> float:
        z = (x.astype(np.float32) - self.mu) / self.sd
        h = np.tanh(self.w1 @ z + self.b1)
        return float(self.w2 @ h + self.b2)

    def score_batch(self, X: np.ndarray) -> np.ndarray:
        z = (X.astype(np.float32) - self.mu) / self.sd
        h = np.tanh(z @ self.w1.T + self.b1)
        return (h @ self.w2 + self.b2).astype(np.float32)


_CACHE: CiteLinkMLP | None = None
_CACHE_PATH: str | None = None


def load_citelink_ranker(path: str | None = None) -> CiteLinkMLP | None:
    global _CACHE, _CACHE_PATH
    p = path if path is not None else rr_citelink_params()["path"]
    resolved = _resolve_path(p) if p else None
    if resolved is None:
        return None
    key = str(resolved)
    if _CACHE is not None and _CACHE_PATH == key:
        return _CACHE
    data = np.load(resolved, allow_pickle=True)

    def _scalar(name: str, default: int | float | None = None):
        if name not in data.files:
            if default is None:
                raise KeyError(name)
            return default
        return data[name].reshape(-1)[0]

    model = CiteLinkMLP(
        data["w1"],
        data["b1"],
        data["w2"],
        float(_scalar("b2")),
        data["mu"],
        data["sd"],
        emb_dim=int(_scalar("emb_dim")),
        extra_dim=int(_scalar("extra_dim", 0)),
        hidden=int(_scalar("hidden")),
    )
    _CACHE = model
    _CACHE_PATH = key
    return model


def _cand_extra(graph: Any, node_by: dict[str, Any], pid: str) -> np.ndarray:
    node = node_by.get(pid)
    indeg = outdeg = 0
    if node is not None and graph is not None:
        try:
            indeg = int(graph.in_degree(node))
            outdeg = int(graph.out_degree(node))
        except Exception:  # noqa: BLE001
            pass
    return np.array(
        [math.log1p(indeg), math.log1p(outdeg)],
        dtype=np.float32,
    )


def score_cands_citelink(
    *,
    cands: list[dict[str, Any]],
    graph: Any,
    model: CiteLinkMLP,
    q_emb: np.ndarray,
    cand_embs: dict[str, np.ndarray],
    universe_pack: dict[str, Any] | None = None,
) -> list[tuple[float, dict[str, Any]]]:
    """Score an arbitrary candidate list with cite-link MLP (no struct prefilter)."""
    if not cands:
        return []
    node_by = (universe_pack or {}).get("node_by") or {}
    q = np.asarray(q_emb, dtype=np.float32).ravel()
    qn = float(np.linalg.norm(q))
    if qn > 1e-9:
        q = q / qn
    rows = []
    keep: list[dict[str, Any]] = []
    for c in cands:
        pid = c["id"]
        ce = cand_embs.get(pid)
        if ce is None:
            continue
        v = np.asarray(ce, dtype=np.float32).ravel()
        vn = float(np.linalg.norm(v))
        if vn > 1e-9:
            v = v / vn
        extra = _cand_extra(graph, node_by, pid) if model.extra_dim else None
        if model.extra_dim and (extra is None or extra.size != model.extra_dim):
            extra = np.zeros((model.extra_dim,), dtype=np.float32)
        rows.append(pack_pair(q, v, extra))
        keep.append(c)
    if not rows:
        return [(0.0, c) for c in cands]
    logits = model.score_batch(np.stack(rows))
    return [(float(logits[i]), keep[i]) for i in range(len(keep))]


def citelink_rank_full_universe(
    *,
    universe_pack: dict[str, Any],
    graph: Any,
    model: CiteLinkMLP,
    q_emb: np.ndarray,
    cand_embs: dict[str, np.ndarray],
    n: int = 400,
) -> list[dict[str, Any]]:
    """Rank full universe by cite-link only (R1 cheap coarse; no struct window)."""
    universe = list(universe_pack.get("universe") or [])
    scored = score_cands_citelink(
        cands=universe,
        graph=graph,
        model=model,
        q_emb=q_emb,
        cand_embs=cand_embs,
        universe_pack=universe_pack,
    )
    scored.sort(key=lambda x: -x[0])
    out: list[dict[str, Any]] = []
    for sc, c in scored[:n]:
        out.append(
            {
                "id": c["id"],
                "title": c.get("title") or "",
                "abstract": c.get("abstract") or "",
                "source": "citelink_full",
                "score": float(sc),
            }
        )
    return out


def rank_universe_citelink(
    *,
    title: str,
    abstract: str,
    universe_pack: dict[str, Any],
    graph: Any,
    model: CiteLinkMLP,
    q_emb: np.ndarray,
    cand_embs: dict[str, np.ndarray],
    n: int = 30,
    window: int | None = None,
    struct_blend: float | None = None,
) -> list[dict[str, Any]]:
    params = rr_citelink_params()
    win = int(window if window is not None else params["window"])
    blend = float(
        struct_blend if struct_blend is not None else params["struct_blend"]
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
    scored = score_cands_citelink(
        cands=coarse,
        graph=graph,
        model=model,
        q_emb=q_emb,
        cand_embs=cand_embs,
        universe_pack=universe_pack,
    )
    if not scored:
        return coarse[:n]
    # mild residual toward coarse order (known weak but calibrated baseline)
    logits = np.array([s for s, _ in scored], dtype=np.float32)
    if blend > 0:
        id_to_ri = {c["id"]: ri for ri, c in enumerate(coarse)}
        for i, (_, c) in enumerate(scored):
            ri = id_to_ri.get(c["id"], i)
            logits[i] = logits[i] + blend * (1.0 - float(ri) / max(win, 1))
    order = np.argsort(-logits)
    out: list[dict[str, Any]] = []
    for i in order[:n]:
        c = scored[int(i)][1]
        out.append(
            {
                "id": c["id"],
                "title": c.get("title") or "",
                "abstract": c.get("abstract") or "",
                "source": "citelink",
                "score": float(logits[int(i)]),
            }
        )
    return out


def save_citelink_meta(out_dir: Path, extra: dict | None = None) -> None:
    meta = {"kind": "rr_citelink_mlp_v3", **(extra or {})}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
