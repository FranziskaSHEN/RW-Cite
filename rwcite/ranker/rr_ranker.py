"""Independent RR pool ranker: structural/lexical features (+ optional embedding MLP).

Scores a universe and returns Top-N for Stage A. Separate from the RR LLM adapter.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from rwcite.ranker.reference_recommend import normalize_arxiv_id
from rwcite.ranker.rr_universe import jaccard_query_cand

FEATURE_NAMES = [
    "support_n",
    "log_support",
    "lex",
    "cocite_n",
    "log_indeg",
    "log_outdeg",
    "in_dense",
    "dense_bonus",
    "has_support",
    "emb_sim",
]


def rr_ranker_params() -> dict[str, Any]:
    def _i(name: str, default: int) -> int:
        raw = (os.environ.get(name) or "").strip()
        if not raw:
            return default
        try:
            return max(1, int(raw))
        except ValueError:
            return default

    path = (os.environ.get("RR_RANKER_PATH") or "").strip()
    if not path:
        # default relative to repo models/
        path = "models/rr-pool-ranker-v1/model.npz"
    return {
        "n": _i("RR_CAND_N", 30),
        "path": path,
        "direct_top10": (os.environ.get("RR_RANKER_DIRECT_TOP10") or "").strip()
        in ("1", "true", "yes"),
    }


def _indeg_outdeg(
    graph: Any,
    node_by: dict[str, Any],
    pid: str,
    *,
    mask_citers: set[str] | None = None,
) -> tuple[int, int]:
    node = node_by.get(pid)
    if node is None or graph is None:
        return 0, 0
    try:
        outdeg = int(graph.out_degree(node))
        if not mask_citers:
            return int(graph.in_degree(node)), outdeg
        indeg = 0
        for u in graph.predecessors(node):
            if normalize_arxiv_id(str(u)) in mask_citers:
                continue
            indeg += 1
        return indeg, outdeg
    except Exception:  # noqa: BLE001
        return 0, 0


def _cocite(
    graph: Any,
    node_by: dict[str, Any],
    pid: str,
    focused: set[str],
    *,
    mask_citers: set[str] | None = None,
) -> int:
    node = node_by.get(pid)
    if node is None or graph is None or not focused:
        return 0
    n = 0
    try:
        for u in graph.predecessors(node):
            uid = normalize_arxiv_id(str(u))
            if mask_citers and uid in mask_citers:
                continue
            if uid in focused:
                n += 1
                if n >= 15:
                    break
    except Exception:  # noqa: BLE001
        return n
    return n


def extract_features(
    *,
    title: str,
    abstract: str,
    cand: dict[str, Any],
    support: dict[str, float],
    dense_rank: dict[str, int],
    focused_hubs: set[str],
    graph: Any,
    node_by: dict[str, Any],
    emb_sim: float = 0.0,
    max_support: float = 1.0,
    query_id: str = "",
    mask_citers: set[str] | None = None,
) -> np.ndarray:
    pid = cand["id"]
    s = float(support.get(pid, 0.0))
    max_s = max(max_support, 1e-6)
    qtext = f"{title}\n{abstract}".strip()
    lex = jaccard_query_cand(qtext, cand.get("title") or "", cand.get("abstract") or "")
    if mask_citers is None and query_id:
        from rwcite.ranker.rr_graph_mask import masked_citer_ids

        mask_citers = masked_citer_ids(query_id=query_id)
    elif mask_citers is None:
        from rwcite.ranker.rr_graph_mask import masked_citer_ids

        mask_citers = masked_citer_ids(query_id="")
    indeg, outdeg = _indeg_outdeg(graph, node_by, pid, mask_citers=mask_citers)
    coc = _cocite(graph, node_by, pid, focused_hubs, mask_citers=mask_citers)
    dr = dense_rank.get(pid, -1)
    vec = np.array(
        [
            s / max_s,
            math.log1p(s),
            lex,
            min(coc, 15) / 15.0,
            math.log1p(indeg),
            math.log1p(outdeg),
            1.0 if dr >= 0 else 0.0,
            (1.0 / (1.0 + dr / 25.0)) if dr >= 0 else 0.0,
            1.0 if s > 0 else 0.0,
            float(emb_sim),
        ],
        dtype=np.float32,
    )
    return vec


@dataclass
class RankerModel:
    w: np.ndarray
    b: float
    mu: np.ndarray
    sd: np.ndarray
    feature_names: list[str]

    def score(self, x: np.ndarray) -> float:
        xn = (x - self.mu) / self.sd
        return float(xn @ self.w + self.b)

    def score_batch(self, X: np.ndarray) -> np.ndarray:
        xn = (X - self.mu) / self.sd
        return xn @ self.w + self.b

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            w=self.w,
            b=np.array([self.b], dtype=np.float64),
            mu=self.mu,
            sd=self.sd,
            feature_names=np.array(self.feature_names, dtype=object),
        )
        meta = {
            "feature_names": self.feature_names,
            "kind": "logistic_pool_ranker_v1",
        }
        (path.parent / "meta.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: str | Path) -> "RankerModel":
        data = np.load(path, allow_pickle=True)
        names = [str(x) for x in data["feature_names"].tolist()]
        return cls(
            w=data["w"].astype(np.float64),
            b=float(data["b"].reshape(-1)[0]),
            mu=data["mu"].astype(np.float64),
            sd=data["sd"].astype(np.float64),
            feature_names=names,
        )


_MODEL_CACHE: RankerModel | None = None
_MODEL_PATH: str | None = None


def load_ranker(path: str | None = None) -> RankerModel | None:
    global _MODEL_CACHE, _MODEL_PATH
    params = rr_ranker_params()
    p = path or params["path"]
    # resolve relative to CWD / common roots
    candidates = [Path(p)]
    if not Path(p).is_absolute():
        here = Path(__file__).resolve()
        repo = here.parents[2]  # RW-Cite root
        candidates.append(repo / p)
        candidates.append(Path.cwd() / p)
    for c in candidates:
        if c.is_file():
            key = str(c.resolve())
            if _MODEL_CACHE is not None and _MODEL_PATH == key:
                return _MODEL_CACHE
            _MODEL_CACHE = RankerModel.load(c)
            _MODEL_PATH = key
            return _MODEL_CACHE
    return None


def heuristic_score(feat: np.ndarray) -> float:
    """Fallback when no trained model: structure-first hybrid."""
    # indices match FEATURE_NAMES
    has_s, s_n, lex, coc, db = feat[8], feat[0], feat[2], feat[3], feat[7]
    return float(has_s) * 10.0 + 0.45 * s_n + 0.30 * lex + 0.15 * coc + 0.10 * db


def rank_universe(
    *,
    title: str,
    abstract: str,
    universe_pack: dict[str, Any],
    graph: Any,
    n: int | None = None,
    emb_sims: dict[str, float] | None = None,
    model: RankerModel | None = None,
    query_id: str = "",
    mask_citers: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Score universe and return Top-N candidate dicts for Stage A."""
    params = rr_ranker_params()
    n = int(n if n is not None else params["n"])
    universe = universe_pack.get("universe") or []
    if not universe:
        return []
    support = universe_pack.get("support") or {}
    dense_rank = universe_pack.get("dense_rank") or {}
    focused = universe_pack.get("focused_hubs") or set()
    node_by = universe_pack.get("node_by") or {}
    max_s = max([float(v) for v in support.values()], default=1.0)
    emb_sims = emb_sims or {}
    qid = normalize_arxiv_id(query_id) if query_id else ""
    if mask_citers is None:
        from rwcite.ranker.rr_graph_mask import masked_citer_ids

        mask_citers = masked_citer_ids(query_id=qid)

    if model is None:
        model = load_ranker()

    scored: list[tuple[float, dict[str, Any]]] = []
    for cand in universe:
        feat = extract_features(
            title=title,
            abstract=abstract,
            cand=cand,
            support=support,
            dense_rank=dense_rank,
            focused_hubs=focused,
            graph=graph,
            node_by=node_by,
            emb_sim=float(emb_sims.get(cand["id"], 0.0)),
            max_support=max_s,
            query_id=qid,
            mask_citers=mask_citers,
        )
        if model is not None:
            sc = model.score(feat)
        else:
            sc = heuristic_score(feat)
        out = {
            "id": cand["id"],
            "title": cand.get("title") or "",
            "abstract": cand.get("abstract") or "",
            "source": cand.get("source") or "proxy",
            "score": sc,
        }
        scored.append((sc, out))
    scored.sort(key=lambda x: -x[0])
    return [c for _, c in scored[:n]]
