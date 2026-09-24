"""MLP pool ranker on [struct feats | q_emb | c_emb | q*c | |q-c|]."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from rwcite.ranker.rr_ranker import FEATURE_NAMES, extract_features


def pack_vector(
    feat10: np.ndarray, q_emb: np.ndarray, c_emb: np.ndarray
) -> np.ndarray:
    q = q_emb.astype(np.float32).ravel()
    c = c_emb.astype(np.float32).ravel()
    return np.concatenate(
        [feat10.astype(np.float32), q, c, q * c, np.abs(q - c)], axis=0
    )


class MLPRanker:
    """2-layer MLP with RankNet-style scoring."""

    def __init__(
        self,
        w1: np.ndarray,
        b1: np.ndarray,
        w2: np.ndarray,
        b2: float,
        mu: np.ndarray,
        sd: np.ndarray,
        hidden: int,
        in_dim: int,
    ):
        self.w1 = w1
        self.b1 = b1
        self.w2 = w2
        self.b2 = float(b2)
        self.mu = mu
        self.sd = sd
        self.hidden = hidden
        self.in_dim = in_dim

    def score_vec(self, x: np.ndarray) -> float:
        xn = (x - self.mu) / self.sd
        h = np.tanh(xn @ self.w1 + self.b1)
        return float(h @ self.w2 + self.b2)

    def score_batch(self, X: np.ndarray) -> np.ndarray:
        xn = (X - self.mu) / self.sd
        h = np.tanh(xn @ self.w1 + self.b1)
        return h @ self.w2 + self.b2

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            w1=self.w1,
            b1=self.b1,
            w2=self.w2,
            b2=np.array([self.b2]),
            mu=self.mu,
            sd=self.sd,
            hidden=np.array([self.hidden]),
            in_dim=np.array([self.in_dim]),
            kind=np.array(["mlp_v1"], dtype=object),
        )
        (path.parent / "meta.json").write_text(
            json.dumps(
                {
                    "kind": "mlp_pool_ranker_v1",
                    "hidden": self.hidden,
                    "in_dim": self.in_dim,
                    "struct_features": FEATURE_NAMES,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "MLPRanker":
        d = np.load(path, allow_pickle=True)
        return cls(
            w1=d["w1"],
            b1=d["b1"],
            w2=d["w2"],
            b2=float(d["b2"].reshape(-1)[0]),
            mu=d["mu"],
            sd=d["sd"],
            hidden=int(d["hidden"].reshape(-1)[0]),
            in_dim=int(d["in_dim"].reshape(-1)[0]),
        )


def train_mlp_ranknet(
    X: np.ndarray,
    y: np.ndarray,
    qids: np.ndarray,
    *,
    hidden: int = 128,
    epochs: int = 40,
    lr: float = 0.05,
    pairs_per_query: int = 200,
    seed: int = 0,
) -> MLPRanker:
    """GPU RankNet (torch). Falls back to a tiny CPU loop only if torch missing."""
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        torch = None  # type: ignore

    rng = np.random.default_rng(seed)
    mu = X.mean(0)
    sd = X.std(0) + 1e-6
    Xn = ((X - mu) / sd).astype(np.float32)
    in_dim = X.shape[1]

    by_q: dict[int, tuple[list[int], list[int]]] = {}
    for i, q in enumerate(qids):
        q = int(q)
        if q not in by_q:
            by_q[q] = ([], [])
        if y[i] == 1:
            by_q[q][0].append(i)
        else:
            by_q[q][1].append(i)
    q_list = [q for q, (p, n) in by_q.items() if p and n]

    # Materialize pair index arrays once
    pos_idx: list[int] = []
    neg_idx: list[int] = []
    for q in q_list:
        pos, neg = by_q[q]
        n_pairs = min(pairs_per_query, max(1, len(pos) * min(len(neg), 50)))
        for _ in range(n_pairs):
            pos_idx.append(int(rng.choice(pos)))
            neg_idx.append(int(rng.choice(neg)))
    pos_idx_a = np.asarray(pos_idx, dtype=np.int64)
    neg_idx_a = np.asarray(neg_idx, dtype=np.int64)
    print(f"ranknet pairs={len(pos_idx_a)} in_dim={in_dim} hidden={hidden}", flush=True)

    if torch is not None and torch.cuda.is_available():
        device = torch.device("cuda")
        Xt = torch.from_numpy(Xn).to(device)
        model = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        ).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        bce = nn.BCEWithLogitsLoss()
        pos_t = torch.from_numpy(pos_idx_a).to(device)
        neg_t = torch.from_numpy(neg_idx_a).to(device)
        ones = torch.ones(len(pos_idx_a), device=device)
        bs = 4096
        for epoch in range(epochs):
            # reshuffle pairs each epoch
            perm = torch.randperm(len(pos_idx_a), device=device)
            total = 0.0
            nstep = 0
            model.train()
            for st in range(0, len(perm), bs):
                idx = perm[st : st + bs]
                sp = model(Xt[pos_t[idx]]).squeeze(-1)
                sn = model(Xt[neg_t[idx]]).squeeze(-1)
                loss = bce(sp - sn, ones[: len(idx)])
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += float(loss.item())
                nstep += 1
            if epoch % 5 == 0 or epoch == epochs - 1:
                print(
                    f"epoch {epoch:3d} loss={total/max(1,nstep):.4f}",
                    flush=True,
                )
        # export weights
        lin1 = model[0]
        lin2 = model[2]
        w1 = lin1.weight.detach().cpu().numpy().T.astype(np.float64)  # (in, hidden)
        b1 = lin1.bias.detach().cpu().numpy().astype(np.float64)
        w2 = lin2.weight.detach().cpu().numpy().ravel().astype(np.float64)
        b2 = float(lin2.bias.detach().cpu().numpy().ravel()[0])
        return MLPRanker(
            w1=w1, b1=b1, w2=w2, b2=b2, mu=mu, sd=sd, hidden=hidden, in_dim=in_dim
        )

    # CPU fallback (slow): few epochs only
    print("torch/cuda unavailable; using reduced CPU RankNet", flush=True)
    w1 = rng.normal(0, 0.05, size=(in_dim, hidden))
    b1 = np.zeros(hidden)
    w2 = rng.normal(0, 0.05, size=(hidden,))
    b2 = 0.0
    epochs = min(epochs, 15)
    for epoch in range(epochs):
        total_loss = 0.0
        for i, j in zip(pos_idx_a, neg_idx_a):
            xi, xj = Xn[i], Xn[j]
            hi = np.tanh(xi @ w1 + b1)
            hj = np.tanh(xj @ w1 + b1)
            si = float(hi @ w2 + b2)
            sj = float(hj @ w2 + b2)
            p = 1.0 / (1.0 + np.exp(-np.clip(si - sj, -30, 30)))
            g = p - 1.0
            total_loss += float(-np.log(max(p, 1e-9)))
            dhi = g * w2 * (1 - hi**2)
            dhj = (-g) * w2 * (1 - hj**2)
            w2 -= lr * (g * hi - g * hj) / len(pos_idx_a)
            b2 -= lr * 0.0
            w1 -= lr * (np.outer(xi, dhi) + np.outer(xj, dhj)) / len(pos_idx_a)
            b1 -= lr * (dhi + dhj) / len(pos_idx_a)
        if epoch % 5 == 0 or epoch == epochs - 1:
            print(
                f"epoch {epoch:3d} loss={total_loss/max(1,len(pos_idx_a)):.4f}",
                flush=True,
            )
    return MLPRanker(
        w1=w1, b1=b1, w2=w2, b2=b2, mu=mu, sd=sd, hidden=hidden, in_dim=in_dim
    )


_MLP_CACHE: MLPRanker | None = None
_MLP_PATH: str | None = None


def load_mlp_ranker(path: str | None = None) -> MLPRanker | None:
    global _MLP_CACHE, _MLP_PATH
    import os

    p = path or (os.environ.get("RR_RANKER_MLP_PATH") or "").strip()
    if not p:
        return None
    candidates = [Path(p)]
    if not Path(p).is_absolute():
        repo = Path(__file__).resolve().parents[3]
        candidates.append(repo / p)
        candidates.append(Path.cwd() / p)
    for c in candidates:
        if c.is_file():
            key = str(c.resolve())
            if _MLP_CACHE is not None and _MLP_PATH == key:
                return _MLP_CACHE
            _MLP_CACHE = MLPRanker.load(c)
            _MLP_PATH = key
            return _MLP_CACHE
    return None
