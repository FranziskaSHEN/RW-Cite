#!/usr/bin/env python3
"""Cite-link v3c: train on structure-window negatives (test-time distribution).

Per query: structure Top-W → all gold-in-window as +, sample in-window non-gold as -.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]

from rwcite.runtime.deps import get_domain_cache, get_retriever_service  # noqa: E402
from rwcite.ranker.reference_recommend import normalize_arxiv_id  # noqa: E402
from rwcite.ranker.rr_pool_build import _encode_factory  # noqa: E402
from rwcite.ranker.rr_ranker import rank_universe  # noqa: E402
from rwcite.ranker.rr_ranker_cite_link import (  # noqa: E402
    pack_pair,
    save_citelink_meta,
)
from rwcite.ranker.rr_universe import build_rr_universe  # noqa: E402


def _load_rows(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("task") in (None, "", "bundle"):
                rows.append(r)
    return rows


def _shard_path(cache: Path, shard_id: int, num_shards: int) -> Path:
    return cache.with_name(f"{cache.stem}.shard{shard_id}of{num_shards}{cache.suffix}")


def _cand_extra(graph, node_by, pid: str) -> np.ndarray:
    node = node_by.get(pid)
    indeg = outdeg = 0
    if node is not None and graph is not None:
        try:
            indeg = int(graph.in_degree(node))
            outdeg = int(graph.out_degree(node))
        except Exception:  # noqa: BLE001
            pass
    return np.array([math.log1p(indeg), math.log1p(outdeg)], dtype=np.float32)


def _build_shard(
    rows: list[dict],
    *,
    domain: str,
    window: int,
    neg_per_list: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    graph = get_domain_cache().get(domain).graph
    retriever = get_retriever_service()
    encode = _encode_factory(retriever)
    assert encode is not None

    def search_fn(query: str, limit: int):
        hits = retriever.search(
            scope="domain",
            domain_id=domain,
            query=query,
            offset=0,
            limit=limit,
            sort="relevance",
        )
        return list(hits.get("results") or [])

    rng = random.Random(seed)
    Xs: list[np.ndarray] = []
    ys: list[float] = []
    print(
        f"building window pairs nq={len(rows)} window={window} neg_per_list={neg_per_list}",
        flush=True,
    )
    for qi, s in enumerate(rows):
        title = s.get("title") or ""
        abstract = s.get("abstract") or ""
        excl = normalize_arxiv_id(str(s.get("source_id") or ""))
        gold = {
            normalize_arxiv_id(g["id"])
            for g in (s.get("gold") or [])
            if normalize_arxiv_id(g["id"])
        }
        pack = build_rr_universe(
            title, abstract, graph, search_fn=search_fn, exclude_id=excl
        )
        coarse = rank_universe(
            title=title,
            abstract=abstract,
            universe_pack=pack,
            graph=graph,
            n=min(window, len(pack.get("universe") or [])),
            model=None,
        )
        if len(coarse) < 5:
            continue
        pos_c = [c for c in coarse if c["id"] in gold]
        neg_c = [c for c in coarse if c["id"] not in gold]
        if not pos_c:
            continue
        if len(neg_c) > neg_per_list:
            neg_c = rng.sample(neg_c, neg_per_list)
        ids = [c["id"] for c in pos_c + neg_c]
        q = encode([f"{title}\n{abstract}".strip()[:800]])[0].astype(np.float32)
        qn = float(np.linalg.norm(q))
        if qn > 1e-9:
            q = q / qn
        emb_map = retriever._ensure_domain_embeddings(domain, ids)
        node_by = pack.get("node_by") or {}
        for c in pos_c + neg_c:
            pid = c["id"]
            v = emb_map.get(pid)
            if v is None:
                continue
            vv = np.asarray(v, dtype=np.float32).ravel()
            vn = float(np.linalg.norm(vv))
            if vn > 1e-9:
                vv = vv / vn
            Xs.append(pack_pair(q, vv, _cand_extra(graph, node_by, pid)))
            ys.append(1.0 if pid in gold else 0.0)
        if (qi + 1) % 20 == 0 or qi + 1 == len(rows):
            print(
                f"  [{qi+1}/{len(rows)}] rows={len(ys)} pos={int(sum(ys))}",
                flush=True,
            )
    if not Xs:
        return (
            np.zeros((0, 4098), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
        )
    return np.stack(Xs).astype(np.float32), np.asarray(ys, dtype=np.float32)


def _train_mlp(
    X: np.ndarray,
    y: np.ndarray,
    *,
    out_path: Path,
    hidden: int,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    device: str,
    meta_extra: dict,
) -> None:
    import torch
    import torch.nn as nn

    np_rng = np.random.RandomState(seed)
    mu = X.mean(0)
    sd = np.maximum(X.std(0), 1e-6)
    Xn = (X - mu) / sd
    n = len(y)
    idx = np.arange(n)
    np_rng.shuffle(idx)
    n_val = max(2000, n // 20)
    val_idx, tr_idx = idx[:n_val], idx[n_val:]

    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    model = nn.Sequential(
        nn.Linear(X.shape[1], hidden),
        nn.Tanh(),
        nn.Linear(hidden, 1),
    ).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    pos_rate = float(y[tr_idx].mean()) if len(tr_idx) else 0.1
    pos_weight = torch.tensor([(1.0 - pos_rate) / max(pos_rate, 1e-3)], device=dev)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def run(indices, train: bool):
        model.train(train)
        tl = tn = correct = 0
        for st in range(0, len(indices), batch_size):
            batch = indices[st : st + batch_size]
            xb = torch.from_numpy(Xn[batch]).to(dev)
            yb = torch.from_numpy(y[batch]).to(dev)
            with torch.set_grad_enabled(train):
                logit = model(xb).squeeze(-1)
                loss = loss_fn(logit, yb)
                if train:
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
            tl += float(loss.item()) * len(batch)
            tn += len(batch)
            correct += int(((torch.sigmoid(logit) >= 0.5).float() == yb).sum().item())
        return tl / max(tn, 1), correct / max(tn, 1)

    best = 1e9
    best_state = None
    for ep in range(1, epochs + 1):
        np_rng.shuffle(tr_idx)
        tr_l, tr_a = run(tr_idx, True)
        va_l, va_a = run(val_idx, False)
        print(
            f"epoch {ep}: train_loss={tr_l:.4f} acc={tr_a:.3f} "
            f"val_loss={va_l:.4f} acc={va_a:.3f}",
            flush=True,
        )
        if va_l < best:
            best = va_l
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }

    assert best_state is not None
    model.load_state_dict(best_state)
    w1 = model[0].weight.detach().cpu().numpy().astype(np.float32)
    b1 = model[0].bias.detach().cpu().numpy().astype(np.float32)
    w2 = model[2].weight.detach().cpu().numpy().astype(np.float32).ravel()
    b2 = float(model[2].bias.detach().cpu().numpy().ravel()[0])
    emb_dim = (X.shape[1] - 2) // 4
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        w1=w1,
        b1=b1,
        w2=w2,
        b2=np.array([b2], np.float32),
        mu=mu.astype(np.float32),
        sd=sd.astype(np.float32),
        emb_dim=np.array([emb_dim], np.int32),
        extra_dim=np.array([2], np.int32),
        hidden=np.array([hidden], np.int32),
    )
    save_citelink_meta(
        out_path.parent,
        {
            "kind": "rr_citelink_mlp_v3c_window_negs",
            "model_file": out_path.name,
            "n_rows": int(len(y)),
            "pos_rate": float(y.mean()),
            "best_val_loss": float(best),
            **meta_extra,
        },
    )
    print(f"saved {out_path} val_loss={best:.4f}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="")
    ap.add_argument("--train-jsonl", default="datasets/reference_recommend/train.jsonl")
    ap.add_argument("--max-queries", type=int, default=0)
    ap.add_argument("--window", type=int, default=400)
    ap.add_argument("--neg-per-list", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--cache", default="datasets/rr_pool_ranker/citelink_v3c_pairs.npz")
    ap.add_argument("--out", default="models/rr-pool-ranker-citelink-v3c/model.npz")
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--build-only", action="store_true")
    ap.add_argument("--merge-shards", action="store_true")
    ap.add_argument("--train-only", action="store_true")
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()

    if args.num_shards < 1 or args.shard_id < 0 or args.shard_id >= args.num_shards:
        raise SystemExit(f"bad shard: {args.shard_id}/{args.num_shards}")

    cache = ROOT / args.cache
    out_path = ROOT / args.out

    if args.merge_shards:
        Xs, ys = [], []
        for i in range(args.num_shards):
            sp = _shard_path(cache, i, args.num_shards)
            if not sp.is_file():
                raise SystemExit(f"missing {sp}")
            d = np.load(sp)
            Xs.append(d["X"].astype(np.float32))
            ys.append(d["y"].astype(np.float32))
            print(f"merge {sp.name}: X={Xs[-1].shape}", flush=True)
        X = np.concatenate(Xs, 0)
        y = np.concatenate(ys, 0)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, X=X, y=y)
        print(f"merged → {cache} X={X.shape} pos_rate={float(y.mean()):.4f}", flush=True)
        if args.build_only:
            return 0
        args.train_only = True

    if args.train_only or (
        cache.is_file()
        and not args.rebuild
        and args.num_shards == 1
        and not args.build_only
    ):
        if not cache.is_file():
            raise SystemExit(f"missing cache {cache}")
        d = np.load(cache)
        X, y = d["X"].astype(np.float32), d["y"].astype(np.float32)
        print(f"loaded {cache} X={X.shape} pos={float(y.mean()):.4f}", flush=True)
        _train_mlp(
            X,
            y,
            out_path=out_path,
            hidden=args.hidden,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            seed=args.seed,
            device=args.device,
            meta_extra={
                "window": args.window,
                "neg_per_list": args.neg_per_list,
                "max_queries": args.max_queries,
            },
        )
        return 0

    rng = random.Random(args.seed)
    rows = _load_rows(ROOT / args.train_jsonl)
    rng.shuffle(rows)
    if args.max_queries and args.max_queries > 0:
        rows = rows[: args.max_queries]
    if args.num_shards > 1:
        rows = [r for i, r in enumerate(rows) if i % args.num_shards == args.shard_id]
        out_cache = _shard_path(cache, args.shard_id, args.num_shards)
        print(
            f"shard {args.shard_id}/{args.num_shards} nq={len(rows)} → {out_cache.name}",
            flush=True,
        )
    else:
        out_cache = cache

    if out_cache.is_file() and not args.rebuild:
        print(f"exists {out_cache} (use --rebuild)", flush=True)
        if args.build_only or args.num_shards > 1:
            return 0

    X, y = _build_shard(
        rows,
        domain=args.domain,
        window=args.window,
        neg_per_list=args.neg_per_list,
        seed=args.seed + args.shard_id * 17,
    )
    out_cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_cache, X=X, y=y)
    print(f"cached → {out_cache} X={X.shape}", flush=True)

    if args.build_only or args.num_shards > 1:
        return 0

    _train_mlp(
        X,
        y,
        out_path=out_path,
        hidden=args.hidden,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        device=args.device,
        meta_extra={
            "window": args.window,
            "neg_per_list": args.neg_per_list,
            "max_queries": args.max_queries,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
