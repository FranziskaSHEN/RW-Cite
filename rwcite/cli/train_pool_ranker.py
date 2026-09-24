#!/usr/bin/env python3
"""Train domain-local structural shortlist ``RankerModel`` (v1 linear family).

Standard RW pipeline (default): weighted logistic BCE on 10-d features with
``emb_sim`` filled at dump time (AS v1 recipe), then inference via
``RR_STRUCT_COARSE=linear`` (emb_sims=None) for CE shortlist.

Also supports legacy pairwise RankNet fit (``--loss ranknet``).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]

from rwcite.graph.domain_paths import (  # noqa: E402
    domain_data_paths,
    working_dir_for_domain,
)
from rwcite.ranker.reference_recommend import normalize_arxiv_id  # noqa: E402
from rwcite.ranker.rr_ranker import (  # noqa: E402
    FEATURE_NAMES,
    RankerModel,
    extract_features,
    heuristic_score,
)
from rwcite.ranker.rr_universe import build_rr_universe  # noqa: E402
from rwcite.runtime.config import load_app_settings  # noqa: E402
from rwcite.runtime.deps import get_domain_cache, get_retriever_service  # noqa: E402


def _load_sources(path: Path) -> list[dict]:
    """One row per source id (the jsonl carries each bundle twice)."""
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


def _gold_ids(graph, row: dict) -> set[str]:
    """In-graph references of the source: the shortlist's retrieval targets."""
    sid = normalize_arxiv_id(str(row.get("source_id") or ""))
    gold = {normalize_arxiv_id(g["id"]) for g in row.get("gold") or [] if g.get("id")}
    if graph is not None and sid in graph:
        gold |= {normalize_arxiv_id(str(t)) for t in graph.successors(sid)}
    gold.discard(sid)
    return {g for g in gold if g}


def _make_encode(retriever):
    retriever._load_embedder()
    import torch

    tok, emb = retriever._tokenizer, retriever._embedder
    device = next(emb.parameters()).device
    emb.eval()

    def encode(texts: list[str]) -> np.ndarray:
        outs: list[np.ndarray] = []
        for i in range(0, len(texts), 32):
            batch = texts[i : i + 32]
            inputs = tok(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                v = emb(**inputs).last_hidden_state[:, 0, :].float()
                v = torch.nn.functional.normalize(v, dim=-1)
            outs.append(v.cpu().numpy())
        return np.concatenate(outs, 0)

    return encode


def _stage_dump(args: argparse.Namespace) -> None:
    root = load_app_settings().data_root
    wd = working_dir_for_domain(args.domain, root)
    jsonl = Path(args.train_jsonl) if args.train_jsonl else Path(
        domain_data_paths(wd)["jsonl_dir"]
    ) / "train.jsonl"
    if not jsonl.is_absolute():
        jsonl = root / jsonl
    rows = _load_sources(jsonl)
    if args.max_sources > 0:
        rows = rows[: args.max_sources]
    rows = [r for i, r in enumerate(rows) if i % args.num_shards == args.shard_id]
    print(
        f"shard {args.shard_id}/{args.num_shards}: {len(rows)} sources from {jsonl} "
        f"with_emb={int(args.with_emb)}",
        flush=True,
    )

    graph = get_domain_cache().get(args.domain).graph
    retriever = get_retriever_service()
    encode = _make_encode(retriever) if args.with_emb else None

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

    feats: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    qids: list[np.ndarray] = []
    per_query: list[dict] = []
    for qi, row in enumerate(rows):
        sid = normalize_arxiv_id(str(row.get("source_id") or ""))
        title = row.get("title") or ""
        abstract = row.get("abstract") or ""
        gold = _gold_ids(graph, row)
        pack = build_rr_universe(
            title,
            abstract,
            graph,
            search_fn=search_fn,
            exclude_id=sid,
            multi_query=True,
            prf=True,
        )
        universe = pack.get("universe") or []
        if not universe or not gold:
            continue
        support = pack.get("support") or {}
        max_s = max([float(v) for v in support.values()], default=1.0)
        dense_rank = pack.get("dense_rank") or {}
        focused = pack.get("focused_hubs") or set()
        node_by = pack.get("node_by") or {}

        emb_sims = np.zeros(len(universe), dtype=np.float32)
        if encode is not None:
            qv = encode([f"{title}\n{abstract}".strip()[:800]])[0]
            texts = [
                f"{c.get('title') or ''}\n{c.get('abstract') or ''}"[:800]
                for c in universe
            ]
            cvs = encode(texts)
            emb_sims = (cvs @ qv).astype(np.float32)

        X = np.zeros((len(universe), len(FEATURE_NAMES)), dtype=np.float32)
        y = np.zeros(len(universe), dtype=np.int8)
        for i, cand in enumerate(universe):
            X[i] = extract_features(
                title=title,
                abstract=abstract,
                cand=cand,
                support=support,
                dense_rank=dense_rank,
                focused_hubs=focused,
                graph=graph,
                node_by=node_by,
                emb_sim=float(emb_sims[i]),
                max_support=max_s,
                query_id=sid,
            )
            if normalize_arxiv_id(cand["id"]) in gold:
                y[i] = 1
        gid = args.shard_id * 1_000_000 + qi
        feats.append(X)
        labels.append(y)
        qids.append(np.full(len(universe), gid, dtype=np.int64))
        per_query.append(
            {
                "source_id": sid,
                "qid": gid,
                "n_universe": len(universe),
                "n_gold_total": len(gold),
                "n_gold_in_universe": int(y.sum()),
                "mean_emb_sim": float(emb_sims.mean()) if emb_sims.size else 0.0,
            }
        )
        if (qi + 1) % 25 == 0:
            print(
                f"  [{qi + 1}/{len(rows)}] {sid} |U|={len(universe)} "
                f"gold_in_U={int(y.sum())}/{len(gold)}",
                flush=True,
            )

    if not feats:
        raise SystemExit("no usable training queries")
    out = Path(args.out_dir)
    if not out.is_absolute():
        out = root / out
    out.mkdir(parents=True, exist_ok=True)
    dst = out / f"features_shard{args.shard_id}of{args.num_shards}.npz"
    np.savez_compressed(
        dst,
        X=np.concatenate(feats, 0),
        y=np.concatenate(labels, 0),
        qid=np.concatenate(qids, 0),
        feature_names=np.array(FEATURE_NAMES, dtype=object),
        with_emb=np.array([1 if args.with_emb else 0]),
    )
    (out / f"queries_shard{args.shard_id}of{args.num_shards}.json").write_text(
        json.dumps(per_query, indent=2), encoding="utf-8"
    )
    print(f"wrote {dst} queries={len(per_query)}", flush=True)


def _group_slices(qid: np.ndarray) -> list[tuple[int, int, int]]:
    """(qid, start, stop) for each contiguous query block."""
    out: list[tuple[int, int, int]] = []
    if qid.size == 0:
        return out
    start = 0
    for i in range(1, qid.size + 1):
        if i == qid.size or qid[i] != qid[start]:
            out.append((int(qid[start]), start, i))
            start = i
    return out


def _recall_at_k(
    scores: np.ndarray,
    y: np.ndarray,
    groups: list[tuple[int, int, int]],
    ks: list[int],
) -> dict[str, float]:
    acc = {k: [] for k in ks}
    for _, a, b in groups:
        yy = y[a:b]
        n_pos = int(yy.sum())
        if n_pos == 0:
            continue
        order = np.argsort(-scores[a:b], kind="stable")
        ranked = yy[order]
        for k in ks:
            acc[k].append(float(ranked[:k].sum()) / n_pos)
    return {f"recall@{k}": float(np.mean(v)) if v else 0.0 for k, v in acc.items()}


def _fit_logistic(
    X: np.ndarray,
    y: np.ndarray,
    *,
    mu: np.ndarray,
    sd: np.ndarray,
    epochs: int,
    lr: float,
) -> tuple[np.ndarray, float]:
    """AS v1 weighted pointwise logistic BCE (numpy)."""
    Xn = ((X - mu) / sd).astype(np.float64)
    w = np.zeros(Xn.shape[1], dtype=np.float64)
    b = 0.0
    yy = y.astype(np.float64)
    n_pos = max(1, int(yy.sum()))
    n_neg = max(1, int((1 - yy).sum()))
    w_pos = 0.5 * len(yy) / n_pos
    w_neg = 0.5 * len(yy) / n_neg
    for epoch in range(epochs):
        z = Xn @ w + b
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        weights = np.where(yy == 1, w_pos, w_neg)
        err = (p - yy) * weights
        w -= lr * (Xn.T @ err) / len(yy)
        b -= lr * float(err.mean())
        if epoch % max(1, epochs // 5) == 0 or epoch == epochs - 1:
            pred = (p >= 0.5).astype(int)
            acc = float((pred == y).mean())
            rec = float((pred[y == 1] == 1).mean()) if n_pos else 0.0
            print(
                f"  epoch {epoch:3d}/{epochs} acc={acc:.3f} pos_recall={rec:.3f}",
                flush=True,
            )
    return w, float(b)


def _fit_ranknet(
    X: np.ndarray,
    y: np.ndarray,
    groups: list[tuple[int, int, int]],
    *,
    mu: np.ndarray,
    sd: np.ndarray,
    epochs: int,
    lr: float,
    pairs_per_query: int,
    l2: float,
    seed: int,
) -> tuple[np.ndarray, float]:
    """Pairwise logistic (RankNet) fit of a linear scorer."""
    import torch

    rng = np.random.default_rng(seed)
    Xn = ((X - mu) / sd).astype(np.float32)

    pos_idx: list[np.ndarray] = []
    neg_idx: list[np.ndarray] = []
    for _, a, b in groups:
        yy = y[a:b]
        p = np.nonzero(yy == 1)[0] + a
        n = np.nonzero(yy == 0)[0] + a
        if p.size and n.size:
            pos_idx.append(p)
            neg_idx.append(n)
    if not pos_idx:
        raise SystemExit("no positive/negative pairs to train on")

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Xt = torch.from_numpy(Xn).to(dev)
    w = torch.zeros(Xn.shape[1], dtype=torch.float32, device=dev, requires_grad=True)
    opt = torch.optim.Adam([w], lr=lr)

    for ep in range(epochs):
        pi_all: list[np.ndarray] = []
        ni_all: list[np.ndarray] = []
        for p, n in zip(pos_idx, neg_idx):
            m = min(pairs_per_query, max(1, p.size * 8))
            pi_all.append(rng.choice(p, size=m, replace=True))
            ni_all.append(rng.choice(n, size=m, replace=True))
        pi = torch.from_numpy(np.concatenate(pi_all)).to(dev)
        ni = torch.from_numpy(np.concatenate(ni_all)).to(dev)
        opt.zero_grad()
        diff = (Xt[pi] - Xt[ni]) @ w
        loss = torch.nn.functional.softplus(-diff).mean() + l2 * (w * w).sum()
        loss.backward()
        opt.step()
        if (ep + 1) % max(1, epochs // 10) == 0:
            print(f"  epoch {ep + 1}/{epochs} loss={float(loss):.4f}", flush=True)

    return w.detach().cpu().numpy().astype(np.float64), 0.0


def _stage_fit(args: argparse.Namespace) -> None:
    root = load_app_settings().data_root
    src = Path(args.out_dir)
    if not src.is_absolute():
        src = root / src
    shards = sorted(src.glob("features_shard*of*.npz"))
    if not shards:
        raise SystemExit(f"no feature shards under {src}")
    Xs, ys, qs = [], [], []
    for p in shards:
        d = np.load(p, allow_pickle=True)
        Xs.append(d["X"])
        ys.append(d["y"])
        qs.append(d["qid"])
        print(f"load {p.name}: rows={len(d['y'])}", flush=True)
    X = np.concatenate(Xs, 0)
    y = np.concatenate(ys, 0).astype(np.int8)
    qid = np.concatenate(qs, 0)
    order = np.argsort(qid, kind="stable")
    X, y, qid = X[order], y[order], qid[order]
    groups = _group_slices(qid)
    print(
        f"total rows={len(y)} queries={len(groups)} pos={int(y.sum())} "
        f"loss={args.loss}",
        flush=True,
    )

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(groups))
    n_dev = max(1, int(len(groups) * args.dev_frac))
    dev_g = [groups[i] for i in sorted(perm[:n_dev])]
    tr_g = [groups[i] for i in sorted(perm[n_dev:])]
    print(f"train queries={len(tr_g)} dev queries={len(dev_g)}", flush=True)

    tr_rows = np.concatenate([np.arange(a, b) for _, a, b in tr_g])
    mu = X[tr_rows].mean(0).astype(np.float64)
    sd = X[tr_rows].std(0).astype(np.float64)
    sd = np.where(sd < 1e-6, 1.0, sd)

    if args.loss == "logistic":
        w, b = _fit_logistic(
            X[tr_rows],
            y[tr_rows],
            mu=mu,
            sd=sd,
            epochs=args.epochs,
            lr=args.lr,
        )
    else:
        w, b = _fit_ranknet(
            X,
            y,
            tr_g,
            mu=mu,
            sd=sd,
            epochs=args.epochs,
            lr=args.lr,
            pairs_per_query=args.pairs_per_query,
            l2=args.l2,
            seed=args.seed,
        )
    model = RankerModel(w=w, b=b, mu=mu, sd=sd, feature_names=list(FEATURE_NAMES))

    ks = [int(k) for k in args.recall_ks.split(",") if k.strip()]
    trained = model.score_batch(X.astype(np.float64))
    heur = np.array([heuristic_score(x) for x in X], dtype=np.float64)
    report = {
        "loss": args.loss,
        "n_queries_train": len(tr_g),
        "n_queries_dev": len(dev_g),
        "feature_names": list(FEATURE_NAMES),
        "weights": {n: float(v) for n, v in zip(FEATURE_NAMES, w)},
        "bias": float(b),
        "emb_sim_mu": float(mu[-1]),
        "emb_sim_w": float(w[-1]),
        "dev": {
            "trained": _recall_at_k(trained, y, dev_g, ks),
            "heuristic": _recall_at_k(heur, y, dev_g, ks),
        },
        "train": {
            "trained": _recall_at_k(trained, y, tr_g, ks),
            "heuristic": _recall_at_k(heur, y, tr_g, ks),
        },
    }
    print(json.dumps(report, indent=2), flush=True)

    out = Path(args.out)
    if not out.is_absolute():
        out = root / out
    model.save(out)
    (out.parent / "train_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(f"saved {out}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=["dump", "fit"], required=True)
    ap.add_argument("--domain", default="")
    ap.add_argument("--train-jsonl", default="")
    ap.add_argument("--out-dir", default="", help="feature shard dir")
    ap.add_argument("--out", default="", help="model.npz path (stage=fit)")
    ap.add_argument("--max-sources", type=int, default=0)
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument(
        "--with-emb",
        action="store_true",
        help="fill emb_sim via BGE (AS v1 recipe); default off",
    )
    ap.add_argument(
        "--loss",
        choices=["logistic", "ranknet"],
        default="logistic",
        help="fit objective (default: logistic = AS v1)",
    )
    ap.add_argument("--dev-frac", type=float, default=0.15)
    ap.add_argument("--epochs", type=int, default=0, help="0 = loss default")
    ap.add_argument("--lr", type=float, default=0.0, help="0 = loss default")
    ap.add_argument("--pairs-per-query", type=int, default=256)
    ap.add_argument("--l2", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--recall-ks", default="50,200,400,800")
    args = ap.parse_args()

    if not args.domain:
        raise SystemExit("--domain required")
    if not args.out_dir:
        raise SystemExit("--out-dir required")
    if args.epochs <= 0:
        args.epochs = 150 if args.loss == "logistic" else 400
    if args.lr <= 0:
        args.lr = 0.25 if args.loss == "logistic" else 0.05

    if args.stage == "dump":
        _stage_dump(args)
    else:
        if not args.out:
            raise SystemExit("--out required for stage=fit")
        _stage_fit(args)


if __name__ == "__main__":
    sys.exit(main())
