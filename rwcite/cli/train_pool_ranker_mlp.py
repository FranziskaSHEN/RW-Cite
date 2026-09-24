#!/usr/bin/env python3
"""Train an MLP structural ranker on a domain's own graph.

Writes ``mlp.npz`` under the domain ranker/struct tree (or ``--out``).
Only domain-local training artifacts are accepted.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]

from rwcite.graph.domain_paths import (  # noqa: E402
    domain_data_paths,
    working_dir_for_domain,
)
from rwcite.ranker.reference_recommend import normalize_arxiv_id  # noqa: E402
from rwcite.ranker.rr_ranker import extract_features  # noqa: E402
from rwcite.ranker.rr_ranker_mlp import pack_vector, train_mlp_ranknet  # noqa: E402
from rwcite.ranker.rr_universe import build_rr_universe  # noqa: E402
from rwcite.runtime.config import load_app_settings  # noqa: E402
from rwcite.runtime.deps import get_domain_cache, get_retriever_service  # noqa: E402


def _load_bundle_rows(path: Path) -> list[dict]:
    """One row per source id (jsonl carries each bundle twice)."""
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
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--domain", required=True)
    ap.add_argument("--train-jsonl", default="")
    ap.add_argument("--max-queries", type=int, default=0, help="0 = all train sources")
    ap.add_argument("--neg-cap", type=int, default=120)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--hidden", type=int, default=192)
    ap.add_argument("--out", default="")
    ap.add_argument("--cache-features", default="")
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    root = load_app_settings().data_root
    wd = working_dir_for_domain(args.domain, root)
    paths = domain_data_paths(wd)
    jsonl = Path(args.train_jsonl) if args.train_jsonl else Path(paths["jsonl_dir"]) / "train.jsonl"
    if not jsonl.is_absolute():
        jsonl = root / jsonl
    out = Path(args.out) if args.out else Path(paths["root"]) / "ranker" / "struct" / "mlp.npz"
    if not out.is_absolute():
        out = root / out
    cache = (
        Path(args.cache_features)
        if args.cache_features
        else out.parent / "feats" / "mlp_features.npz"
    )
    if not cache.is_absolute():
        cache = root / cache

    # Refuse the deprecated shared ranker so domain-local artifacts cannot be
    # silently replaced by an incompatible checkpoint.
    bad = ("rr-pool-ranker-v1",)
    for p in (out, cache):
        s = str(p.resolve()) if p.exists() else str(p)
        if any(b in s for b in bad):
            print(f"ERROR: refuse AS/shared path: {p}", file=sys.stderr)
            sys.exit(2)

    if cache.is_file() and not args.rebuild:
        data = np.load(cache)
        X = data["X"].astype(np.float32)
        y = data["y"].astype(np.int8)
        qids = data["qids"].astype(np.int32)
        print(f"loaded cache {cache} X={X.shape} pos={int(y.sum())}", flush=True)
    else:
        rng = random.Random(args.seed)
        graph = get_domain_cache().get(args.domain).graph
        retriever = get_retriever_service()
        retriever._load_embedder()
        import torch

        tok, emb = retriever._tokenizer, retriever._embedder
        device = next(emb.parameters()).device
        emb.eval()

        def encode(texts: list[str]) -> np.ndarray:
            outs = []
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

        def search_fn(query: str, limit: int):
            return list(
                retriever.search(
                    scope="domain",
                    domain_id=args.domain,
                    query=query,
                    offset=0,
                    limit=limit,
                    sort="relevance",
                ).get("results")
                or []
            )

        rows = _load_bundle_rows(jsonl)
        rng.shuffle(rows)
        if args.max_queries > 0:
            rows = rows[: args.max_queries]
        print(
            f"MLP feat dump: domain={args.domain} nq={len(rows)} jsonl={jsonl}",
            flush=True,
        )

        X_list, y_list, q_list = [], [], []
        for qi, s in enumerate(rows):
            title = s.get("title") or ""
            abstract = s.get("abstract") or ""
            excl = normalize_arxiv_id(str(s.get("source_id") or ""))
            gold = {normalize_arxiv_id(g["id"]) for g in s.get("gold") or []}
            pack = build_rr_universe(
                title,
                abstract,
                graph,
                search_fn=search_fn,
                exclude_id=excl,
                multi_query=True,
                prf=True,
            )
            U = pack["universe"]
            by = {c["id"]: c for c in U}
            pos = [p for p in gold if p in by]
            neg = [c["id"] for c in U if c["id"] not in gold]
            rng.shuffle(neg)
            neg = neg[: args.neg_cap]
            if not pos:
                continue
            ids = pos + neg
            qv = encode([f"{title}\n{abstract}".strip()[:800]])[0]
            texts = [
                f"{by[i].get('title') or ''}\n{by[i].get('abstract') or ''}"[:800]
                for i in ids
            ]
            cvs = encode(texts)
            max_s = max(
                [float(v) for v in (pack.get("support") or {}).values()], default=1.0
            )
            for pid, cv in zip(ids, cvs):
                feat = extract_features(
                    title=title,
                    abstract=abstract,
                    cand=by[pid],
                    support=pack.get("support") or {},
                    dense_rank=pack.get("dense_rank") or {},
                    focused_hubs=pack.get("focused_hubs") or set(),
                    graph=graph,
                    node_by=pack.get("node_by") or {},
                    emb_sim=float(cv @ qv),
                    max_support=max_s,
                )
                X_list.append(pack_vector(feat, qv, cv))
                y_list.append(1 if pid in gold else 0)
                q_list.append(qi)
            if (qi + 1) % 10 == 0:
                print(
                    f"[{qi+1}/{len(rows)}] rows={len(y_list)} pos={sum(y_list)}",
                    flush=True,
                )

        if not X_list:
            print("ERROR: no training rows", file=sys.stderr)
            sys.exit(1)
        X = np.stack(X_list).astype(np.float32)
        y = np.array(y_list, dtype=np.int8)
        qids = np.array(q_list, dtype=np.int32)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, X=X, y=y, qids=qids)
        print(f"cached features → {cache}", flush=True)

    print(f"train X={X.shape} pos={int(y.sum())} queries={len(set(qids.tolist()))}")
    model = train_mlp_ranknet(
        X,
        y,
        qids,
        hidden=args.hidden,
        epochs=args.epochs,
        lr=args.lr,
        pairs_per_query=300,
        seed=args.seed,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    model.save(out)
    print(f"saved {out}", flush=True)


if __name__ == "__main__":
    main()
