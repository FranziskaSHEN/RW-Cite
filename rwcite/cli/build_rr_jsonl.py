#!/usr/bin/env python3
"""Build Reference Recommend train/test jsonl from a domain GEXF.

Gold = all graph successors (full-gold, variable length).
Product K=10; oracle pool injects all gold with N >= |gold| + m_neg.
"""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path

import networkx as nx

ROOT = Path(__file__).resolve().parents[2]


def _load_embeddings(parquet_path: Path) -> dict[str, list[float]] | None:
    if not parquet_path.exists():
        return None
    try:
        import pyarrow.parquet as pq
        import numpy as np
    except ImportError:
        return None
    table = pq.read_table(parquet_path)
    cols = set(table.column_names)
    id_col = "paper_id" if "paper_id" in cols else ("id" if "id" in cols else None)
    emb_col = "embedding" if "embedding" in cols else ("embeddings" if "embeddings" in cols else None)
    if not id_col or not emb_col:
        # try first two columns
        names = table.column_names
        if len(names) < 2:
            return None
        id_col, emb_col = names[0], names[1]
    ids = table.column(id_col).to_pylist()
    embs = table.column(emb_col).to_pylist()
    out: dict[str, list[float]] = {}
    for i, e in zip(ids, embs):
        if i is None or e is None:
            continue
        if hasattr(e, "tolist"):
            e = e.tolist()
        out[str(i)] = list(e)
    return out if out else None


def _top_hard_negatives(
    source_id: str,
    emb_map: dict[str, list[float]],
    forbid: set[str],
    pool: list[str],
    m: int,
) -> list[str]:
    import numpy as np

    if source_id not in emb_map or str(source_id).startswith("_"):
        return []
    mat = emb_map.get("_mat")
    ids = emb_map.get("_ids")
    row = emb_map.get("_row")
    if mat is not None and ids is not None and row is not None and source_id in row:
        # Restrict to ``pool`` (graph nodes); emb index may be a stale superset/subset.
        pool_set = set(pool)
        q = mat[row[source_id]]
        sims = mat @ q
        order = np.argsort(-sims)
        out: list[str] = []
        for j in order:
            pid = ids[int(j)]
            if pid in forbid or pid not in pool_set:
                continue
            out.append(pid)
            if len(out) >= m:
                break
        return out

    q = np.asarray(emb_map[source_id], dtype=np.float32)
    qn = q / (np.linalg.norm(q) + 1e-9)
    scored: list[tuple[float, str]] = []
    for pid in pool:
        if pid in forbid or pid not in emb_map or str(pid).startswith("_"):
            continue
        v = np.asarray(emb_map[pid], dtype=np.float32)
        vn = v / (np.linalg.norm(v) + 1e-9)
        scored.append((float(qn @ vn), pid))
    scored.sort(reverse=True)
    return [pid for _, pid in scored[:m]]


def _prepare_emb_matrix(emb_map: dict) -> None:
    import numpy as np

    ids = [k for k in emb_map.keys() if not str(k).startswith("_")]
    if not ids:
        return
    mat = np.stack([np.asarray(emb_map[i], dtype=np.float32) for i in ids])
    mat = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9)
    emb_map["_ids"] = ids
    emb_map["_mat"] = mat
    emb_map["_row"] = {pid: i for i, pid in enumerate(ids)}


def build_sample(
    g: nx.DiGraph,
    source: str,
    k: int,
    n_pool: int,
    rng: random.Random,
    emb_map: dict[str, list[float]] | None,
    all_nodes: list[str],
) -> dict | None:
    outs = list(g.successors(source))
    if len(outs) < k:
        return None
    # Full-gold: all successors (no sample(..., k))
    gold_ids = list(outs)
    rng.shuffle(gold_ids)
    gold = []
    for tid in gold_ids:
        sent = g.edges[source, tid].get("sentence") or g.edges[source, tid].get("label") or ""
        gold.append(
            {
                "id": tid,
                "title": (g.nodes[tid].get("title") or tid).strip(),
                "sentence": str(sent).strip(),
            }
        )

    # Product target K=10 (subset of full gold for supervised select/bundle)
    assistant_ids = rng.sample(gold_ids, k)

    forbid = set(outs) | {source}
    # Oracle pool: inject all gold; N >= |gold| + m_neg (m_neg ~= legacy n_pool - k)
    m_neg = max(n_pool - k, 1)
    neg_pool = [n for n in all_nodes if n not in forbid]
    negs: list[str] = []
    if emb_map:
        negs = _top_hard_negatives(source, emb_map, forbid, neg_pool, m_neg)
    need = m_neg - len(negs)
    if need > 0:
        remain = [n for n in neg_pool if n not in negs]
        if len(remain) < need:
            return None
        negs.extend(rng.sample(remain, need))

    cand_ids = gold_ids + negs
    rng.shuffle(cand_ids)
    candidates = [
        {"id": cid, "title": (g.nodes[cid].get("title") or cid).strip()}
        for cid in cand_ids
    ]
    return {
        "source_id": source,
        "title": (g.nodes[source].get("title") or source).strip(),
        "abstract": (g.nodes[source].get("abstract") or "").strip(),
        "k": k,
        "n_pool": len(candidates),
        "candidates": candidates,
        "gold": gold,
        "gold_ids": gold_ids,
        "assistant_ids": assistant_ids,
        "full_gold": True,
    }


def _load_id_list(path: Path) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise SystemExit(f"expected JSON list in {path}")
    return [str(x) for x in data]


def _resolve_split(
    *,
    split_file: str,
) -> tuple[set[str], set[str], str, str]:
    """Return train_nodes, test_nodes, split_rule, split_file_resolved."""
    sf = Path(split_file)
    if not sf.is_absolute():
        sf = ROOT / sf
    if not sf.is_dir():
        raise SystemExit("--split-file must be a directory with train_*/test_* json")
    train_p = sf / "train_sources.json"
    test_p = sf / "test_sources.json"
    if not train_p.exists():
        train_p = sf / "train_nodes.json"
    if not test_p.exists():
        test_p = sf / "test_nodes.json"
    if not train_p.exists() or not test_p.exists():
        raise SystemExit(f"missing train/test lists under {sf}")
    train_nodes = set(_load_id_list(train_p))
    test_nodes = set(_load_id_list(test_p))
    rule = f"split_file={sf}"
    meta_p = sf / "split_meta.json"
    if meta_p.exists():
        try:
            m = json.loads(meta_p.read_text(encoding="utf-8"))
            rule = m.get("split_rule") or m.get("split_id") or rule
        except json.JSONDecodeError:
            pass
    return train_nodes, test_nodes, rule, str(sf)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--gexf",
        required=True,
        help="path to domain test_graph_rr.gexf",
    )
    ap.add_argument(
        "--embeddings",
        default="",
        help="optional domain embeddings parquet (unused if empty)",
    )
    ap.add_argument("--out-dir", default="datasets/reference_recommend")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--n-pool", type=int, default=30)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--split-file",
        required=True,
        help="directory with train_sources.json+test_sources.json (from dump_split)",
    )
    ap.add_argument(
        "--which",
        choices=("both", "train", "test"),
        default="both",
        help="which split jsonl to build (gate can use test-only)",
    )
    args = ap.parse_args()

    gexf = ROOT / args.gexf
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {gexf} ...")
    g = nx.read_gexf(gexf)
    if not g.is_directed():
        g = g.to_directed()

    nodes = list(g.nodes())
    train_nodes, test_nodes, split_rule, split_file_res = _resolve_split(
        split_file=args.split_file
    )
    # sources in split lists that are not in graph → ignore
    train_nodes &= set(nodes)
    test_nodes &= set(nodes)
    print(
        f"split: {split_rule} train={len(train_nodes)} test={len(test_nodes)} "
        f"(file={split_file_res})"
    )

    emb_map = _load_embeddings(ROOT / args.embeddings) if args.embeddings else None
    if emb_map:
        print(f"embeddings: yes {len([k for k in emb_map if not str(k).startswith('_')])}")
        _prepare_emb_matrix(emb_map)
        print("embeddings: matrix ready for hard-neg", flush=True)
    else:
        print("embeddings: no (random negatives)")

    rng = random.Random(args.seed)
    all_nodes = nodes

    def collect(split_nodes: set[str], repeats: int) -> list[dict]:
        rows: list[dict] = []
        eligible = [n for n in split_nodes if g.out_degree(n) >= args.k]
        for src in eligible:
            for _ in range(repeats):
                sample = build_sample(
                    g, src, args.k, args.n_pool, rng, emb_map, all_nodes
                )
                if sample:
                    rows.append(sample)
        rng.shuffle(rows)
        return rows

    train_rows: list[dict] = []
    test_rows: list[dict] = []
    if args.which in ("both", "train"):
        train_rows = collect(train_nodes, args.repeats)
    if args.which in ("both", "test"):
        test_rows = collect(test_nodes, 1)

    for name, rows in (("train", train_rows), ("test", test_rows)):
        if args.which not in ("both", name):
            continue
        path = out_dir / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"wrote {path} n={len(rows)}")

    gold_ns = [len(r["gold"]) for r in train_rows + test_rows]
    pool_ns = [len(r["candidates"]) for r in train_rows + test_rows]
    meta = {
        "k": args.k,
        "n_pool_min": args.n_pool,
        "full_gold": True,
        "oracle_pool": "all_gold + m_neg (m_neg = max(n_pool - k, 1))",
        "repeats_per_source": args.repeats,
        "seed": args.seed,
        "gexf": str(gexf),
        "which": args.which,
        "train_samples": len(train_rows),
        "test_samples": len(test_rows),
        "train_sources_eligible": sum(
            1 for n in train_nodes if g.out_degree(n) >= args.k
        ),
        "test_sources_eligible": sum(
            1 for n in test_nodes if g.out_degree(n) >= args.k
        ),
        "gold_n_p50": sorted(gold_ns)[len(gold_ns) // 2] if gold_ns else 0,
        "gold_n_max": max(gold_ns) if gold_ns else 0,
        "pool_n_p50": sorted(pool_ns)[len(pool_ns) // 2] if pool_ns else 0,
        "pool_n_max": max(pool_ns) if pool_ns else 0,
        "split_rule": split_rule,
        "split_mode": "time_elig10",
        "split_file": split_file_res,
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    (out_dir / "meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print("meta", meta)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
