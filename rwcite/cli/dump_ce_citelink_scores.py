#!/usr/bin/env python3
"""Dump CE + citelink raw scores on n80 for offline blend sweep.

One CE forward pass per query; citelink always scored on the shortlist so β can
be re-applied without re-running CE.

Supports --shard-id/--num-shards and --merge-shards.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

from rwcite.runtime.deps import get_domain_cache, get_retriever_service  # noqa: E402
from rwcite.ranker.reference_recommend import normalize_arxiv_id  # noqa: E402
from rwcite.ranker.rr_pool_build import _encode_factory  # noqa: E402
from rwcite.ranker.rr_ranker import rank_universe  # noqa: E402
from rwcite.ranker.rr_struct_coarse import rank_struct_coarse  # noqa: E402
from rwcite.ranker.rr_ranker_ce_sent import (  # noqa: E402
    dump_window_ce_citelink_scores,
    load_ce_sent_ranker,
    rr_ce_sent_params,
)
from rwcite.ranker.rr_ranker_cite_link import load_citelink_ranker  # noqa: E402
from rwcite.ranker.rr_universe import build_rr_universe  # noqa: E402


def _load_rows(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("task") in (None, "", "bundle"):
                rows.append(r)
    return rows


def _merge_shards(*, out_dir: Path, tag: str, num_shards: int, out_path: Path) -> None:
    details: list[dict] = []
    meta: dict = {}
    for i in range(num_shards):
        sp = out_dir / f"{tag}_shard{i}of{num_shards}.json"
        if not sp.is_file():
            raise SystemExit(f"missing shard: {sp}")
        data = json.loads(sp.read_text(encoding="utf-8"))
        details.extend(data.get("details") or [])
        if not meta:
            meta = {k: v for k, v in data.items() if k != "details"}
        print(f"merge {sp.name}: +{len(data.get('details') or [])}", flush=True)
    details.sort(key=lambda d: str(d.get("source_id") or ""))
    out = {
        **meta,
        "n_samples": len(details),
        "merged_from": [f"{tag}_shard{i}of{num_shards}.json" for i in range(num_shards)],
        "details": details,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"merged n={len(details)} → {out_path}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="")
    ap.add_argument(
        "--test-jsonl",
        default="",
    )
    ap.add_argument(
        "--ranker-ce-sent",
        default="",
    )
    ap.add_argument(
        "--ranker-citelink",
        default="",
    )
    ap.add_argument("--max-samples", type=int, default=80)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--merge-shards", action="store_true")
    ap.add_argument("--tag", default="ce_citelink_score_dump_n80")
    ap.add_argument("--out-dir", default="datasets/rr_pool_ranker")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    if args.merge_shards:
        out = (
            Path(args.out)
            if args.out
            else Path(args.out_dir) / f"{args.tag}_merged.json"
        )
        _merge_shards(
            out_dir=Path(args.out_dir),
            tag=args.tag,
            num_shards=args.num_shards,
            out_path=out,
        )
        return

    if args.num_shards < 1 or args.shard_id < 0 or args.shard_id >= args.num_shards:
        raise SystemExit(f"bad shard: id={args.shard_id} num={args.num_shards}")

    os.environ["RR_RANKER_CE_SENT_PATH"] = str(Path(args.ranker_ce_sent).resolve())
    os.environ["RR_RANKER_CITELINK_PATH"] = str(Path(args.ranker_citelink).resolve())
    os.environ.setdefault("RR_CE_SENT_SHORT", "1")
    os.environ.setdefault("RR_CE_SENT_WINDOW", "400")
    os.environ.setdefault("RR_CE_SENT_MAX_SENTS", "3")
    os.environ.setdefault("RR_CE_SENT_MAX_LEN", "256")
    os.environ.setdefault("RR_RANKER_FULL_EMB", "1")
    os.environ.setdefault("RR_RANKER_WITH_EMB", "1")
    # Force citelink scoring path regardless of blend
    os.environ["RR_CE_SENT_CITELINK_BLEND"] = "0"

    ce = load_ce_sent_ranker()
    citelink = load_citelink_ranker()
    if ce is None:
        raise SystemExit(f"missing CE: {args.ranker_ce_sent}")
    if citelink is None:
        raise SystemExit(f"missing citelink: {args.ranker_citelink}")

    sp = rr_ce_sent_params()
    graph = get_domain_cache().get(args.domain).graph
    retriever = get_retriever_service()
    encode = _encode_factory(retriever)
    if encode is None:
        raise SystemExit("embedder unavailable")

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

    rows = _load_rows(Path(args.test_jsonl))
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    rows = rows[: args.max_samples]
    if args.num_shards > 1:
        rows = [r for i, r in enumerate(rows) if i % args.num_shards == args.shard_id]
        print(
            f"shard {args.shard_id}/{args.num_shards} n={len(rows)} "
            f"(of max_samples={args.max_samples})",
            flush=True,
        )

    details: list[dict] = []
    win = int(sp["window"])
    for qi, s in enumerate(rows):
        title = s.get("title") or ""
        abstract = s.get("abstract") or ""
        excl = normalize_arxiv_id(str(s.get("source_id") or ""))
        gold = sorted(
            {
                normalize_arxiv_id(g["id"])
                for g in (s.get("gold") or [])
                if normalize_arxiv_id(g["id"])
            }
        )
        pack = build_rr_universe(
            title,
            abstract,
            graph,
            search_fn=search_fn,
            exclude_id=excl,
            multi_query=True,
            prf=True,
        )
        # Embeddings for struct window (coarse=struct default)
        coarse_probe = rank_struct_coarse(
            title=title,
            abstract=abstract,
            universe_pack=pack,
            graph=graph,
            n=min(win, len(pack.get("universe") or [])),
            encode=encode,
            retriever=retriever,
        )
        ids = [c["id"] for c in coarse_probe]
        q_emb = encode([f"{title}\n{abstract}".strip()[:800]])[0]
        cand_embs = retriever._ensure_domain_embeddings(args.domain, ids)
        dump = dump_window_ce_citelink_scores(
            title=title,
            abstract=abstract,
            universe_pack=pack,
            graph=graph,
            ce=ce,
            exclude_id=excl,
            citelink=citelink,
            cand_embs=cand_embs,
            q_emb=q_emb,
            window=win,
            score_m=int(sp.get("score_m") or win),
            coarse_mode=str(sp.get("coarse") or "struct"),
            encode=encode,
            retriever=retriever,
        )
        u_ids = set(dump.get("universe_ids") or [])
        if not u_ids:
            u_ids = {
                normalize_arxiv_id(c.get("id") or "")
                for c in (pack.get("universe") or [])
                if c.get("id")
            }
        gold_set = set(gold)
        n_ce = sum(1 for c in dump["cands"] if c.get("ce_scored"))
        details.append(
            {
                "source_id": excl,
                "n_gold": len(gold_set),
                "gold": gold,
                "universe_recall": (
                    len(gold_set & u_ids) / len(gold_set) if gold_set else 0.0
                ),
                "universe_size": dump.get("universe_size"),
                "window": dump.get("window"),
                "n_ce_scored": n_ce,
                "cands": dump["cands"],
            }
        )
        print(
            f"[{qi+1}/{len(rows)}] {excl}: gold={len(gold_set)} "
            f"win={dump.get('window')} ce={n_ce} U={details[-1]['universe_recall']:.2f}",
            flush=True,
        )

    out_path = (
        Path(args.out)
        if args.out
        else Path(args.out_dir) / f"{args.tag}_shard{args.shard_id}of{args.num_shards}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "protocol": "ce_citelink_score_dump_n80",
        "domain": args.domain,
        "test_jsonl": args.test_jsonl,
        "ce_path": args.ranker_ce_sent,
        "citelink_path": args.ranker_citelink,
        "max_samples": args.max_samples,
        "seed": args.seed,
        "shard_id": args.shard_id,
        "num_shards": args.num_shards,
        "ce_params": {
            "window": sp["window"],
            "score_m": sp["score_m"],
            "coarse": sp["coarse"],
            "max_sents": sp["max_sents"],
            "short_cand": sp["short_cand"],
            "struct_blend": sp["struct_blend"],
        },
        "n_samples": len(details),
        "details": details,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
