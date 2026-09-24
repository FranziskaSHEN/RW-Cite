#!/usr/bin/env python3
"""Judge frozen score-and-rank-fusion pools with base Qwen3-32B."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import networkx as nx
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from rwcite.adapter.dataset_v6 import _load_jsonl
from rwcite.adapter.paths import ensure_adapter_dirs, resolve_adapter_v6
from rwcite.gat import protocol as P
from rwcite.ranker.reference_recommend import (
    build_analyze_user_content,
    normalize_arxiv_id,
    parse_analysis,
)


def _resolve_base(root: Path, base_model: str) -> str:
    p = Path(base_model)
    if not p.is_absolute():
        cand = root / p
        if cand.exists():
            return str(cand)
    return str(p)


def _pool_entries(pool_row: dict[str, Any], pool_n: int) -> list[dict[str, Any]]:
    """Normalize and deduplicate one saved fused candidate pool."""
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for paper in list(pool_row.get("pool") or [])[:pool_n]:
        paper_id = normalize_arxiv_id(str(paper.get("id") or ""))
        if not paper_id or paper_id in seen:
            continue
        seen.add(paper_id)
        out.append(
            {
                "id": paper_id,
                "title": (paper.get("title") or paper_id).strip(),
                "score": float(paper.get("score") or 0.0),
                "abstract": (paper.get("abstract") or "")[:500],
            }
        )
    return out


def _enrich_pool_abstracts(
    pool: list[dict[str, Any]],
    graph: nx.DiGraph,
    node_by_id: dict[str, Any],
) -> None:
    """Fill missing abstracts from the frozen graph without changing pool order."""
    for paper in pool:
        if paper.get("abstract"):
            continue
        node = node_by_id.get(paper["id"])
        if node and node in graph:
            paper["abstract"] = (graph.nodes[node].get("abstract") or "")[:500]


def _load_guider(path: Path | None) -> list[dict[str, str]]:
    if path is None or not path.is_file():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    msgs = data.get("messages") if isinstance(data, dict) else data
    out: list[dict[str, str]] = []
    for m in msgs or []:
        role = (m.get("role") or "").strip()
        content = (m.get("content") or "").strip()
        if role in ("user", "assistant") and content:
            out.append({"role": role, "content": content})
    return out


def _encode_chat(
    tokenizer,
    msgs: list[dict[str, str]],
) -> list[int]:
    try:
        encoded = tokenizer.apply_chat_template(
            msgs,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors=None,
            enable_thinking=False,
        )
    except TypeError as e:
        raise SystemExit(f"enable_thinking unsupported: {e}") from e
    if isinstance(encoded, dict) or hasattr(encoded, "input_ids"):
        ids = encoded["input_ids"]
        return list(ids[0] if isinstance(ids[0], (list, tuple)) else ids)
    if isinstance(encoded, list) and encoded and isinstance(encoded[0], list):
        return list(encoded[0])
    return list(encoded)


def synth_split(
    *,
    base_model: str,
    pool_path: Path,
    o2_gexf: Path,
    out_jsonl: Path,
    analyze_n: int = 30,
    pool_n: int = 50,
    max_new_tokens: int = 128,
    max_sources: int = 0,
    shard_id: int = 0,
    num_shards: int = 1,
    batch_size: int = 8,
    guider_messages: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    rows = _load_jsonl(pool_path)
    if max_sources > 0:
        rows = rows[:max_sources]
    if num_shards > 1:
        rows = [r for i, r in enumerate(rows) if i % num_shards == shard_id]
    g = nx.read_gexf(str(o2_gexf), node_type=None, relabel=False, version="1.2draft")
    node_by = {normalize_arxiv_id(str(n)): n for n in g.nodes}

    bnb = BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_8bit_use_double_quant=True,
        bnb_8bit_quant_type="nf8",
        bnb_8bit_compute_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    device = next(model.parameters()).device
    bs = max(1, int(batch_size))

    guider = list(guider_messages or [])
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    done: set[tuple[str, str]] = set()
    n_ok = n_total = 0
    if out_jsonl.is_file() and out_jsonl.stat().st_size > 0:
        for line in out_jsonl.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            sid0 = normalize_arxiv_id(str(rec.get("source_id") or ""))
            cid0 = normalize_arxiv_id(str(rec.get("cand_id") or ""))
            if sid0 and cid0:
                done.add((sid0, cid0))
                n_total += 1
                if rec.get("parse_ok"):
                    n_ok += 1
        print(
            f"resume shard={shard_id}/{num_shards} loaded_done={len(done)} "
            f"running_ok={n_ok}/{n_total}",
            flush=True,
        )

    mode = "a" if done else "w"
    with out_jsonl.open(mode, encoding="utf-8") as fout:
        for pr in rows:
            sid = normalize_arxiv_id(str(pr.get("source_id") or ""))
            node = node_by.get(sid)
            if not sid or not node:
                continue
            title = (pr.get("title") or g.nodes[node].get("title") or sid).strip()
            abstract = (
                pr.get("abstract") or g.nodes[node].get("abstract") or ""
            ).strip()
            pool = _pool_entries(pr, pool_n)[:analyze_n]
            _enrich_pool_abstracts(pool, g, node_by)
            pending = [
                p
                for p in pool
                if (sid, normalize_arxiv_id(str(p["id"]))) not in done
            ]
            if not pending:
                continue
            for i0 in range(0, len(pending), bs):
                chunk = pending[i0 : i0 + bs]
                id_lists: list[list[int]] = []
                for p in chunk:
                    user = build_analyze_user_content(title, abstract, p)
                    msgs = list(guider) + [{"role": "user", "content": user}]
                    id_lists.append(_encode_chat(tokenizer, msgs))
                max_len = max(len(x) for x in id_lists)
                pad_id = int(tokenizer.pad_token_id)
                input_ids = torch.full(
                    (len(id_lists), max_len), pad_id, dtype=torch.long, device=device
                )
                attention_mask = torch.zeros(
                    (len(id_lists), max_len), dtype=torch.long, device=device
                )
                for bi, ids in enumerate(id_lists):
                    input_ids[bi, -len(ids) :] = torch.tensor(
                        ids, dtype=torch.long, device=device
                    )
                    attention_mask[bi, -len(ids) :] = 1
                with torch.no_grad():
                    out = model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )
                prompt_len = input_ids.shape[-1]
                for bi, p in enumerate(chunk):
                    n_total += 1
                    gen = tokenizer.decode(
                        out[bi][prompt_len:], skip_special_tokens=True
                    )
                    parsed = parse_analysis(gen)
                    if parsed.get("ok"):
                        n_ok += 1
                    cid = normalize_arxiv_id(str(p["id"]))
                    done.add((sid, cid))
                    rec = {
                        "source_id": sid,
                        "cand_id": p["id"],
                        "relevance": parsed.get("relevance") or "",
                        "reason": parsed.get("reason") or "",
                        "analysis": (
                            f"Relevance: {parsed.get('relevance') or 'none'}\n"
                            f"Reason for Citation: "
                            f"{parsed.get('reason') or gen.strip()[:240]}"
                        ),
                        "raw": gen,
                        "parse_ok": bool(parsed.get("ok")),
                    }
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fout.flush()
            if n_total % 30 == 0:
                print(
                    f"synth shard={shard_id}/{num_shards} {sid} "
                    f"running_ok={n_ok}/{n_total}",
                    flush=True,
                )

    summary = {
        "n_sources": len(rows),
        "n_pairs": n_total,
        "parse_rate": n_ok / max(1, n_total),
        "out": str(out_jsonl),
        "generator": "base",
        "guider": bool(guider),
        "analyze_n": int(analyze_n),
        "batch_size": int(bs),
        "shard_id": int(shard_id),
        "num_shards": int(num_shards),
        "n_resumed": len(done),
    }
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--domain", required=True, choices=("ewm", "sqc", "gw"))
    ap.add_argument("--split", default="train", choices=("train", "test"))
    ap.add_argument("--base-model", default="models/base/Qwen3-32B")
    ap.add_argument("--analyze-n", type=int, default=30)
    ap.add_argument("--pool-n", type=int, default=50)
    ap.add_argument("--max-sources", type=int, default=0)
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--guider-file", default=None)
    ap.add_argument("--root", default=None)
    args = ap.parse_args(argv)

    root = P.root_path() if not args.root else Path(args.root)
    os.environ.setdefault("RWCITE_ROOT", str(root))
    paths = resolve_adapter_v6(args.domain, root)
    ensure_adapter_dirs(paths)
    pool = paths.pools_train if args.split == "train" else paths.pools_test
    if not pool.is_file():
        raise SystemExit(f"missing fused {args.split} pool: {pool}")
    o2 = paths.gat.full_o2_gexf
    if not o2.is_file():
        raise SystemExit(f"missing O2 gexf: {o2}")

    guider_path = Path(args.guider_file) if args.guider_file else None
    if guider_path and not guider_path.is_absolute():
        guider_path = root / guider_path
    guider = _load_guider(guider_path)

    out = paths.data_dir / f"relevance_analyses_{args.split}.jsonl"
    if args.num_shards > 1:
        out = paths.data_dir / (
            f"analyses_{args.split}.shard{args.shard_id}of{args.num_shards}.jsonl"
        )

    synth_split(
        base_model=_resolve_base(root, args.base_model),
        pool_path=pool,
        o2_gexf=o2,
        out_jsonl=out,
        analyze_n=int(args.analyze_n),
        pool_n=int(args.pool_n),
        max_sources=int(args.max_sources),
        shard_id=int(args.shard_id),
        num_shards=int(args.num_shards),
        batch_size=int(args.batch_size),
        guider_messages=guider,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
