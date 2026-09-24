"""Evaluate citation sentences for a fixed fused Top-10 recommendation list."""

from __future__ import annotations

import argparse
import json
import os
import re
import warnings
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from rwcite.adapter.paths import resolve_adapter_v6
from rwcite.adapter.prompts import sample_to_messages
from rwcite.gat import protocol as P
from rwcite.ranker.reference_recommend import (
    align_to_candidates,
    cite_consistent_with_title,
    cite_token_jaccard,
    cites_near_duplicate,
    parse_bundle,
    pairwise_cite_distinct_rate,
    title_significant_tokens,
)


def _filter_noisy_warnings() -> None:
    import logging

    warnings.filterwarnings("ignore", message=".*MatMul8bitLt.*")
    warnings.filterwarnings("ignore", message=".*cast from.*")
    warnings.filterwarnings("ignore", category=UserWarning, module="bitsandbytes")
    os.environ.setdefault("BITSANDBYTES_NOWELCOME", "1")
    os.environ.setdefault("PYTHONWARNINGS", "ignore")
    logging.getLogger("bitsandbytes").setLevel(logging.ERROR)


def _shard_dir(out_json: Path) -> Path:
    return out_json.parent / f"{out_json.stem}.shards"


def _shard_path(out_json: Path, rank: int) -> Path:
    return _shard_dir(out_json) / f"rank{rank}.jsonl"


def _load_done_sids(shard_path: Path) -> dict[str, dict]:
    """Load latest record per source_id from a shard JSONL."""
    done: dict[str, dict] = {}
    if not shard_path.is_file():
        return done
    with shard_path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            sid = str(row.get("source_id") or "").strip()
            if sid:
                done[sid] = row
    return done


def _append_shard(shard_path: Path, record: dict) -> None:
    shard_path.parent.mkdir(parents=True, exist_ok=True)
    with shard_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    tmp.replace(path)


def _pairs_from_details(details: list[dict]) -> list[list[str]]:
    all_pairs: list[list[str]] = []
    for d in details:
        if d.get("bertscore_pairs"):
            for a, b in d["bertscore_pairs"]:
                if str(a).strip() and str(b).strip():
                    all_pairs.append([str(a), str(b)])
            continue
        for c in d.get("cites") or []:
            a = (c.get("sentence") or "").strip()
            b = (c.get("gold_sentence") or "").strip()
            if a and b:
                all_pairs.append([a, b])
    return all_pairs


def _write_intermediates(out_json: Path, details: list[dict]) -> Path:
    """Persist pairs + cites BEFORE BertScore so L2 can be filled offline."""
    pairs_path = out_json.with_name(out_json.stem + ".bertscore_pairs.json")
    all_pairs = _pairs_from_details(details)
    _atomic_write_json(pairs_path, {"n": len(all_pairs), "pairs": all_pairs})
    # Full details with cites (drop in-memory-only bulky key duplication later)
    raw_path = out_json.with_name(out_json.stem + ".details_with_cites.json")
    slim_raw = []
    for d in details:
        dd = dict(d)
        # keep cites; drop pairs from this file (pairs live in sidecar)
        dd.pop("bertscore_pairs", None)
        slim_raw.append(dd)
    _atomic_write_json(raw_path, {"n": len(slim_raw), "details": slim_raw})
    print(f"wrote intermediates pairs={pairs_path} details={raw_path}", flush=True)
    return pairs_path


def _ddp_world_size() -> int:
    return int(os.environ.get("WORLD_SIZE") or "1")


def _ddp_rank() -> int:
    return int(os.environ.get("RANK") or os.environ.get("LOCAL_RANK") or "0")


def _ddp_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK") or "0")


def _maybe_init_ddp() -> tuple[int, int, int]:
    import datetime as _dt

    ws = _ddp_world_size()
    rank = _ddp_rank()
    local = _ddp_local_rank()
    if ws > 1 and not torch.distributed.is_initialized():
        # Long timeout: fast ranks wait at end barrier while slow ranks finish generates.
        torch.distributed.init_process_group(
            backend="nccl",
            timeout=_dt.timedelta(hours=6),
        )
    return ws, rank, local


def _load_all_done_sids(out_json: Path) -> dict[str, dict]:
    """Union of all rank*.jsonl under <out>.shards (safe across NPROC changes)."""
    merged: dict[str, dict] = {}
    d = _shard_dir(out_json)
    if not d.is_dir():
        return merged
    for p in sorted(d.glob("rank*.jsonl")):
        merged.update(_load_done_sids(p))
    return merged


def _load_jsonl(path: Path, max_samples: int) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if max_samples > 0 and len(rows) >= max_samples:
                break
    return rows


def _load_pools_by_source(path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    if not path.is_file():
        return out
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            sid = str(row.get("source_id") or "").strip()
            if sid:
                out[sid] = row
    return out


def _title_kw_hit(cite: str, title: str) -> bool:
    toks = title_significant_tokens(title)
    if not toks:
        return True
    cite_toks = set(re.findall(r"[a-z0-9]+", (cite or "").lower()))
    return bool(toks & cite_toks)


def _selected_from_pool(pool_row: dict | None, eval_row: dict, k: int) -> list[dict]:
    """DIRECT_TOP10: L0 pool prefix Top-K with titles."""
    pool = list((pool_row or {}).get("pool") or [])
    selected: list[dict] = []
    for p in pool[:k]:
        selected.append(
            {
                "id": str(p.get("id") or "").strip(),
                "title": (p.get("title") or "").strip(),
                "abstract": (p.get("abstract") or "")[:500],
                "sentence": "",
            }
        )
    if selected:
        return selected
    # Fallback: pool_ids + gold titles / bare ids
    gold_by = {str(g.get("id") or ""): g for g in (eval_row.get("gold") or [])}
    for pid in list(eval_row.get("pool_ids") or [])[:k]:
        g = gold_by.get(str(pid)) or {}
        selected.append(
            {
                "id": str(pid),
                "title": (g.get("title") or str(pid)).strip(),
                "abstract": (g.get("abstract") or "")[:500],
                "sentence": "",
            }
        )
    return selected


def _gold_sents(eval_row: dict) -> dict[str, str]:
    out: dict[str, str] = {}
    for g in eval_row.get("gold") or []:
        gid = str(g.get("id") or "").strip()
        if gid:
            out[gid] = (g.get("sentence") or "").strip()
    return out


def _score_one(selected: list[dict], gold_sents: dict[str, str], *, dup_thr: float) -> dict:
    k = len(selected)
    ids = [(c.get("id") or "").strip() for c in selected]
    sents = [(c.get("sentence") or "").strip() for c in selected]
    titles = [(c.get("title") or "").strip() for c in selected]

    n_parse = sum(1 for s in sents if s)
    n_id_ok = sum(1 for i in ids if i)
    n_title_ok = sum(
        1 for s, t in zip(sents, titles) if s and cite_consistent_with_title(s, t)
    )
    n_kw = sum(1 for s, t in zip(sents, titles) if s and _title_kw_hit(s, t))
    distinct = pairwise_cite_distinct_rate(sents, thr=dup_thr)
    max_j = 0.0
    for i in range(k):
        for j in range(i + 1, k):
            if sents[i] and sents[j]:
                max_j = max(max_j, cite_token_jaccard(sents[i], sents[j]))

    bs_pairs: list[tuple[str, str]] = []
    for cid, sent in zip(ids, sents):
        g = gold_sents.get(cid) or ""
        if sent and g:
            bs_pairs.append((sent, g))

    return {
        "k": k,
        "n_nonempty": n_parse,
        "frac_nonempty": n_parse / max(1, k),
        "frac_id_present": n_id_ok / max(1, k),
        "frac_cite_consistent_with_title": n_title_ok / max(1, k),
        "frac_title_kw": n_kw / max(1, k),
        "pairwise_distinct_rate": distinct,
        "max_pairwise_jaccard": max_j,
        "n_near_dup_pairs": sum(
            1
            for i in range(k)
            for j in range(i + 1, k)
            if sents[i]
            and sents[j]
            and cites_near_duplicate(sents[i], sents[j], thr=dup_thr)
        ),
        "bertscore_pairs": bs_pairs,
    }


def _mean(xs: list[float]) -> float | None:
    return float(sum(xs) / len(xs)) if xs else None


def _summarize(details: list[dict], *, label: str) -> dict[str, Any]:
    keys = [
        "frac_nonempty",
        "frac_id_present",
        "frac_cite_consistent_with_title",
        "frac_title_kw",
        "pairwise_distinct_rate",
        "max_pairwise_jaccard",
    ]
    out: dict[str, Any] = {
        "label": label,
        "n": len(details),
        "kind": "rr_v6_cite_l0_l2",
        "mean_dedup_rewrites": _mean(
            [float(d.get("dedup_rewrites") or 0) for d in details]
        ),
        "think_leak_any": any(bool(d.get("think_leak")) for d in details),
    }
    for k in keys:
        vals = [float(d[k]) for d in details if d.get(k) is not None]
        out[f"mean_{k}"] = _mean(vals)

    pairs: list[tuple[str, str]] = []
    for d in details:
        pairs.extend(d.get("bertscore_pairs") or [])
    out["n_bertscore_pairs"] = len(pairs)
    out["mean_cite_bertscore_f1"] = None
    if pairs:
        try:
            from rwcite.runtime.services.bertscore_service import get_bertscore_service

            rows = get_bertscore_service().score_many(
                [a for a, _ in pairs], [b for _, b in pairs]
            )
            if rows:
                out["mean_cite_bertscore_f1"] = sum(r["f1"] for r in rows) / len(rows)
        except Exception as e:  # noqa: BLE001
            out["bertscore_error"] = str(e)
    return out


def _generate(
    *,
    model: Any,
    tokenizer: Any,
    messages: list[dict[str, str]],
    max_new_tokens: int,
) -> str:
    try:
        encoded = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            enable_thinking=False,
        )
    except TypeError as e:
        raise SystemExit(f"enable_thinking unsupported: {e}") from e
    if hasattr(encoded, "input_ids"):
        input_ids = encoded["input_ids"]
        attention_mask = encoded.get("attention_mask")
    else:
        input_ids = encoded
        attention_mask = None
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    gen_kwargs: dict[str, Any] = {
        "input_ids": input_ids,
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if attention_mask is not None:
        gen_kwargs["attention_mask"] = attention_mask.to(device)
    with torch.no_grad():
        out = model.generate(**gen_kwargs)
    return tokenizer.decode(out[0][input_ids.shape[-1] :], skip_special_tokens=True)


def _strip_cite_prefix(text: str) -> str:
    one = (text or "").split("\n", 1)[0].strip()
    return re.sub(r"^(Cite\s*:\s*)", "", one, flags=re.IGNORECASE).strip()


def _apply_cite_one_rewrites(
    *,
    model: Any,
    tokenizer: Any,
    title: str,
    abstract: str,
    selected: list[dict],
    dup_thr: float,
) -> int:
    """Regenerate title-inconsistent and near-duplicate candidate sentences."""
    for item in selected:
        sent = (item.get("sentence") or "").strip()
        if sent and cite_consistent_with_title(sent, item.get("title") or ""):
            continue
        sample = {
            "task": "cite_one",
            "title": title,
            "abstract": abstract,
            "paper": item,
            "selected": [item],
        }
        msgs = sample_to_messages(sample)[:-1]
        raw = _generate(
            model=model, tokenizer=tokenizer, messages=msgs, max_new_tokens=96
        )
        one = _strip_cite_prefix(raw)
        if one:
            item["sentence"] = one

    rewrites = 0
    for ii, item in enumerate(selected):
        sent = (item.get("sentence") or "").strip()
        if not sent:
            continue
        conflict = any(
            (selected[j].get("sentence") or "").strip()
            and cites_near_duplicate(
                sent, selected[j].get("sentence") or "", thr=dup_thr
            )
            for j in range(ii)
        )
        if not conflict:
            continue
        sample = {
            "task": "cite_one",
            "title": title,
            "abstract": abstract,
            "paper": item,
            "selected": [item],
        }
        msgs = sample_to_messages(sample)[:-1]
        raw = _generate(
            model=model, tokenizer=tokenizer, messages=msgs, max_new_tokens=96
        )
        one = _strip_cite_prefix(raw)
        if one:
            item["sentence"] = one
            rewrites += 1
    return rewrites


def eval_cite(
    *,
    base_model: str,
    adapter_dir: Path,
    eval_jsonl: Path,
    pools_test: Path,
    out_json: Path,
    max_samples: int = 80,
    k: int = 10,
    dup_thr: float = 0.8,
    rewrite_cite_one: bool = True,
    max_new_tokens: int = 512,
    resume: bool = True,
) -> dict[str, Any]:
    from peft import PeftModel

    _filter_noisy_warnings()
    ws, rank, local_rank = _maybe_init_ddp()

    rows = _load_jsonl(eval_jsonl, max_samples)
    if not rows:
        raise SystemExit(f"empty eval jsonl: {eval_jsonl}")
    pools_by = _load_pools_by_source(pools_test)

    if ws > 1:
        rows = [r for i, r in enumerate(rows) if i % ws == rank]

    shard_path = _shard_path(out_json, rank)
    # Global done set so resume works when NPROC changes (e.g. 4 → 8).
    done_map = _load_all_done_sids(out_json) if resume else {}
    if done_map and rank == 0:
        print(
            f"resume: {_shard_dir(out_json)} global_done={len(done_map)} "
            f"world_size={ws}",
            flush=True,
        )
    if not resume and shard_path.is_file():
        shard_path.unlink()

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device_map: Any = {"": local_rank}
    else:
        device_map = "auto"

    bnb = BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_8bit_use_double_quant=True,
        bnb_8bit_quant_type="nf8",
        bnb_8bit_compute_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(str(base_model), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        str(base_model),
        quantization_config=bnb,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, str(adapter_dir))
    model.eval()

    details: list[dict[str, Any]] = []
    n_shard = len(rows)
    n_skip = 0
    for i, row in enumerate(rows):
        sid = str(row.get("source_id") or "")
        if resume and sid and sid in done_map:
            n_skip += 1
            continue
        title = (row.get("title") or "").strip()
        abstract = (row.get("abstract") or "").strip()
        selected = _selected_from_pool(pools_by.get(sid), row, k)
        if not selected:
            print(f"[rank{rank}] skip empty selected {sid}", flush=True)
            continue

        sample = {
            "task": "cite_bundle",
            "title": title,
            "abstract": abstract,
            "selected": [dict(s) for s in selected],
        }
        msgs = sample_to_messages(sample)[:-1]
        gen = _generate(
            model=model,
            tokenizer=tokenizer,
            messages=msgs,
            max_new_tokens=max_new_tokens,
        )
        think_leak = "<think>" in gen or "</think>" in gen
        cited = align_to_candidates(parse_bundle(gen), selected)
        cite_map = {c["id"]: (c.get("sentence") or "").strip() for c in cited}
        for item in selected:
            if item["id"] in cite_map and cite_map[item["id"]]:
                item["sentence"] = cite_map[item["id"]]

        dedup_rewrites = 0
        if rewrite_cite_one:
            dedup_rewrites = _apply_cite_one_rewrites(
                model=model,
                tokenizer=tokenizer,
                title=title,
                abstract=abstract,
                selected=selected,
                dup_thr=dup_thr,
            )

        gold_sents = _gold_sents(row)
        sc = _score_one(selected, gold_sents, dup_thr=dup_thr)
        sc["source_id"] = sid
        sc["dedup_rewrites"] = dedup_rewrites
        sc["think_leak"] = think_leak
        # Compact cites for offline L2 BertScore rescoring.
        sc["cites"] = [
            {
                "id": (item.get("id") or "").strip(),
                "sentence": (item.get("sentence") or "").strip(),
                "gold_sentence": (
                    gold_sents.get((item.get("id") or "").strip()) or ""
                ).strip(),
            }
            for item in selected
        ]
        # Persist immediately (with pairs) so crash/OOM never loses this sample.
        _append_shard(shard_path, sc)
        details.append(sc)
        n_done_global = len(done_map) + len(details)
        print(
            f"[rank{rank}] new {len(details)}/{n_shard - n_skip} "
            f"shard {n_skip + len(details)}/{n_shard} {sid} "
            f"(skipped_done={n_skip} global_done≈{n_done_global}) "
            f"title_ok={sc['frac_cite_consistent_with_title']:.2f} "
            f"distinct={sc['pairwise_distinct_rate']:.2f} "
            f"rewrites={dedup_rewrites}",
            flush=True,
        )

    # Prefer shard files as source of truth (covers resume + this run).
    details = list(_load_done_sids(shard_path).values())

    if ws > 1:
        # device_ids avoids NCCL "unknown device" hang after early-finish ranks.
        torch.distributed.barrier(device_ids=[local_rank])
        gathered: list[Any] = [None] * ws
        torch.distributed.all_gather_object(gathered, details)
        if rank == 0:
            # Merge from shard files (more durable than in-memory gather).
            merged_map: dict[str, dict] = {}
            for p in sorted(_shard_dir(out_json).glob("rank*.jsonl")):
                merged_map.update(_load_done_sids(p))
            # Fallback to gather if a shard is empty
            if not merged_map:
                for part in gathered:
                    for d in part or []:
                        sid = str(d.get("source_id") or "")
                        if sid:
                            merged_map[sid] = d
            details = list(merged_map.values())

    summary: dict[str, Any] = {}
    if rank == 0:
        # CRITICAL: write pairs/cites BEFORE BertScore (L2 can fail/OOM independently).
        pairs_path = _write_intermediates(out_json, details)
        summary = _summarize(details, label="v6")
        summary.update(
            {
                "adapter": str(adapter_dir),
                "eval_jsonl": str(eval_jsonl),
                "pools_test": str(pools_test),
                "k": int(k),
                "dup_thr": float(dup_thr),
                "rewrite_cite_one": bool(rewrite_cite_one),
                "world_size": int(ws),
                "direct_top10": True,
                "shard_dir": str(_shard_dir(out_json)),
                "pairs_sidecar": str(pairs_path),
                "note": "AS v6 L0–L2; not Select/Cite@10-P/Hits@10",
            }
        )
        slim = []
        for d in details:
            dd = dict(d)
            dd.pop("bertscore_pairs", None)
            slim.append(dd)
        _atomic_write_json(out_json, {"summary": summary, "details": slim})
        print(json.dumps(summary, indent=2), flush=True)
        print(f"wrote {out_json}", flush=True)
        print(f"wrote {pairs_path} n_pairs={summary.get('n_bertscore_pairs')}", flush=True)

    if ws > 1:
        torch.distributed.barrier()
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--domain", required=True, choices=("ewm", "sqc", "gw"))
    ap.add_argument("--base-model", default="models/base/Qwen3-32B")
    ap.add_argument(
        "--max-samples",
        type=int,
        default=80,
        help="0 = full cite_eval; >0 truncates",
    )
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--dup-thr", type=float, default=0.8)
    ap.add_argument(
        "--rewrite-cite-one",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="candidate-specific title and near-duplicate rewrites (default: on)",
    )
    ap.add_argument("--out", default=None, help="output JSON path")
    ap.add_argument(
        "--eval-jsonl",
        default=None,
        help="override cite_eval.jsonl (default: domain adapter path)",
    )
    ap.add_argument(
        "--pools-test",
        default=None,
        help="override pools_test_l0.jsonl (DIRECT_TOP10 titles)",
    )
    ap.add_argument("--root", default=None)
    ap.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="resume from per-rank shard JSONL under <out>.shards/ (default: on)",
    )
    args = ap.parse_args(argv)

    root = P.root_path() if not args.root else Path(args.root)
    os.environ.setdefault("RWCITE_ROOT", str(root))
    paths = resolve_adapter_v6(args.domain, root)
    base = Path(args.base_model)
    if not base.is_absolute():
        base = root / base
    eval_path = Path(args.eval_jsonl) if args.eval_jsonl else paths.cite_eval_jsonl
    if not eval_path.is_absolute():
        eval_path = root / eval_path
    pools_test = Path(args.pools_test) if args.pools_test else paths.pools_test
    if not pools_test.is_absolute():
        pools_test = root / pools_test
    if not eval_path.is_file():
        raise SystemExit(f"missing {eval_path}; run dump+build first")
    if not pools_test.is_file():
        raise SystemExit(f"missing {pools_test}")
    if not (paths.model_dir / "adapter_config.json").is_file() and not (
        paths.model_dir / "adapter_model.safetensors"
    ).is_file():
        raise SystemExit(f"missing adapter weights in {paths.model_dir}")

    max_samples = int(args.max_samples)
    if args.out:
        out_json = Path(args.out)
        if not out_json.is_absolute():
            out_json = root / out_json
    elif max_samples == 80:
        out_json = paths.data_dir / "eval_cite_n80.json"
    else:
        # Resolve N from file when full
        n_all = sum(1 for line in eval_path.open(encoding="utf-8") if line.strip())
        n_eff = n_all if max_samples <= 0 else min(max_samples, n_all)
        out_json = paths.data_dir / f"eval_cite_full{n_eff}.json"

    eval_cite(
        base_model=str(base),
        adapter_dir=paths.model_dir,
        eval_jsonl=eval_path,
        pools_test=pools_test,
        out_json=out_json,
        max_samples=max_samples,
        k=int(args.k),
        dup_thr=float(args.dup_thr),
        rewrite_cite_one=bool(args.rewrite_cite_one),
        resume=bool(args.resume),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
