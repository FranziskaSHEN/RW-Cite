#!/usr/bin/env python3
"""Train the citation-aware SciBERT cross-encoder on structural windows.

Candidate text contains the title and up to K incoming citation sentences;
the query's outgoing citation is excluded. Negatives are sampled from
non-gold candidates in the same structural window.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from datetime import timedelta
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

ROOT = Path(__file__).resolve().parents[2]

from rwcite.runtime.deps import get_domain_cache, get_retriever_service  # noqa: E402
from rwcite.ranker.reference_recommend import normalize_arxiv_id  # noqa: E402
from rwcite.ranker.rr_ranker import rank_universe  # noqa: E402
from rwcite.ranker.rr_struct_coarse import rank_struct_coarse  # noqa: E402
from rwcite.ranker.rr_ranker_ce import CERanker, _query_text  # noqa: E402
from rwcite.ranker.rr_ranker_ce_sent import cand_text_with_sents  # noqa: E402
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


def build_pair_shard(
    *,
    domain: str,
    train_jsonl: Path,
    out_jsonl: Path,
    max_queries: int,
    window: int,
    neg_per_list: int,
    max_sents: int,
    seed: int,
    shard_id: int,
    num_shards: int,
    short_cand: bool = False,
    hard_neg_citelink: str = "",
) -> int:
    import numpy as np

    from rwcite.ranker.rr_pool_build import _encode_factory
    from rwcite.ranker.rr_ranker_cite_link import (
        _cand_extra,
        load_citelink_ranker,
        pack_pair,
    )

    rng = random.Random(seed + shard_id * 17)
    graph = get_domain_cache().get(domain).graph
    retriever = get_retriever_service()
    encode = _encode_factory(retriever) if hard_neg_citelink else None
    if hard_neg_citelink:
        os.environ["RR_RANKER_CITELINK_PATH"] = str(
            Path(hard_neg_citelink).resolve()
            if Path(hard_neg_citelink).exists()
            else hard_neg_citelink
        )
    cl = load_citelink_ranker() if hard_neg_citelink else None
    if hard_neg_citelink and (cl is None or encode is None):
        print("WARN: hard-neg citelink unavailable; fallback random negs", flush=True)
        cl = None

    def search_fn(query: str, limit: int):
        return list(
            retriever.search(
                scope="domain",
                domain_id=domain,
                query=query,
                offset=0,
                limit=limit,
                sort="relevance",
            ).get("results")
            or []
        )

    rows = _load_rows(train_jsonl)
    # deterministic shard after global shuffle with seed
    rng_all = random.Random(seed)
    order = list(range(len(rows)))
    rng_all.shuffle(order)
    if max_queries > 0:
        order = order[:max_queries]
    order = [i for j, i in enumerate(order) if j % num_shards == shard_id]
    selected = [rows[i] for i in order]

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    n_pairs = 0
    n_q = 0
    n_hard = 0
    with out_jsonl.open("w", encoding="utf-8") as out:
        for qi, s in enumerate(selected):
            title = s.get("title") or ""
            abstract = s.get("abstract") or ""
            excl = normalize_arxiv_id(str(s.get("source_id") or ""))
            gold = {
                normalize_arxiv_id(g["id"])
                for g in (s.get("gold") or [])
                if normalize_arxiv_id(g["id"])
            }
            pack = build_rr_universe(
                title,
                abstract,
                graph,
                search_fn=search_fn,
                exclude_id=excl,
                multi_query=True,
                prf=True,
            )
            coarse = rank_struct_coarse(
                title=title,
                abstract=abstract,
                universe_pack=pack,
                graph=graph,
                n=min(window, len(pack.get("universe") or [])),
                retriever=retriever,
            )
            if len(coarse) < 5:
                continue
            pos_c = [c for c in coarse if c["id"] in gold]
            neg_c = [c for c in coarse if c["id"] not in gold]
            if not pos_c:
                continue
            if len(neg_c) > neg_per_list:
                if cl is not None and encode is not None:
                    q_emb = encode([f"{title}\n{abstract}".strip()[:800]])[0]
                    qv = np.asarray(q_emb, dtype=np.float32).ravel()
                    qn = float(np.linalg.norm(qv))
                    if qn > 1e-9:
                        qv = qv / qn
                    ids = [c["id"] for c in neg_c]
                    emb_map = retriever._ensure_domain_embeddings(domain, ids)
                    node_by = pack.get("node_by") or {}
                    scored = []
                    for c in neg_c:
                        v = emb_map.get(c["id"])
                        if v is None:
                            scored.append((-1e9, c))
                            continue
                        vv = np.asarray(v, dtype=np.float32).ravel()
                        vn = float(np.linalg.norm(vv))
                        if vn > 1e-9:
                            vv = vv / vn
                        extra = (
                            _cand_extra(graph, node_by, c["id"])
                            if cl.extra_dim
                            else None
                        )
                        if cl.extra_dim and (
                            extra is None or extra.size != cl.extra_dim
                        ):
                            extra = np.zeros((cl.extra_dim,), dtype=np.float32)
                        sc = float(cl.score_batch(np.stack([pack_pair(qv, vv, extra)]))[0])
                        scored.append((sc, c))
                    scored.sort(key=lambda x: -x[0])
                    neg_c = [c for _, c in scored[:neg_per_list]]
                    n_hard += 1
                else:
                    neg_c = rng.sample(neg_c, neg_per_list)
            qtext = _query_text(title, abstract)
            global_qid = shard_id * 1_000_000 + qi
            for c in pos_c + neg_c:
                rec = {
                    "qid": global_qid,
                    "query": qtext,
                    "cand": cand_text_with_sents(
                        c,
                        graph=graph,
                        exclude_citer=excl,
                        max_sents=max_sents,
                        short=short_cand,
                    ),
                    "label": 1 if c["id"] in gold else 0,
                    "cid": c["id"],
                }
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_pairs += 1
            n_q += 1
            if (qi + 1) % 20 == 0 or qi + 1 == len(selected):
                print(
                    f"  shard{shard_id} [{qi+1}/{len(selected)}] "
                    f"queries={n_q} pairs={n_pairs} hard_lists={n_hard}",
                    flush=True,
                )
    print(
        f"wrote {n_pairs} pairs ({n_q} q, hard={n_hard}) → {out_jsonl} "
        f"short={short_cand}",
        flush=True,
    )
    return n_pairs


def merge_pair_shards(cache: Path, num_shards: int) -> Path:
    parts = [_shard_path(cache, i, num_shards) for i in range(num_shards)]
    for p in parts:
        if not p.is_file():
            raise SystemExit(f"missing shard: {p}")
    cache.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with cache.open("w", encoding="utf-8") as out:
        for p in parts:
            with p.open(encoding="utf-8") as f:
                for line in f:
                    out.write(line)
                    n += 1
    print(f"merged {n} pairs → {cache}", flush=True)
    return cache


class PairDataset(Dataset):
    def __init__(self, path: Path) -> None:
        self.rows: list[dict] = []
        with path.open(encoding="utf-8") as f:
            for line in f:
                self.rows.append(json.loads(line))
        self.by_q: dict[int, list[int]] = {}
        for i, r in enumerate(self.rows):
            self.by_q.setdefault(int(r["qid"]), []).append(i)
        self.qids = [
            q
            for q, idxs in self.by_q.items()
            if any(self.rows[i]["label"] == 1 for i in idxs)
            and any(self.rows[i]["label"] == 0 for i in idxs)
        ]

    def __len__(self) -> int:
        return len(self.rows)


def sample_list(
    ds: PairDataset, rng: random.Random, *, list_size: int
) -> tuple[list[dict], list[int]]:
    qid = rng.choice(ds.qids)
    idxs = ds.by_q[qid]
    poss = [ds.rows[i] for i in idxs if ds.rows[i]["label"] == 1]
    negs = [ds.rows[i] for i in idxs if ds.rows[i]["label"] == 0]
    rng.shuffle(poss)
    rng.shuffle(negs)
    n_pos = min(len(poss), max(1, list_size // 4))
    take_pos = poss[:n_pos]
    take_neg = negs[: max(0, list_size - len(take_pos))]
    rows = take_pos + take_neg
    rng.shuffle(rows)
    pos_idx = [i for i, r in enumerate(rows) if r["label"] == 1]
    return rows, pos_idx


def _dist_world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def _dist_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def _dist_is_main() -> bool:
    return _dist_local_rank() == 0


def _dist_init() -> None:
    if _dist_world_size() <= 1:
        return
    torch.cuda.set_device(_dist_local_rank())
    if not torch.distributed.is_initialized():
        # Freeze epoch runs on rank 0 only; other ranks wait at barrier >10min default.
        torch.distributed.init_process_group(
            backend="nccl",
            timeout=timedelta(hours=3),
        )


def _dist_cleanup() -> None:
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


def _is_classifier_param(name: str) -> bool:
    return name.startswith("classifier") or ".classifier." in name


def _set_backbone_requires_grad(model, enabled: bool) -> None:
    for name, p in model.named_parameters():
        if _is_classifier_param(name):
            p.requires_grad = True
        else:
            p.requires_grad = enabled


def _make_optimizer(model, lr: float):
    head, body = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (head if _is_classifier_param(n) else body).append(p)
    groups = []
    if body:
        groups.append({"params": body, "lr": lr})
    if head:
        groups.append({"params": head, "lr": lr * 5.0})
    return torch.optim.AdamW(groups, weight_decay=0.01)


def train(
    *,
    base: Path,
    pairs: Path,
    out: Path,
    epochs: int,
    lr: float,
    max_length: int,
    seed: int,
    warmup_ratio: float,
    freeze_epochs: int,
    list_size: int,
    ddp_epoch_mode: str = "full",
    lr_scale_ddp: bool = True,
) -> None:
    use_ddp = _dist_world_size() > 1
    if use_ddp:
        _dist_init()
    rank = _dist_local_rank()
    rng = random.Random(seed + rank)
    torch.manual_seed(seed + rank)
    device = torch.device(
        f"cuda:{rank}" if torch.cuda.is_available() else "cpu"
    )
    tok = AutoTokenizer.from_pretrained(str(base))
    # BertModel / SciBERT towers lack a classifier; ignore size mismatch to attach a fresh CE head.
    raw_model = AutoModelForSequenceClassification.from_pretrained(
        str(base), num_labels=1, ignore_mismatched_sizes=True
    )
    raw_model.to(device)
    raw_model.train()
    ddp_model = None

    ds = PairDataset(pairs)
    ws = _dist_world_size()
    full_steps_per_epoch = max(1, len(ds.qids) * 2)
    mode = (ddp_epoch_mode or "full").strip().lower()
    if mode not in ("full", "shard"):
        raise SystemExit(f"invalid ddp_epoch_mode={ddp_epoch_mode!r}; use full|shard")
    # shard: old behavior (steps // world_size). full: each rank matches 1-GPU steps.
    if use_ddp and mode == "shard":
        ddp_steps_per_epoch = max(1, full_steps_per_epoch // ws)
    else:
        ddp_steps_per_epoch = full_steps_per_epoch
    lr_mult = (ws**0.5) if (use_ddp and lr_scale_ddp and mode == "full") else 1.0
    effective_lr = lr * lr_mult
    full_backbone_updates = 0
    if _dist_is_main():
        name_or_path = getattr(raw_model.config, "_name_or_path", None) or str(base)
        print(
            f"base={base} _name_or_path={name_or_path} "
            f"pairs={len(ds)} queries_with_both={len(ds.qids)} device={device} "
            f"list_size={list_size} ddp={use_ddp} world_size={ws} "
            f"ddp_epoch_mode={mode} ddp_steps_per_epoch={ddp_steps_per_epoch} "
            f"full_steps_per_epoch={full_steps_per_epoch} "
            f"lr={lr} lr_mult={lr_mult:.4f} effective_lr={effective_lr} "
            f"freeze_epochs={freeze_epochs}",
            flush=True,
        )
    step = 0
    opt = None
    sched = None

    if use_ddp and freeze_epochs == 0:
        ddp_model = torch.nn.parallel.DistributedDataParallel(
            raw_model,
            device_ids=[rank],
            output_device=rank,
            find_unused_parameters=False,
        )

    for ep in range(epochs):
        freeze = ep < freeze_epochs
        _set_backbone_requires_grad(raw_model, enabled=not freeze)

        if use_ddp and ep == freeze_epochs and freeze_epochs > 0:
            for p in raw_model.parameters():
                torch.distributed.broadcast(p.data, src=0)
            torch.distributed.barrier()
            ddp_model = torch.nn.parallel.DistributedDataParallel(
                raw_model,
                device_ids=[rank],
                output_device=rank,
                find_unused_parameters=False,
            )
            opt = None

        freeze_single = use_ddp and freeze
        if freeze_single:
            if _dist_is_main():
                train_model = raw_model
                epoch_steps = full_steps_per_epoch
                # Freeze phase: head-only; keep unscaled head lr (lr*3 body slot unused).
                cur_lr = lr * 3.0
                if opt is None:
                    opt = _make_optimizer(train_model, cur_lr)
                    remain = max(1, (epochs - ep) * epoch_steps)
                    sched = get_linear_schedule_with_warmup(
                        opt,
                        num_warmup_steps=int(remain * warmup_ratio),
                        num_training_steps=remain,
                    )
                    print(
                        f"epoch {ep+1}: freeze_backbone={freeze} lr_body={cur_lr} "
                        f"trainable={sum(p.numel() for p in raw_model.parameters() if p.requires_grad)/1e6:.2f}M "
                        f"ddp=False (rank0 only)",
                        flush=True,
                    )
                lw_sum, lw_n = 0.0, 0
                recall30_sum, recall30_n = 0.0, 0
                for _ in range(epoch_steps):
                    rows, pos_idx = sample_list(ds, rng, list_size=list_size)
                    while not pos_idx:
                        rows, pos_idx = sample_list(ds, rng, list_size=list_size)
                    enc = tok(
                        [r["query"] for r in rows],
                        [r["cand"] for r in rows],
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=max_length,
                    )
                    enc = {k: v.to(device) for k, v in enc.items()}
                    logits = train_model(**enc).logits.squeeze(-1)
                    logp = F.log_softmax(logits, dim=0)
                    loss = -logp[pos_idx].mean()
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(train_model.parameters(), 1.0)
                    opt.step()
                    sched.step()
                    lw_sum += float(loss.item())
                    lw_n += 1
                    step += 1
                    with torch.no_grad():
                        order = torch.argsort(logits, descending=True).tolist()
                        top30 = set(order[:30])
                        hit = sum(1 for i in pos_idx if i in top30) / len(pos_idx)
                        recall30_sum += hit
                        recall30_n += 1
                print(
                    f"epoch {ep+1}/{epochs} listwise_nll={lw_sum/max(1,lw_n):.4f} "
                    f"train_list_R@30={recall30_sum/max(1,recall30_n):.3f} step={step}",
                    flush=True,
                )
            torch.distributed.barrier()
            continue

        if use_ddp:
            train_model = ddp_model
            epoch_steps = ddp_steps_per_epoch
        else:
            train_model = raw_model
            epoch_steps = full_steps_per_epoch

        cur_lr = effective_lr if not freeze else effective_lr * 3.0
        if opt is None or ep == freeze_epochs:
            opt = _make_optimizer(train_model, cur_lr)
            remain = max(1, (epochs - ep) * epoch_steps)
            sched = get_linear_schedule_with_warmup(
                opt,
                num_warmup_steps=int(remain * warmup_ratio),
                num_training_steps=remain,
            )
            if _dist_is_main():
                print(
                    f"epoch {ep+1}: freeze_backbone={freeze} lr_body={cur_lr} "
                    f"trainable={sum(p.numel() for p in raw_model.parameters() if p.requires_grad)/1e6:.2f}M "
                    f"ddp={use_ddp and not freeze} mode={mode}",
                    flush=True,
                )

        lw_sum, lw_n = 0.0, 0
        recall30_sum, recall30_n = 0.0, 0
        for _ in range(epoch_steps):
            rows, pos_idx = sample_list(ds, rng, list_size=list_size)
            while not pos_idx:
                rows, pos_idx = sample_list(ds, rng, list_size=list_size)
            enc = tok(
                [r["query"] for r in rows],
                [r["cand"] for r in rows],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            logits = train_model(**enc).logits.squeeze(-1)
            logp = F.log_softmax(logits, dim=0)
            loss = -logp[pos_idx].mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(train_model.parameters(), 1.0)
            opt.step()
            sched.step()
            lw_sum += float(loss.item())
            lw_n += 1
            step += 1
            if not freeze:
                full_backbone_updates += 1
            with torch.no_grad():
                order = torch.argsort(logits, descending=True).tolist()
                top30 = set(order[:30])
                hit = sum(1 for i in pos_idx if i in top30) / len(pos_idx)
                recall30_sum += hit
                recall30_n += 1

        if _dist_is_main():
            print(
                f"epoch {ep+1}/{epochs} listwise_nll={lw_sum/max(1,lw_n):.4f} "
                f"train_list_R@30={recall30_sum/max(1,recall30_n):.3f} step={step}",
                flush=True,
            )

    if _dist_is_main():
        out.mkdir(parents=True, exist_ok=True)
        raw_model.save_pretrained(str(out))
        tok.save_pretrained(str(out))
        meta = {
            "kind": "rr_ce_sent_v4",
            "base": str(base),
            "pairs": str(pairs),
            "epochs": epochs,
            "lr": lr,
            "effective_lr": effective_lr,
            "max_length": max_length,
            "list_size": list_size,
            "loss": "listwise_softmax",
            "n_pairs": len(ds),
            "n_queries": len(ds.qids),
            "ddp_world_size": ws,
            "ddp_epoch_mode": mode if use_ddp else "n/a",
            "lr_scale_ddp": bool(lr_scale_ddp and use_ddp and mode == "full"),
            "full_backbone_updates": full_backbone_updates,
        }
        (out / "train_meta.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )
        print(f"saved CE-sent → {out}", flush=True)
        ce = CERanker(raw_model.eval(), tok, device=device, max_length=max_length)
        s = ce.score_pairs(
            "test query about world models",
            [
                "unrelated chemistry abstract",
                "Cite: We build on prior world-model work for planning.",
            ],
        )
        print(f"sanity scores={s}", flush=True)
    if use_ddp:
        _dist_cleanup()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="")
    ap.add_argument("--train-jsonl", default="datasets/reference_recommend/train.jsonl")
    ap.add_argument(
        "--base",
        default="models/base/scibert_scivocab_uncased",
        help="CE init (Release default: SciBERT; attaches new CE head)",
    )
    ap.add_argument("--pairs", default="datasets/rr_pool_ranker/ce_sent_v4_pairs.jsonl")
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--build-only", action="store_true")
    ap.add_argument("--merge-shards", action="store_true")
    ap.add_argument("--max-queries", type=int, default=800)
    ap.add_argument("--window", type=int, default=400)
    ap.add_argument("--neg-per-list", type=int, default=64)
    ap.add_argument("--max-sents", type=int, default=4)
    ap.add_argument(
        "--short-cand",
        action="store_true",
        help="v4b: title + Cite sentences (drop long abstract)",
    )
    ap.add_argument(
        "--hard-neg-citelink",
        default="",
        help="v4b: pick in-window negs by citelink score (model.npz)",
    )
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--freeze-epochs", type=int, default=1)
    ap.add_argument("--list-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-length", type=int, default=384)
    ap.add_argument("--warmup-ratio", type=float, default=0.06)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--out", default="models/rr-pool-ranker-ce-sent-v4")
    ap.add_argument("--train-only", action="store_true")
    ap.add_argument(
        "--ddp-epoch-mode",
        choices=("full", "shard"),
        default="full",
        help="full: each rank runs n_qids*2 steps/epoch (1-GPU equivalent); "
        "shard: steps // world_size (legacy under-training)",
    )
    ap.add_argument(
        "--lr-scale-ddp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="when ddp_epoch_mode=full, multiply lr by sqrt(world_size)",
    )
    args = ap.parse_args()

    pairs = Path(args.pairs)

    if args.merge_shards:
        merge_pair_shards(pairs, args.num_shards)
        if args.build_only:
            return
        # fall through to train

    if args.build_only or (
        args.rebuild and not args.merge_shards and not args.train_only
    ):
        out_p = (
            _shard_path(pairs, args.shard_id, args.num_shards)
            if args.num_shards > 1
            else pairs
        )
        if args.rebuild or not out_p.is_file():
            build_pair_shard(
                domain=args.domain,
                train_jsonl=Path(args.train_jsonl),
                out_jsonl=out_p,
                max_queries=args.max_queries,
                window=args.window,
                neg_per_list=args.neg_per_list,
                max_sents=args.max_sents,
                seed=args.seed,
                shard_id=args.shard_id,
                num_shards=args.num_shards,
                short_cand=bool(args.short_cand),
                hard_neg_citelink=args.hard_neg_citelink,
            )
        if args.build_only:
            return

    if args.train_only or args.merge_shards or pairs.is_file():
        if not pairs.is_file():
            raise SystemExit(f"pairs missing: {pairs}")
        train(
            base=Path(args.base),
            pairs=pairs,
            out=Path(args.out),
            epochs=args.epochs,
            lr=args.lr,
            max_length=args.max_length,
            seed=args.seed,
            warmup_ratio=args.warmup_ratio,
            freeze_epochs=args.freeze_epochs,
            list_size=args.list_size,
            ddp_epoch_mode=args.ddp_epoch_mode,
            lr_scale_ddp=bool(args.lr_scale_ddp),
        )
    else:
        raise SystemExit(f"pairs missing: {pairs}; pass --rebuild or --merge-shards")


if __name__ == "__main__":
    main()
