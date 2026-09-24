#!/usr/bin/env python3
"""Train the MasterSet-style shared SciBERT encoder with multi-positive NT-Xent."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup

from experiments.ewm_top10.common import candidate_text, read_jsonl
from experiments.ewm_top10.modeling import encode_batch, multi_positive_nt_xent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="outputs/ewm_top10/data/train.jsonl")
    parser.add_argument("--corpus", default="outputs/ewm_top10/data/corpus.jsonl")
    parser.add_argument("--base", default="models/base/scibert_scivocab_uncased")
    parser.add_argument("--out", default="outputs/ewm_top10/checkpoints/scibert_ntx")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    corpus = {row["paper_id"]: row for row in read_jsonl(args.corpus)}
    train = [row for row in read_jsonl(args.train) if row.get("gold_ids")]
    if not train:
        raise SystemExit("no train queries with eligible gold")

    tokenizer = AutoTokenizer.from_pretrained(args.base)
    model = AutoModel.from_pretrained(args.base).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = math.ceil(len(train) / args.batch_size / args.grad_accum)
    total_steps = max(1, steps_per_epoch * args.epochs)
    scheduler = get_cosine_schedule_with_warmup(optimizer, int(total_steps * args.warmup_ratio), total_steps)
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(args.epochs):
        random.shuffle(train)
        running = 0.0
        model.train()
        for batch_no, start in enumerate(range(0, len(train), args.batch_size), start=1):
            batch = train[start : start + args.batch_size]
            positive_ids = [random.choice(row["gold_ids"]) for row in batch]
            if any(pid not in corpus for pid in positive_ids):
                raise SystemExit("gold paper missing from corpus")
            query_emb = encode_batch(model, tokenizer, [candidate_text(row) for row in batch], args.max_length, device)
            cand_emb = encode_batch(model, tokenizer, [candidate_text(corpus[pid]) for pid in positive_ids], args.max_length, device)
            scores = query_emb @ cand_emb.T
            mask = torch.tensor(
                [[candidate_id in set(row["gold_ids"]) for candidate_id in positive_ids] for row in batch],
                dtype=torch.bool,
                device=device,
            )
            loss = multi_positive_nt_xent(scores, mask, args.temperature) / args.grad_accum
            loss.backward()
            running += float(loss.item()) * args.grad_accum
            if batch_no % args.grad_accum == 0 or start + args.batch_size >= len(train):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
        print(f"epoch={epoch + 1} loss={running / max(1, batch_no):.6f}", flush=True)

    out = Path(f"{args.out}_seed{args.seed}")
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    tokenizer.save_pretrained(out)
    (out / "experiment_meta.json").write_text(json.dumps(vars(args), indent=2) + "\n", encoding="utf-8")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
