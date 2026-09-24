#!/usr/bin/env python3
"""Train the EWM adaptation of HLM-Cite's two-stage GTE retriever."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from experiments.ewm_top10.common import candidate_text, read_jsonl
from experiments.ewm_top10.modeling import encode_batch, freeze_bottom_encoder_layers, neural_ndcg_loss


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, choices=(1, 2), required=True)
    parser.add_argument("--train", default="outputs/ewm_top10/data/hlm_train.jsonl")
    parser.add_argument("--corpus", default="outputs/ewm_top10/data/corpus.jsonl")
    parser.add_argument("--base", default="models/base/gte-base")
    parser.add_argument("--stage1", default="outputs/ewm_top10/checkpoints/hlm_stage1")
    parser.add_argument("--out", default="outputs/ewm_top10/checkpoints/hlm")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--rank-temperature", type=float, default=0.1)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    corpus = {row["paper_id"]: row for row in read_jsonl(args.corpus)}
    rows = list(read_jsonl(args.train))
    source = args.base if args.stage == 1 else f"{args.stage1}_seed{args.seed}"
    tokenizer = AutoTokenizer.from_pretrained(source)
    # gte-base is distributed with FP16 weights. Adam's default epsilon
    # underflows in FP16 parameter/state tensors and turns training into NaNs.
    model = AutoModel.from_pretrained(source).float().to(device)
    freeze_bottom_encoder_layers(model, 7)
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=args.lr)
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(args.epochs):
        random.shuffle(rows)
        losses = []
        model.train()
        if args.stage == 1:
            usable = [row for row in rows if row.get("core_ids") and row.get("negative_ids")]
            batches = list(range(0, len(usable), args.batch_size))
            for step, start in enumerate(batches, 1):
                batch = usable[start : start + args.batch_size]
                positive_ids = [random.choice(row["core_ids"]) for row in batch]
                q = encode_batch(model, tokenizer, [candidate_text(row) for row in batch], args.max_length, device)
                c = encode_batch(model, tokenizer, [candidate_text(corpus[pid]) for pid in positive_ids], args.max_length, device)
                logits = q @ c.T
                labels = torch.arange(len(batch), dtype=torch.long, device=device)
                # Other queries' positives are in-batch negatives, as in HLM stage 1.
                loss = F.cross_entropy(logits, labels) / args.grad_accum
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite HLM stage 1 loss at epoch={epoch + 1} step={step}")
                loss.backward()
                losses.append(float(loss.item()) * args.grad_accum)
                if step % args.grad_accum == 0 or step == len(batches):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
        else:
            usable = [row for row in rows if row.get("core_ids") and row.get("superficial_ids") and row.get("negative_ids")]
            for step, row in enumerate(usable, 1):
                core = random.sample(row["core_ids"], min(5, len(row["core_ids"])))
                superficial = random.sample(row["superficial_ids"], min(5, len(row["superficial_ids"])))
                negatives = random.sample(row["negative_ids"], min(10, len(row["negative_ids"])))
                ids = core + superficial + negatives
                relevance = torch.tensor([2.0] * len(core) + [1.0] * len(superficial) + [0.0] * len(negatives), device=device)
                q = encode_batch(model, tokenizer, [candidate_text(row)], args.max_length, device)
                c = encode_batch(model, tokenizer, [candidate_text(corpus[pid]) for pid in ids], args.max_length, device)
                loss = neural_ndcg_loss((q @ c.T).squeeze(0), relevance, args.rank_temperature) / args.grad_accum
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite HLM stage 2 loss at epoch={epoch + 1} step={step}")
                loss.backward()
                losses.append(float(loss.item()) * args.grad_accum)
                if step % args.grad_accum == 0 or step == len(usable):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
        if not losses:
            raise SystemExit(f"no usable HLM training rows for stage {args.stage}")
        print(f"stage={args.stage} epoch={epoch + 1} loss={sum(losses)/len(losses):.6f}", flush=True)

    prefix = args.stage1 if args.stage == 1 else f"{args.out}_stage2"
    out = Path(f"{prefix}_seed{args.seed}")
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    tokenizer.save_pretrained(out)
    (out / "experiment_meta.json").write_text(json.dumps(vars(args), indent=2) + "\n", encoding="utf-8")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
