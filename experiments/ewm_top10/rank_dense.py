#!/usr/bin/env python3
"""Rank the full temporally eligible EWM corpus with a shared dense encoder."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer

from experiments.ewm_top10.common import candidate_text, parse_month, read_jsonl, write_jsonl
from experiments.ewm_top10.modeling import encode_batch, encode_corpus


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--test", default="outputs/ewm_top10/data/test.jsonl")
    parser.add_argument("--corpus", default="outputs/ewm_top10/data/corpus.jsonl")
    parser.add_argument("--out", required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).to(device).eval()
    corpus = list(read_jsonl(args.corpus))
    queries = list(read_jsonl(args.test))
    corpus_months = [parse_month(row["published_at"]) for row in corpus]
    if any(month is None for month in corpus_months):
        raise SystemExit("corpus contains invalid published_at")

    started = time.perf_counter()
    candidate_matrix = encode_corpus(
        model, tokenizer, (candidate_text(row) for row in corpus), args.max_length, args.batch_size, device
    )
    encode_seconds = time.perf_counter() - started
    output = []
    with torch.no_grad():
        for query in queries:
            qmonth = parse_month(query["published_at"])
            if qmonth is None:
                raise SystemExit(f"invalid query date: {query['query_id']}")
            tick = time.perf_counter()
            q = encode_batch(model, tokenizer, [candidate_text(query)], args.max_length, device).cpu()
            scores = (q @ candidate_matrix.T).squeeze(0)
            eligible = torch.tensor(
                [month < qmonth and row["paper_id"] != query["query_id"] for row, month in zip(corpus, corpus_months)],
                dtype=torch.bool,
            )
            scores[~eligible] = -torch.inf
            available = int(eligible.sum().item())
            k = min(args.top_k, available)
            values, indices = torch.topk(scores, k=k)
            output.append(
                {
                    "query_id": query["query_id"],
                    "method": args.method,
                    "ranked_ids": [corpus[int(i)]["paper_id"] for i in indices],
                    "scores": [float(v) for v in values],
                    "latency_ms": (time.perf_counter() - tick) * 1000.0,
                }
            )
    write_jsonl(args.out, output)
    meta = {"method": args.method, "model": args.model, "top_k": args.top_k, "n_queries": len(output), "corpus_encode_seconds": encode_seconds}
    Path(args.out).with_suffix(".meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
