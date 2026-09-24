#!/usr/bin/env python3
"""Rerank frozen candidate IDs with a dense encoder without changing recall."""

from __future__ import annotations

import argparse
import time

import torch
from transformers import AutoModel, AutoTokenizer

from experiments.ewm_top10.common import candidate_text, read_jsonl, write_jsonl
from experiments.ewm_top10.modeling import encode_batch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--method", default="scibert_ntx_fixed_window")
    parser.add_argument("--out", required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--max-length", type=int, default=256)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).to(device).eval()
    queries = {row["query_id"]: row for row in read_jsonl(args.test)}
    corpus = {row["paper_id"]: row for row in read_jsonl(args.corpus)}
    output = []
    with torch.no_grad():
        for window in read_jsonl(args.windows):
            qid = window["query_id"]
            ids = window.get("ranked_ids") or []
            if qid not in queries or len(ids) < args.top_k or any(pid not in corpus for pid in ids):
                raise SystemExit(f"invalid fixed window for {qid}")
            tick = time.perf_counter()
            query_embedding = encode_batch(
                model, tokenizer, [candidate_text(queries[qid])], args.max_length, device
            )
            candidate_embeddings = encode_batch(
                model,
                tokenizer,
                [candidate_text(corpus[pid]) for pid in ids],
                args.max_length,
                device,
            )
            scores = (query_embedding @ candidate_embeddings.T).squeeze(0)
            order = torch.argsort(scores, descending=True).tolist()[: args.top_k]
            output.append(
                {
                    "query_id": qid,
                    "method": args.method,
                    "ranked_ids": [ids[index] for index in order],
                    "scores": [float(scores[index]) for index in order],
                    "latency_ms": (time.perf_counter() - tick) * 1000.0,
                }
            )
    write_jsonl(args.out, output)
    print(f"wrote {args.out} n={len(output)}")


if __name__ == "__main__":
    main()
