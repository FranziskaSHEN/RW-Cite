#!/usr/bin/env python3
"""Dependency-free BM25 baseline over title and abstract with temporal filtering."""

from __future__ import annotations

import argparse
import math
import re
import time
from collections import Counter, defaultdict

from experiments.ewm_top10.common import candidate_text, parse_month, read_jsonl, write_jsonl


TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return TOKEN.findall(text.lower())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", default="outputs/ewm_top10/data/test.jsonl")
    parser.add_argument("--corpus", default="outputs/ewm_top10/data/corpus.jsonl")
    parser.add_argument("--out", default="outputs/ewm_top10/predictions/bm25.jsonl")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--k1", type=float, default=1.2)
    parser.add_argument("--b", type=float, default=0.75)
    args = parser.parse_args()

    corpus = list(read_jsonl(args.corpus))
    doc_terms = [Counter(tokenize(candidate_text(row))) for row in corpus]
    doc_lengths = [sum(counter.values()) for counter in doc_terms]
    months = [parse_month(row["published_at"]) for row in corpus]
    if any(month is None for month in months):
        raise SystemExit("corpus contains invalid published_at")

    # Build the inverted index once. Query-time document frequencies and
    # average lengths are cached per temporal cutoff, which leaves the BM25
    # formula unchanged while avoiding a full corpus scan for every term.
    postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for index, terms in enumerate(doc_terms):
        for term, frequency in terms.items():
            postings[term].append((index, frequency))
    cutoff_cache: dict[tuple[int, int], tuple[list[int], float, Counter[str]]] = {}
    output = []

    for query in read_jsonl(args.test):
        tick = time.perf_counter()
        qmonth = parse_month(query["published_at"])
        if qmonth is None:
            raise SystemExit(f"invalid query date: {query['query_id']}")
        if qmonth not in cutoff_cache:
            eligible_for_cutoff = [
                i for i, month in enumerate(months) if month is not None and month < qmonth
            ]
            average_length = (
                sum(doc_lengths[i] for i in eligible_for_cutoff) / len(eligible_for_cutoff)
                if eligible_for_cutoff
                else 0.0
            )
            document_frequencies: Counter[str] = Counter()
            for i in eligible_for_cutoff:
                document_frequencies.update(doc_terms[i].keys())
            cutoff_cache[qmonth] = (
                eligible_for_cutoff,
                average_length,
                document_frequencies,
            )
        eligible_at_cutoff, cached_avg_len, dfs = cutoff_cache[qmonth]
        eligible = [i for i in eligible_at_cutoff if corpus[i]["paper_id"] != query["query_id"]]
        if len(eligible) < args.top_k:
            raise RuntimeError(f"fewer than {args.top_k} eligible candidates for {query['query_id']}")
        query_terms = set(tokenize(candidate_text(query)))
        scores: dict[int, float] = defaultdict(float)
        eligible_set = set(eligible)
        excluded = set(eligible_at_cutoff) - eligible_set
        avg_len = (
            (cached_avg_len * len(eligible_at_cutoff) - sum(doc_lengths[i] for i in excluded))
            / len(eligible)
        )
        for term in query_terms:
            # Normally the query paper is ineligible because it has the cutoff
            # month. Account for an inconsistent corpus timestamp explicitly so
            # cached statistics remain identical to the uncached definition.
            df = dfs[term] - sum(term in doc_terms[i] for i in excluded)
            idf = math.log(1.0 + (len(eligible) - df + 0.5) / (df + 0.5))
            for i, tf in postings.get(term, ()):
                if i not in eligible_set:
                    continue
                norm = tf + args.k1 * (1.0 - args.b + args.b * doc_lengths[i] / max(avg_len, 1e-9))
                scores[i] += idf * tf * (args.k1 + 1.0) / norm
        scored = [(scores.get(i, 0.0), corpus[i]["paper_id"]) for i in eligible]
        scored.sort(key=lambda item: (-item[0], item[1]))
        top = scored[: args.top_k]
        output.append(
            {
                "query_id": query["query_id"],
                "method": "bm25",
                "ranked_ids": [pid for _, pid in top],
                "scores": [score for score, _ in top],
                "latency_ms": (time.perf_counter() - tick) * 1000.0,
            }
        )
    write_jsonl(args.out, output)
    print(f"wrote {args.out} n={len(output)}")


if __name__ == "__main__":
    main()
