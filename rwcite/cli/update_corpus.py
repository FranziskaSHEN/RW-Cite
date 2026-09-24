#!/usr/bin/env python3
"""Incrementally update Topics supplement and BGE embeddings (optional OAI fetch).

Typical flow after metadata incremental fetch:
  python -m rwcite.cli.update_corpus --skip-fetch
  # or batch GLM topics, then embeddings separately (see scripts/)
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import yaml

from rwcite.graph.domain_retrieve import _load_embedder
from rwcite.retrieve.retriever.corpus_utils import (
    CORPUS_STATE,
    append_supplement_embeddings,
    append_supplement_topics,
    build_id2topics,
    build_paper_list,
    encode_topic_levels,
    load_corpus_state,
    load_supplement_embeddings,
    load_supplement_topics,
    save_corpus_state,
)
from rwcite.retrieve.retriever.metadata_fetch import (
    DEFAULT_META_PATH,
    fetch_oai_metadata,
    infer_next_since_date,
    merge_metadata,
)
from rwcite.retrieve.retriever.topic_generator import generate_topics_batch
from rwcite.runtime.paths import RWCITE_ROOT

ARXIV_META = DEFAULT_META_PATH
DEFAULT_APP_CONFIG = "configs/app.yaml"


def _read_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Update RW-Cite topics supplement and BGE embeddings (optional OAI fetch)"
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_APP_CONFIG,
        help="YAML with corpus_update + retriever.embedder (default: configs/app.yaml)",
    )
    parser.add_argument(
        "--since-date",
        default=None,
        help=(
            "Lower bound YYYY-MM-DD: OAI fetch since (if not --skip-fetch); "
            "also filters topic candidates by update_date. "
            "Default with --skip-fetch: corpus_state.last_metadata_from"
        ),
    )
    parser.add_argument(
        "--until-date",
        default=None,
        help="Upper bound YYYY-MM-DD for OAI / topic update_date filter (default: today UTC)",
    )
    parser.add_argument(
        "--max-metadata-records",
        type=int,
        default=None,
        help="Cap OAI fetch size (debug)",
    )
    parser.add_argument(
        "--topic-backend",
        choices=("llm", "openai", "heuristic"),
        default=None,
        help="Topic generator (default: config corpus_update.topic_backend, llm=GLM)",
    )
    parser.add_argument(
        "--topic-model",
        default=None,
        help="LLM model name (default: config corpus_update.llm_model)",
    )
    parser.add_argument(
        "--topic-batch-size",
        type=int,
        default=None,
        help="Papers per topic sub-batch (append checkpoint)",
    )
    parser.add_argument(
        "--max-topics",
        type=int,
        default=None,
        help="Stop after N missing papers (batch / debug)",
    )
    parser.add_argument(
        "--topic-sleep",
        type=float,
        default=None,
        help="Sleep between sequential LLM calls (ignored when workers > 1)",
    )
    parser.add_argument(
        "--topic-workers",
        type=int,
        default=None,
        help="Concurrent GLM threads",
    )
    parser.add_argument(
        "--all-missing-topics",
        action="store_true",
        help="Ignore date window; process every metadata paper lacking topics",
    )
    parser.add_argument(
        "--embed-batch-size",
        type=int,
        default=256,
        help="BGE batch size for topic-level encode",
    )
    parser.add_argument(
        "--skip-fetch",
        action="store_true",
        help="Skip OAI metadata fetch (use existing snapshot)",
    )
    parser.add_argument(
        "--skip-topics",
        action="store_true",
        help="Skip topic generation",
    )
    parser.add_argument(
        "--skip-embeddings",
        action="store_true",
        help="Skip embedding generation",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report planned work without writing",
    )
    return parser.parse_args()


def resolve_topic_since(args: argparse.Namespace) -> str | None:
    """Date filter for topic candidates. None = no lower bound."""
    if args.all_missing_topics:
        return None
    if args.since_date:
        return args.since_date
    env = (os.environ.get("SINCE_DATE") or os.environ.get("TOPIC_SINCE") or "").strip()
    if env:
        return env
    state = load_corpus_state()
    # Prefer the start of the last metadata incremental window.
    if state.get("last_metadata_from"):
        return str(state["last_metadata_from"])[:10]
    if state.get("last_topics_until"):
        # Continue after previous topic pass (day after).
        from datetime import timedelta

        from rwcite.retrieve.retriever.metadata_fetch import format_date, parse_date

        return format_date(parse_date(state["last_topics_until"]) + timedelta(days=1))
    return None


def papers_missing_topics(
    metadata_path: Path | str,
    *,
    since_date: str | None = None,
    until_date: str | None = None,
    max_topics: int | None = None,
) -> list[dict]:
    """Stream metadata JSONL; return papers lacking HF/supplement topics."""
    print("Loading topic id index (HF Topics + supplement)...", flush=True)
    id2topics = build_id2topics()
    print(f"Topic index size: {len(id2topics):,}", flush=True)

    path = Path(metadata_path)
    missing: list[dict] = []
    scanned = 0
    in_window = 0
    print(
        "Scanning metadata for papers without topics"
        + (
            f" (update_date {since_date or '-inf'} .. {until_date or '+inf'})"
            if (since_date or until_date)
            else " (full corpus)"
        )
        + "...",
        flush=True,
    )
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            scanned += 1
            if scanned % 500_000 == 0:
                print(
                    f"  scanned {scanned:,} rows; window_hits={in_window:,}; "
                    f"missing={len(missing):,}",
                    flush=True,
                )
            entry = json.loads(line)
            pid = entry.get("id")
            if not pid:
                continue
            upd = (entry.get("update_date") or "")[:10]
            if since_date and upd < since_date:
                continue
            if until_date and upd and upd > until_date:
                continue
            in_window += 1
            if pid in id2topics:
                continue
            missing.append(entry)
            if max_topics is not None and len(missing) >= max_topics:
                break

    print(
        f"Metadata scan done: rows={scanned:,}, in_window={in_window:,}, "
        f"missing_topics={len(missing):,}",
        flush=True,
    )
    return missing


def _load_corpus_config(config_path: str) -> dict:
    path = Path(config_path)
    if not path.is_absolute():
        path = Path(RWCITE_ROOT) / path
    if not path.exists():
        return {}
    return _read_yaml(path)


def main() -> int:
    args = parse_args()
    os.chdir(RWCITE_ROOT)

    config = _load_corpus_config(args.config)
    corpus_cfg = config.get("corpus_update", {})
    embedder_name = (
        config.get("retriever", {}) or {}
    ).get("embedder") or corpus_cfg.get("embedder", "models/base/bge-large-en-v1.5")

    topic_backend = args.topic_backend or corpus_cfg.get("topic_backend", "llm")
    topic_model = args.topic_model or corpus_cfg.get("llm_model", "GLM-5.1")
    topic_batch_size = args.topic_batch_size or corpus_cfg.get("topic_batch_size", 128)
    topic_sleep = (
        args.topic_sleep
        if args.topic_sleep is not None
        else float(corpus_cfg.get("topic_sleep_sec", 0.0))
    )
    topic_workers = (
        args.topic_workers
        if args.topic_workers is not None
        else int(corpus_cfg.get("topic_workers", 16))
    )

    until = args.until_date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    fetch_since = args.since_date
    if fetch_since is None and not args.skip_fetch:
        fetch_since = infer_next_since_date(ARXIV_META)
    if fetch_since and fetch_since > until:
        fetch_since = until

    topic_since = resolve_topic_since(args)
    if topic_since and topic_since > until and not args.all_missing_topics:
        topic_since = until

    if not args.skip_fetch:
        print(f"OAI fetch window: {fetch_since} -> {until}", flush=True)
    if not args.skip_topics:
        if args.all_missing_topics:
            print("Topic filter: all metadata papers missing topics", flush=True)
        else:
            print(
                f"Topic filter: update_date {topic_since or '-inf'} -> {until}",
                flush=True,
            )
        llm_cfg = os.environ.get(
            "RWCITE_LLM_BASE_URL", corpus_cfg.get("llm_base_url", "")
        )
        print(
            f"Topic backend: {topic_backend}, model: {topic_model}, endpoint: {llm_cfg}",
            flush=True,
        )
        if topic_backend in ("llm", "openai"):
            print(
                f"Topic concurrency: workers={topic_workers}, sub-batch={topic_batch_size}",
                flush=True,
            )

    if not args.skip_fetch:
        fetched = fetch_oai_metadata(
            fetch_since, until, max_records=args.max_metadata_records
        )
        print(f"Fetched {len(fetched)} OAI metadata records", flush=True)
        added, updated = merge_metadata(ARXIV_META, fetched, dry_run=args.dry_run)
        print(f"Metadata merge: added={added}, updated={updated}", flush=True)
    else:
        print("Skipping metadata fetch (--skip-fetch)", flush=True)

    missing: list[dict] = []
    if not args.skip_topics:
        missing = papers_missing_topics(
            ARXIV_META,
            since_date=None if args.all_missing_topics else topic_since,
            until_date=None if args.all_missing_topics else until,
            max_topics=args.max_topics,
        )
        print(f"Papers to generate topics for: {len(missing)}", flush=True)

    new_topic_entries: list[dict] = []
    if not args.skip_topics and missing:
        for i in range(0, len(missing), topic_batch_size):
            batch = missing[i : i + topic_batch_size]
            print(
                f"Generating topics batch {i // topic_batch_size + 1} "
                f"({len(batch)} papers, backend={topic_backend})",
                flush=True,
            )
            batch_entries = generate_topics_batch(
                batch,
                backend=topic_backend,
                model=topic_model,
                sleep_sec=topic_sleep
                if topic_backend in ("llm", "openai") and topic_workers <= 1
                else 0.0,
                config=config,
                workers=topic_workers if topic_backend in ("llm", "openai") else 1,
            )
            new_topic_entries.extend(batch_entries)
            if args.dry_run:
                continue
            append_supplement_topics(batch_entries)
        if not args.dry_run:
            print(
                f"Appended {len(new_topic_entries)} topic records to supplement",
                flush=True,
            )
    elif args.skip_topics:
        print("Skipping topic generation (--skip-topics)", flush=True)
    elif not missing:
        print("No papers missing topics in the selected window", flush=True)

    if not args.skip_embeddings:
        id2topics = build_id2topics()
        supplement_ids = set(load_supplement_topics())
        paper_ids = build_paper_list()
        candidate_ids = [pid for pid in paper_ids if pid in supplement_ids]
        existing_embed_ids = set(load_supplement_embeddings()[0])
        to_embed = [pid for pid in candidate_ids if pid not in existing_embed_ids]
        print(f"Supplement papers needing embeddings: {len(to_embed)}", flush=True)
        if to_embed and not args.dry_run:
            print(f"Loading embedder: {embedder_name}", flush=True)
            tokenizer, model = _load_embedder(embedder_name)
            embeddings = encode_topic_levels(
                model,
                tokenizer,
                to_embed,
                id2topics,
                batch_size=args.embed_batch_size,
            )
            appended = append_supplement_embeddings(to_embed, embeddings)
            print(f"Appended {appended} embedding rows", flush=True)
        elif args.dry_run and to_embed:
            print(f"[dry-run] Would embed {len(to_embed)} papers", flush=True)
    else:
        print("Skipping embeddings (--skip-embeddings)", flush=True)

    if not args.dry_run:
        state = load_corpus_state()
        state["last_run"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        state["topics_supplement_total"] = len(load_supplement_topics())
        state["topic_backend"] = topic_backend
        state["topic_model"] = topic_model
        if not args.skip_topics:
            state["last_topics_from"] = topic_since
            state["last_topics_until"] = until
        if not args.skip_fetch:
            state["last_metadata_from"] = fetch_since
            state["last_metadata_until"] = until
        save_corpus_state(state)
        print(f"Corpus state saved: {CORPUS_STATE}", flush=True)
        print(
            "Next: bash scripts/run_supplement_embeddings.sh "
            "or bash scripts/run_domain_graph_build.sh --domain <tag> --cutover --seed-only",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
