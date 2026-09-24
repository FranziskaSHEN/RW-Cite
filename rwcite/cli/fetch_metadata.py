#!/usr/bin/env python3
"""Fetch / merge arXiv OAI metadata into datasets/arxiv-metadata-oai-snapshot.json.

Modes:
  --download-snapshot   full JSONL from HF mirror (first-time or replace)
  --incremental         date-window OAI fetch until today (or --until-date)
  (default)             single OAI window [--since-date, --until-date]
  --merge-staging       merge staging JSONL only
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from rwcite.runtime.paths import RWCITE_ROOT
from rwcite.retrieve.retriever.metadata_fetch import (
    DEFAULT_FETCH_STATE,
    DEFAULT_META_PATH,
    DEFAULT_SNAPSHOT_URL,
    DEFAULT_STAGING_PATH,
    clear_staging,
    download_full_snapshot,
    fetch_incremental_batches,
    fetch_oai_metadata,
    infer_next_since_date,
    load_staging_records,
    merge_metadata,
    update_corpus_state_after_merge,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch arXiv metadata via OAI-PMH and merge into RW-Cite JSONL snapshot"
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--download-snapshot",
        action="store_true",
        help="Download full arxiv-metadata-oai-snapshot.json from HF mirror",
    )
    mode.add_argument(
        "--merge-staging",
        action="store_true",
        help="Merge staging JSONL into main metadata file only",
    )

    parser.add_argument("--since-date", default=None, help="OAI from date YYYY-MM-DD")
    parser.add_argument("--until-date", default=None, help="OAI until date YYYY-MM-DD")
    parser.add_argument(
        "--incremental",
        action="store_true",
        help="Fetch in date windows until --until-date (or today); merge after each window",
    )
    parser.add_argument("--batch-days", type=int, default=31)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--batch-sleep", type=float, default=5.0)
    parser.add_argument("--output", default=str(DEFAULT_META_PATH))
    parser.add_argument("--staging", default=str(DEFAULT_STAGING_PATH))
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--timeout-sec", type=int, default=180)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--page-sleep", type=float, default=3.0)
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Do not resume from datasets/metadata_fetch_state.json",
    )
    parser.add_argument("--clear-staging", action="store_true")
    parser.add_argument("--snapshot-url", default=DEFAULT_SNAPSHOT_URL)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    os.chdir(RWCITE_ROOT)
    args = parse_args()
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = RWCITE_ROOT / output_path
    staging_path = Path(args.staging)
    if not staging_path.is_absolute():
        staging_path = RWCITE_ROOT / staging_path
    fetch_state = RWCITE_ROOT / DEFAULT_FETCH_STATE

    if args.download_snapshot:
        download_full_snapshot(output_path, url=args.snapshot_url)
        return 0

    if args.merge_staging:
        records = load_staging_records(staging_path)
        if not records:
            print(f"No records in staging: {staging_path}")
            return 1
        added, updated = merge_metadata(output_path, records, dry_run=args.dry_run)
        print(
            f"Merge from staging: fetched={len(records)}, added={added}, updated={updated}"
        )
        if not args.dry_run:
            clear_staging(staging_path, fetch_state)
            print(f"Cleared staging: {staging_path}")
        return 0

    until = args.until_date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    since = args.since_date or infer_next_since_date(output_path)

    if args.incremental:
        print(f"Output: {output_path}")
        print(f"Staging: {staging_path}")
        if args.clear_staging:
            clear_staging(staging_path, fetch_state)
            print("Cleared staging and fetch state")

        summary = fetch_incremental_batches(
            output_path,
            since_date=since,
            until_date=until,
            batch_days=args.batch_days,
            max_batches=args.max_batches,
            max_records=args.max_records,
            timeout_sec=args.timeout_sec,
            retries=args.retries,
            page_sleep_sec=args.page_sleep,
            batch_sleep_sec=args.batch_sleep,
            staging_path=staging_path,
            resume=not args.no_resume,
            dry_run=args.dry_run,
        )
        if not args.dry_run and summary.get("batches", 0) > 0:
            print(
                "Done. Next: bash scripts/run_topic_supplement_batches.sh "
                "then bash scripts/run_supplement_embeddings.sh "
                "(or bash scripts/run_update_corpus.sh --skip-fetch)."
            )
        return 0

    print(f"Metadata fetch window: {since} -> {until}")
    print(f"Output: {output_path}")
    print(f"Staging: {staging_path}")

    if args.clear_staging:
        clear_staging(staging_path, fetch_state)
        print("Cleared staging and fetch state")

    records = fetch_oai_metadata(
        since,
        until,
        max_records=args.max_records,
        timeout_sec=args.timeout_sec,
        retries=args.retries,
        page_sleep_sec=args.page_sleep,
        staging_path=staging_path,
        fetch_state_path=fetch_state,
        resume=not args.no_resume,
    )
    print(f"OAI fetch complete: {len(records)} records")

    if args.dry_run:
        print("[dry-run] Skipping merge into main metadata")
        return 0

    merge_source = (
        load_staging_records(staging_path) if staging_path.exists() else records
    )
    added, updated = merge_metadata(output_path, merge_source, dry_run=False)
    print(
        f"Merged into {output_path}: added={added}, updated={updated}, "
        f"total_source={len(merge_source)}"
    )

    update_corpus_state_after_merge(since, until, output_path)
    clear_staging(staging_path, fetch_state)
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
