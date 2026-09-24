"""arXiv metadata fetch via OAI-PMH and merge into JSONL snapshot."""

from __future__ import annotations

import json
import shutil
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from rwcite.retrieve.retriever.corpus_utils import CORPUS_STATE, load_corpus_state, load_metadata_index, save_corpus_state

# Official OAI-PMH endpoint (Identify baseURL). Do NOT use https://arxiv.org/oai2 —
# it 302-redirects to http://export.arxiv.org/oai2, which often times out on port 80.
ARXIV_OAI = "https://oaipmh.arxiv.org/oai"
OAI_NS = "http://www.openarchives.org/OAI/2.0/"
ARXIV_NS = "http://arxiv.org/OAI/arXiv/"

DEFAULT_META_PATH = Path("datasets/arxiv-metadata-oai-snapshot.json")
DEFAULT_STAGING_PATH = Path("datasets/arxiv-metadata-oai-staging.jsonl")
DEFAULT_FETCH_STATE = Path("datasets/metadata_fetch_state.json")
DEFAULT_BATCH_STATE = Path("datasets/metadata_fetch_batch_state.json")
DEFAULT_SNAPSHOT_URL = (
    "https://hf-mirror.com/spaces/ddiddu/simsearch/resolve/main/"
    "arxiv-metadata-oai-snapshot.json"
)


def arxiv_tag(name: str) -> str:
    return f"{{{ARXIV_NS}}}{name}"


def oai_tag(name: str) -> str:
    return f"{{{OAI_NS}}}{name}"


def iso_date(date_str: str) -> str:
    """OAI date arg: oaipmh.arxiv.org expects YYYY-MM-DD (not ISO8601 datetime)."""
    return date_str[:10]


def parse_date(date_str: str) -> date:
    return date.fromisoformat(iso_date(date_str))


def format_date(d: date) -> str:
    return d.isoformat()


def iter_date_windows(start: str, end: str, batch_days: int) -> list[tuple[str, str]]:
    """Split [start, end] into inclusive OAI date windows of at most batch_days."""
    if batch_days < 1:
        raise ValueError("batch_days must be >= 1")
    windows: list[tuple[str, str]] = []
    cur = parse_date(start)
    end_d = parse_date(end)
    while cur <= end_d:
        window_end = min(cur + timedelta(days=batch_days - 1), end_d)
        windows.append((format_date(cur), format_date(window_end)))
        cur = window_end + timedelta(days=1)
    return windows


def infer_since_date(metadata_path: Path | str = DEFAULT_META_PATH) -> str:
    """Legacy: return last known metadata date (may overlap last fetched day)."""
    metadata_path = Path(metadata_path)
    state = load_corpus_state()
    if state.get("last_metadata_until"):
        return state["last_metadata_until"][:10]
    max_date = "1991-01-01"
    if metadata_path.exists():
        with open(metadata_path, encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line)
                upd = entry.get("update_date") or ""
                if upd > max_date:
                    max_date = upd
    return max_date


def infer_next_since_date(metadata_path: Path | str = DEFAULT_META_PATH) -> str:
    """Next OAI from-date: day after last completed batch (no overlap)."""
    batch_state = load_batch_state()
    if batch_state.get("last_completed_until"):
        nxt = parse_date(batch_state["last_completed_until"]) + timedelta(days=1)
        return format_date(nxt)

    state = load_corpus_state()
    if state.get("last_metadata_until"):
        nxt = parse_date(state["last_metadata_until"]) + timedelta(days=1)
        return format_date(nxt)

    return infer_since_date(metadata_path)


def authors_to_string(authors_el: ET.Element | None) -> str:
    if authors_el is None:
        return ""
    names = []
    for author in authors_el.findall(arxiv_tag("author")):
        keyname = author.find(arxiv_tag("keyname"))
        forenames = author.find(arxiv_tag("forenames"))
        parts = []
        if forenames is not None and forenames.text:
            parts.append(forenames.text.strip())
        if keyname is not None and keyname.text:
            parts.append(keyname.text.strip())
        if parts:
            names.append(" ".join(parts))
    return ", ".join(names)


def parse_oai_record(record_el: ET.Element) -> dict | None:
    meta = record_el.find(oai_tag("metadata"))
    if meta is None:
        return None
    arxiv = meta.find(arxiv_tag("arXiv"))
    if arxiv is None:
        return None
    pid_el = arxiv.find(arxiv_tag("id"))
    if pid_el is None or not pid_el.text:
        return None
    paper_id = pid_el.text.strip()
    title_el = arxiv.find(arxiv_tag("title"))
    abstract_el = arxiv.find(arxiv_tag("abstract"))
    categories_el = arxiv.find(arxiv_tag("categories"))
    created_el = arxiv.find(arxiv_tag("created"))
    updated_el = arxiv.find(arxiv_tag("updated"))
    title = (title_el.text or "").strip() if title_el is not None else ""
    abstract = (abstract_el.text or "").strip() if abstract_el is not None else ""
    categories = (categories_el.text or "").strip() if categories_el is not None else ""
    created = (created_el.text or "").strip() if created_el is not None else ""
    updated = (updated_el.text or updated or created).strip()
    created_gmt = f"{created} 00:00:00 GMT" if created and "GMT" not in created else created
    return {
        "id": paper_id,
        "submitter": "",
        "authors": authors_to_string(arxiv.find(arxiv_tag("authors"))),
        "title": title,
        "comments": None,
        "journal-ref": None,
        "doi": None,
        "report-no": None,
        "categories": categories,
        "license": None,
        "abstract": abstract,
        "versions": [{"version": "v1", "created": created_gmt}],
        "update_date": updated[:10] if updated else created[:10],
        "authors_parsed": [],
    }


def load_batch_state(path: Path = DEFAULT_BATCH_STATE) -> dict:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_batch_state(state: dict, path: Path = DEFAULT_BATCH_STATE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)


def clear_staging(staging_path: Path = DEFAULT_STAGING_PATH, fetch_state_path: Path = DEFAULT_FETCH_STATE) -> None:
    staging_path.unlink(missing_ok=True)
    fetch_state_path.unlink(missing_ok=True)


def load_fetch_state(path: Path = DEFAULT_FETCH_STATE) -> dict:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_fetch_state(state: dict, path: Path = DEFAULT_FETCH_STATE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)


def append_staging_records(records: list[dict], staging_path: Path = DEFAULT_STAGING_PATH) -> None:
    staging_path.parent.mkdir(parents=True, exist_ok=True)
    with open(staging_path, "a", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def load_staging_records(staging_path: Path = DEFAULT_STAGING_PATH) -> list[dict]:
    if not staging_path.exists():
        return []
    records = []
    with open(staging_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _request_oai(url: str, timeout_sec: int, retries: int) -> bytes:
    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        req = urllib.request.Request(url, headers={"User-Agent": "RW-Cite-metadata-fetch/1.0"})
        try:
            with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
                return resp.read()
        except (urllib.error.URLError, TimeoutError, urllib.error.HTTPError) as exc:
            last_exc = exc
            print(
                f"OAI request failed (attempt {attempt}/{retries}): {type(exc).__name__}: {exc}",
                flush=True,
            )
            if attempt < retries:
                time.sleep(min(attempt * 5, 30))
    raise RuntimeError(f"OAI request failed after {retries} attempts: {last_exc}") from last_exc


def fetch_oai_metadata(
    since_date: str,
    until_date: str,
    *,
    max_records: int | None = None,
    timeout_sec: int = 180,
    retries: int = 5,
    page_sleep_sec: float = 3.0,
    resumption_token: str | None = None,
    staging_path: Path | None = DEFAULT_STAGING_PATH,
    fetch_state_path: Path = DEFAULT_FETCH_STATE,
    resume: bool = True,
) -> list[dict]:
    """Fetch metadata records from arXiv OAI-PMH with retry and optional resume."""
    records: list[dict] = []
    state = load_fetch_state(fetch_state_path) if resume else {}

    if resumption_token is None and resume and state.get("resumption_token"):
        if state.get("since_date") == since_date and state.get("until_date") == until_date:
            resumption_token = state["resumption_token"]
            records = load_staging_records(staging_path or DEFAULT_STAGING_PATH)
            print(
                f"Resuming OAI fetch from token, {len(records)} records already staged",
                flush=True,
            )

    if resumption_token:
        url = ARXIV_OAI + "?" + urllib.parse.urlencode(
            {"verb": "ListRecords", "resumptionToken": resumption_token}
        )
    else:
        params = {
            "verb": "ListRecords",
            "metadataPrefix": "arXiv",
            "from": iso_date(since_date),
            "until": iso_date(until_date),
        }
        url = ARXIV_OAI + "?" + urllib.parse.urlencode(params)

    page = 0
    while url:
        page += 1
        print(f"[page {page}] OAI fetch...", flush=True)
        payload = _request_oai(url, timeout_sec=timeout_sec, retries=retries)
        root = ET.fromstring(payload)

        error_el = root.find(f".//{oai_tag('error')}")
        if error_el is not None:
            code = error_el.get("code", "unknown")
            msg = (error_el.text or "").strip()
            # Empty window is normal for sparse days / dry-run smoke.
            if code == "noRecordsMatch":
                print(f"  OAI empty window ({code}): {msg or 'no records'}", flush=True)
                break
            raise RuntimeError(f"OAI error [{code}]: {msg}")

        page_records: list[dict] = []
        for record_el in root.findall(f".//{oai_tag('record')}"):
            header = record_el.find(oai_tag("header"))
            if header is not None and header.get("status") == "deleted":
                continue
            parsed = parse_oai_record(record_el)
            if parsed:
                page_records.append(parsed)

        records.extend(page_records)
        if staging_path is not None and page_records:
            append_staging_records(page_records, staging_path)

        print(
            f"  fetched {len(page_records)} this page, {len(records)} total",
            flush=True,
        )

        token_el = root.find(f".//{oai_tag('resumptionToken')}")
        next_token = token_el.text if token_el is not None and token_el.text else None
        if next_token:
            save_fetch_state(
                {
                    "since_date": since_date,
                    "until_date": until_date,
                    "resumption_token": next_token,
                    "records_fetched": len(records),
                    "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                },
                fetch_state_path,
            )
            url = ARXIV_OAI + "?" + urllib.parse.urlencode(
                {"verb": "ListRecords", "resumptionToken": next_token}
            )
            time.sleep(page_sleep_sec)
        else:
            url = None

        if max_records and len(records) >= max_records:
            records = records[:max_records]
            break

    if fetch_state_path.exists() and not url:
        fetch_state_path.unlink(missing_ok=True)

    return records


def merge_metadata(
    existing_path: Path | str,
    new_records: list[dict],
    *,
    dry_run: bool = False,
    backup: bool = True,
) -> tuple[int, int]:
    existing_path = Path(existing_path)
    existing = load_metadata_index(str(existing_path)) if existing_path.exists() else {}
    added = updated = 0
    for rec in new_records:
        pid = rec["id"]
        if pid in existing:
            if rec.get("update_date", "") >= existing[pid].get("update_date", ""):
                existing[pid] = rec
                updated += 1
        else:
            existing[pid] = rec
            added += 1
    if dry_run:
        return added, updated

    existing_path.parent.mkdir(parents=True, exist_ok=True)
    if backup and existing_path.exists():
        backup_path = existing_path.with_suffix(
            existing_path.suffix + f".bak.{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
        )
        shutil.copy2(existing_path, backup_path)
        print(f"Metadata backup: {backup_path}", flush=True)

    with open(existing_path, "w", encoding="utf-8") as f:
        for pid in sorted(existing.keys()):
            f.write(json.dumps(existing[pid], ensure_ascii=False) + "\n")
    return added, updated


def fetch_incremental_batches(
    output_path: Path | str = DEFAULT_META_PATH,
    *,
    since_date: str | None = None,
    until_date: str | None = None,
    batch_days: int = 31,
    max_batches: int | None = None,
    max_records: int | None = None,
    timeout_sec: int = 180,
    retries: int = 5,
    page_sleep_sec: float = 3.0,
    batch_sleep_sec: float = 5.0,
    staging_path: Path = DEFAULT_STAGING_PATH,
    fetch_state_path: Path = DEFAULT_FETCH_STATE,
    batch_state_path: Path = DEFAULT_BATCH_STATE,
    resume: bool = True,
    dry_run: bool = False,
) -> dict:
    """Fetch OAI metadata in date windows, merge after each window, until latest."""
    output_path = Path(output_path)
    until = until_date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    since = since_date or infer_next_since_date(output_path)

    if parse_date(since) > parse_date(until):
        print(f"Metadata already up to date (next since {since} > until {until})", flush=True)
        return {"batches": 0, "added": 0, "updated": 0, "fetched": 0}

    windows = iter_date_windows(since, until, batch_days)
    if max_batches is not None:
        windows = windows[:max_batches]

    overall_since = since
    print(
        f"Incremental metadata fetch: {overall_since} -> {until}, "
        f"{len(windows)} window(s), batch_days={batch_days}",
        flush=True,
    )

    total_added = total_updated = total_fetched = 0
    batches_done = 0

    for batch_idx, (win_from, win_until) in enumerate(windows, start=1):
        print(
            f"\n=== Metadata batch {batch_idx}/{len(windows)}: {win_from} -> {win_until} ===",
            flush=True,
        )

        records = fetch_oai_metadata(
            win_from,
            win_until,
            max_records=max_records,
            timeout_sec=timeout_sec,
            retries=retries,
            page_sleep_sec=page_sleep_sec,
            staging_path=staging_path,
            fetch_state_path=fetch_state_path,
            resume=resume,
        )
        total_fetched += len(records)
        print(f"Window fetch complete: {len(records)} records", flush=True)

        if dry_run:
            clear_staging(staging_path, fetch_state_path)
            batches_done += 1
            if batch_idx < len(windows):
                time.sleep(batch_sleep_sec)
            continue

        merge_source = load_staging_records(staging_path) if staging_path.exists() else records
        added, updated = merge_metadata(
            output_path,
            merge_source,
            dry_run=False,
            backup=(batch_idx == 1),
        )
        total_added += added
        total_updated += updated
        print(
            f"Merged window: added={added}, updated={updated}, source={len(merge_source)}",
            flush=True,
        )

        update_corpus_state_after_merge(overall_since, win_until, output_path)
        save_batch_state(
            {
                "overall_since": overall_since,
                "overall_until": until,
                "last_completed_until": win_until,
                "batches_completed": batch_idx,
                "records_fetched_total": total_fetched,
                "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            },
            batch_state_path,
        )
        clear_staging(staging_path, fetch_state_path)
        batches_done += 1

        if batch_idx < len(windows):
            time.sleep(batch_sleep_sec)

    if batches_done == len(windows) and parse_date(windows[-1][1]) >= parse_date(until):
        batch_state_path.unlink(missing_ok=True)
        print("Incremental fetch complete; cleared batch state", flush=True)

    summary = {
        "batches": batches_done,
        "added": total_added,
        "updated": total_updated,
        "fetched": total_fetched,
        "since": overall_since,
        "until": until if batches_done == len(windows) else windows[batches_done - 1][1],
    }
    print(
        f"\nIncremental summary: batches={summary['batches']}, fetched={summary['fetched']}, "
        f"added={summary['added']}, updated={summary['updated']}",
        flush=True,
    )
    return summary


def download_full_snapshot(
    output_path: Path | str = DEFAULT_META_PATH,
    url: str = DEFAULT_SNAPSHOT_URL,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".partial")
    print(f"Downloading full snapshot -> {output_path}", flush=True)
    print(f"URL: {url}", flush=True)
    req = urllib.request.Request(url, headers={"User-Agent": "RW-Cite-metadata-fetch/1.0"})
    with urllib.request.urlopen(req, timeout=600) as resp, open(tmp_path, "wb") as out:
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
    if output_path.exists():
        backup = output_path.with_suffix(
            output_path.suffix + f".bak.{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
        )
        shutil.copy2(output_path, backup)
        print(f"Existing metadata backup: {backup}", flush=True)
    tmp_path.replace(output_path)
    count = sum(1 for _ in open(output_path, encoding="utf-8"))
    print(f"Download complete: {count:,} records", flush=True)
    return output_path


def update_corpus_state_after_merge(since_date: str, until_date: str, metadata_path: Path | str) -> None:
    metadata_path = Path(metadata_path)
    total = sum(1 for _ in open(metadata_path, encoding="utf-8")) if metadata_path.exists() else 0
    state = load_corpus_state()
    state.update(
        {
            "last_metadata_fetch": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "last_metadata_from": since_date,
            "last_metadata_until": until_date,
            "metadata_total": total,
        }
    )
    save_corpus_state(state)
