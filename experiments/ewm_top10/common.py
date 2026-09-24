from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Iterator


ARXIV_NEW = re.compile(r"^(?P<yy>\d{2})(?P<mm>0[1-9]|1[0-2])\.\d{4,5}$")
ARXIV_OLD = re.compile(r"^[a-z-]+(?:\.[A-Z]{2})?/(?P<yy>\d{2})(?P<mm>0[1-9]|1[0-2])\d+$", re.I)


def normalize_id(value: Any) -> str:
    text = str(value or "").strip()
    text = text.removeprefix("https://arxiv.org/abs/")
    text = text.removeprefix("arXiv:")
    return re.sub(r"v\d+$", "", text, flags=re.I)


def arxiv_month(paper_id: str) -> tuple[int, int] | None:
    """Return an arXiv submission month; old IDs use the same YY pivot as arXiv."""
    pid = normalize_id(paper_id)
    match = ARXIV_NEW.match(pid) or ARXIV_OLD.match(pid)
    if not match:
        return None
    yy, month = int(match.group("yy")), int(match.group("mm"))
    year = 1900 + yy if yy >= 91 else 2000 + yy
    return year, month


def paper_month(paper_id: str, attrs: dict[str, Any] | None = None) -> tuple[int, int] | None:
    attrs = attrs or {}
    for key in ("published_at", "published", "date", "year_month"):
        value = str(attrs.get(key) or "").strip()
        match = re.search(r"(19\d{2}|20\d{2})[-/](0?[1-9]|1[0-2])", value)
        if match:
            return int(match.group(1)), int(match.group(2))
    year = attrs.get("year")
    if year is not None:
        try:
            return int(year), int(attrs.get("month") or 1)
        except (TypeError, ValueError):
            pass
    return arxiv_month(paper_id)


def month_string(value: tuple[int, int] | None) -> str:
    return f"{value[0]:04d}-{value[1]:02d}" if value else ""


def parse_month(value: str) -> tuple[int, int] | None:
    match = re.fullmatch(r"(19\d{2}|20\d{2})-(0[1-9]|1[0-2])", str(value or "").strip())
    return (int(match.group(1)), int(match.group(2))) if match else None


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{number}: {exc}") from exc


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def chronological_split(
    rows: list[dict[str, Any]], dev_ratio: float
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], tuple[int, int]]:
    """Split on a whole-month boundary, keeping unknown dates in training only."""
    if not 0.0 < dev_ratio < 1.0:
        raise ValueError("dev_ratio must be between 0 and 1")
    dated = [(row, parse_month(str(row.get("published_at") or ""))) for row in rows]
    counts = Counter(month for _, month in dated if month is not None)
    if len(counts) < 2:
        raise ValueError("need at least two distinct, parseable query months")
    target = max(1, round(len(rows) * dev_ratio))
    selected = 0
    cutoff = max(counts)
    for month in sorted(counts, reverse=True):
        cutoff = month
        selected += counts[month]
        if selected >= target:
            break
    train = [row for row, month in dated if month is None or month < cutoff]
    dev = [row for row, month in dated if month is not None and month >= cutoff]
    if not train or not dev:
        raise ValueError("chronological split produced an empty partition")
    return train, dev, cutoff


def query_text(row: dict[str, Any]) -> str:
    return f"Title: {row.get('title', '').strip()} Abstract: {row.get('abstract', '').strip()}".strip()


def candidate_text(row: dict[str, Any]) -> str:
    return query_text(row)


def unique_top_k(ids: Iterable[str], allowed: set[str], k: int) -> list[str]:
    out: list[str] = []
    for raw in ids:
        pid = normalize_id(raw)
        if pid and pid in allowed and pid not in out:
            out.append(pid)
        if len(out) == k:
            break
    return out
