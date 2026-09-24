from __future__ import annotations

import json
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from rwcite.runtime.services.survey_heuristics import SQL_SURVEY_TITLE_CLAUSE, is_survey_title

PAPER_COLUMNS = (
    "id",
    "title",
    "authors",
    "categories",
    "abstract",
    "comments",
    "journal_ref",
    "doi",
    "update_date",
    "topics_l1",
    "topics_l2",
    "topics_l3",
    "source",
    "synced_at",
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS papers (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL DEFAULT '',
    authors TEXT NOT NULL DEFAULT '',
    categories TEXT NOT NULL DEFAULT '',
    abstract TEXT NOT NULL DEFAULT '',
    comments TEXT,
    journal_ref TEXT,
    doi TEXT,
    update_date TEXT,
    topics_l1 TEXT NOT NULL DEFAULT '',
    topics_l2 TEXT NOT NULL DEFAULT '',
    topics_l3 TEXT NOT NULL DEFAULT '',
    source TEXT NOT NULL DEFAULT 'snapshot',
    synced_at TEXT
);

CREATE VIRTUAL TABLE IF NOT EXISTS papers_fts USING fts5(
    paper_id UNINDEXED,
    title,
    authors,
    categories,
    abstract,
    topics,
    tokenize='unicode61'
);

CREATE TABLE IF NOT EXISTS sync_state (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""

_FTS_TOKEN = re.compile(r"[\w.-]+", re.UNICODE)


def build_fts_query(raw: str) -> str | None:
    tokens = _FTS_TOKEN.findall(raw.strip())
    if not tokens:
        return None
    return " AND ".join(f'"{t}"' for t in tokens)


def _topics_to_text(topics: list | str | None) -> str:
    if not topics:
        return ""
    if isinstance(topics, str):
        return topics
    if isinstance(topics, list):
        return ", ".join(str(t) for t in topics)
    return str(topics)


def row_from_arxiv_entry(entry: dict, source: str = "snapshot") -> dict:
    now = datetime.now(timezone.utc).isoformat()
    return {
        "id": entry["id"],
        "title": entry.get("title") or "",
        "authors": entry.get("authors") or "",
        "categories": entry.get("categories") or "",
        "abstract": entry.get("abstract") or "",
        "comments": entry.get("comments") or "",
        "journal_ref": entry.get("journal-ref") or entry.get("journal_ref") or "",
        "doi": entry.get("doi") or "",
        "update_date": entry.get("update_date") or "",
        "topics_l1": entry.get("topics_l1") or "",
        "topics_l2": entry.get("topics_l2") or "",
        "topics_l3": entry.get("topics_l3") or "",
        "source": source,
        "synced_at": now,
    }


def row_from_gexf(paper_id: str, attrs: dict) -> dict:
    now = datetime.now(timezone.utc).isoformat()
    return {
        "id": paper_id,
        "title": attrs.get("title") or "",
        "authors": "",
        "categories": "",
        "abstract": attrs.get("abstract") or "",
        "comments": "",
        "journal_ref": "",
        "doi": "",
        "update_date": "",
        "topics_l1": "",
        "topics_l2": "",
        "topics_l3": "",
        "source": "gexf",
        "synced_at": now,
    }


def row_from_topics_supplement(entry: dict) -> dict:
    return {
        "id": entry["paper_id"],
        "topics_l1": _topics_to_text(entry.get("Level 1")),
        "topics_l2": _topics_to_text(entry.get("Level 2")),
        "topics_l3": _topics_to_text(entry.get("Level 3")),
        "source": "topics_supplement",
        "synced_at": datetime.now(timezone.utc).isoformat(),
    }


class MetadataStore:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)

    @property
    def ready(self) -> bool:
        return self.db_path.is_file() and self.db_path.stat().st_size > 0

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    @staticmethod
    def init_db(db_path: Path, reset: bool = False) -> sqlite3.Connection:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        if reset and db_path.exists():
            db_path.unlink()
        conn = sqlite3.connect(str(db_path))
        conn.executescript(SCHEMA)
        MetadataStore._migrate_schema(conn)
        conn.commit()
        return conn

    @staticmethod
    def _migrate_schema(conn: sqlite3.Connection) -> None:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(papers)").fetchall()}
        for col, typedef in (
            ("topics_l1", "TEXT NOT NULL DEFAULT ''"),
            ("topics_l2", "TEXT NOT NULL DEFAULT ''"),
            ("topics_l3", "TEXT NOT NULL DEFAULT ''"),
            ("source", "TEXT NOT NULL DEFAULT 'snapshot'"),
            ("synced_at", "TEXT"),
        ):
            if col not in cols:
                conn.execute(f"ALTER TABLE papers ADD COLUMN {col} {typedef}")
        conn.execute("DROP TRIGGER IF EXISTS papers_ai")
        conn.execute("DROP TRIGGER IF EXISTS papers_ad")
        conn.execute("DROP TRIGGER IF EXISTS papers_au")

    def _tune_for_bulk_load(self, conn: sqlite3.Connection) -> None:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=OFF")
        conn.execute("PRAGMA temp_store=MEMORY")
        conn.execute("PRAGMA cache_size=-512000")

    def _upsert_fts_rows(self, conn: sqlite3.Connection, paper_ids: list[str]) -> None:
        """Refresh FTS rows for the given paper IDs (delete + insert)."""
        if not paper_ids:
            return
        # Chunk to avoid SQLite variable limits
        chunk = 500
        for i in range(0, len(paper_ids), chunk):
            ids = paper_ids[i : i + chunk]
            placeholders = ",".join("?" for _ in ids)
            conn.execute(
                f"DELETE FROM papers_fts WHERE paper_id IN ({placeholders})",
                ids,
            )
            conn.execute(
                f"""
                INSERT INTO papers_fts(paper_id, title, authors, categories, abstract, topics)
                SELECT id, title, authors, categories, abstract,
                       topics_l1 || ' ' || topics_l2 || ' ' || topics_l3
                FROM papers WHERE id IN ({placeholders})
                """,
                ids,
            )

    def bulk_upsert_papers(self, rows: list[dict], sync_fts: bool = False) -> int:
        if not rows:
            return 0
        if not self.db_path.exists():
            self.init_db(self.db_path)
        placeholders = ",".join("?" for _ in PAPER_COLUMNS)
        updates = ",".join(f"{c}=excluded.{c}" for c in PAPER_COLUMNS if c != "id")
        sql = (
            f"INSERT INTO papers ({','.join(PAPER_COLUMNS)}) VALUES ({placeholders}) "
            f"ON CONFLICT(id) DO UPDATE SET {updates}"
        )
        ids = [row["id"] for row in rows if row.get("id")]
        conn = self.connect()
        try:
            self._migrate_schema(conn)
            self._tune_for_bulk_load(conn)
            conn.executemany(
                sql,
                [tuple(row.get(c, "") for c in PAPER_COLUMNS) for row in rows],
            )
            if sync_fts:
                self._upsert_fts_rows(conn, ids)
            conn.commit()
        finally:
            conn.close()
        return len(rows)

    def bulk_merge_topics(self, rows: list[dict], sync_fts: bool = False) -> int:
        """Upsert L1/L2/L3 topics; insert stub row if paper not yet in DB."""
        if not rows:
            return 0
        if not self.db_path.exists():
            self.init_db(self.db_path)
        sql = """
            INSERT INTO papers (id, topics_l1, topics_l2, topics_l3, source, synced_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                topics_l1=excluded.topics_l1,
                topics_l2=excluded.topics_l2,
                topics_l3=excluded.topics_l3,
                synced_at=excluded.synced_at
        """
        ids = [row["id"] for row in rows if row.get("id")]
        conn = self.connect()
        try:
            self._migrate_schema(conn)
            self._tune_for_bulk_load(conn)
            conn.executemany(
                sql,
                [
                    (
                        row["id"],
                        row.get("topics_l1") or "",
                        row.get("topics_l2") or "",
                        row.get("topics_l3") or "",
                        row.get("source") or "topics",
                        row.get("synced_at") or datetime.now(timezone.utc).isoformat(),
                    )
                    for row in rows
                ],
            )
            if sync_fts:
                self._upsert_fts_rows(conn, ids)
            conn.commit()
        finally:
            conn.close()
        return len(rows)

    def rebuild_fts(self) -> None:
        conn = self.connect()
        try:
            conn.execute("DELETE FROM papers_fts")
            conn.execute(
                """
                INSERT INTO papers_fts(paper_id, title, authors, categories, abstract, topics)
                SELECT id, title, authors, categories, abstract,
                       topics_l1 || ' ' || topics_l2 || ' ' || topics_l3
                FROM papers
                """
            )
            conn.commit()
        finally:
            conn.close()

    def upsert_rows(
        self, rows: list[dict], merge_topics_only: bool = False, sync_fts: bool = True
    ) -> tuple[int, int]:
        if not rows:
            return 0, 0
        if not self.db_path.exists():
            self.init_db(self.db_path)

        inserted = skipped = 0
        touched: list[str] = []
        conn = self.connect()
        try:
            self._migrate_schema(conn)
            for row in rows:
                pid = row["id"]
                existing = conn.execute(
                    "SELECT id, update_date, title, abstract, authors, categories, "
                    "comments, journal_ref, doi, topics_l1, topics_l2, topics_l3, source "
                    "FROM papers WHERE id = ?",
                    (pid,),
                ).fetchone()

                if merge_topics_only:
                    if existing:
                        conn.execute(
                            "UPDATE papers SET topics_l1=?, topics_l2=?, topics_l3=?, synced_at=? WHERE id=?",
                            (
                                row.get("topics_l1") or existing["topics_l1"],
                                row.get("topics_l2") or existing["topics_l2"],
                                row.get("topics_l3") or existing["topics_l3"],
                                row.get("synced_at"),
                                pid,
                            ),
                        )
                        inserted += 1
                        touched.append(pid)
                    else:
                        skipped += 1
                    continue

                merged: dict = {c: "" for c in PAPER_COLUMNS}
                if existing:
                    merged.update(dict(existing))
                for key in PAPER_COLUMNS:
                    if key == "id":
                        continue
                    val = row.get(key)
                    if val in (None, ""):
                        continue
                    if key == "update_date" and existing:
                        if val < (existing["update_date"] or ""):
                            continue
                    merged[key] = val

                if existing:
                    new_src = row.get("source") or existing["source"]
                    if existing["source"] == "gexf" and row.get("source") == "snapshot":
                        new_src = "snapshot+gexf"
                    merged["source"] = new_src
                else:
                    merged["source"] = row.get("source", "snapshot")

                merged["id"] = pid
                merged.setdefault("synced_at", datetime.now(timezone.utc).isoformat())

                placeholders = ",".join("?" for _ in PAPER_COLUMNS)
                updates = ",".join(f"{c}=excluded.{c}" for c in PAPER_COLUMNS if c != "id")
                conn.execute(
                    f"INSERT INTO papers ({','.join(PAPER_COLUMNS)}) VALUES ({placeholders}) "
                    f"ON CONFLICT(id) DO UPDATE SET {updates}",
                    tuple(merged.get(c, "") for c in PAPER_COLUMNS),
                )
                inserted += 1
                touched.append(pid)
            if sync_fts:
                self._upsert_fts_rows(conn, touched)
            conn.commit()
        finally:
            conn.close()
        return inserted, skipped

    def existing_ids(self) -> set[str]:
        if not self.ready:
            return set()
        with self.connect() as conn:
            return {r[0] for r in conn.execute("SELECT id FROM papers")}

    def set_sync_state(self, key: str, value: str) -> None:
        conn = self.connect()
        try:
            conn.execute(
                "INSERT INTO sync_state(key, value) VALUES(?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, value),
            )
            conn.commit()
        finally:
            conn.close()

    def get_sync_state(self, key: str) -> str | None:
        if not self.ready:
            return None
        conn = self.connect()
        try:
            row = conn.execute("SELECT value FROM sync_state WHERE key=?", (key,)).fetchone()
            return row["value"] if row else None
        finally:
            conn.close()

    def stats(self) -> dict:
        if not self.ready:
            return {"paper_count": 0, "by_source": {}, "with_topics": 0}
        conn = self.connect()
        try:
            total = conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
            by_source = dict(
                conn.execute(
                    "SELECT source, COUNT(*) FROM papers GROUP BY source ORDER BY COUNT(*) DESC"
                ).fetchall()
            )
            with_topics = conn.execute(
                "SELECT COUNT(*) FROM papers WHERE topics_l1 != '' OR topics_l2 != '' OR topics_l3 != ''"
            ).fetchone()[0]
        finally:
            conn.close()
        return {
            "paper_count": total,
            "with_topics": with_topics,
            "by_source": by_source,
            "last_sync": self.get_sync_state("last_sync"),
        }

    def get(self, paper_id: str) -> dict | None:
        if not self.ready:
            return None
        conn = self.connect()
        try:
            row = conn.execute(
                f"SELECT {','.join(PAPER_COLUMNS)} FROM papers WHERE id = ?",
                (paper_id,),
            ).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def get_many(self, paper_ids: list[str]) -> dict[str, dict]:
        if not self.ready or not paper_ids:
            return {}
        placeholders = ",".join("?" for _ in paper_ids)
        conn = self.connect()
        try:
            rows = conn.execute(
                f"SELECT {','.join(PAPER_COLUMNS)} FROM papers WHERE id IN ({placeholders})",
                paper_ids,
            ).fetchall()
            return {row["id"]: dict(row) for row in rows}
        finally:
            conn.close()

    def search(
        self,
        query: str,
        allowed_ids: set[str] | None = None,
        offset: int = 0,
        limit: int = 20,
        sort: str = "relevance",
        year_from: int | None = None,
        year_to: int | None = None,
        survey_filter: bool = False,
        window: int = 1000,
    ) -> tuple[list[tuple[str, float]], int]:
        if not self.ready:
            return [], 0
        fts_q = build_fts_query(query)
        if not fts_q:
            return [], 0

        need_post = (
            sort in ("recency", "title")
            or year_from is not None
            or year_to is not None
            or survey_filter
        )
        conn = self.connect()
        try:
            if allowed_ids is None and not need_post:
                total = conn.execute(
                    "SELECT COUNT(*) FROM papers_fts WHERE papers_fts MATCH ?",
                    (fts_q,),
                ).fetchone()[0]
                rows = conn.execute(
                    "SELECT paper_id, bm25(papers_fts) AS rank "
                    "FROM papers_fts WHERE papers_fts MATCH ? "
                    "ORDER BY rank LIMIT ? OFFSET ?",
                    (fts_q, limit, offset),
                ).fetchall()
                matched = [(r["paper_id"], float(r["rank"])) for r in rows]
                return matched, total

            fetch_limit = min(max(window, limit * (50 if survey_filter else 20)), 5000)
            rows = conn.execute(
                "SELECT paper_id, bm25(papers_fts) AS rank "
                "FROM papers_fts WHERE papers_fts MATCH ? "
                "ORDER BY rank LIMIT ?",
                (fts_q, fetch_limit),
            ).fetchall()
            matched = [(r["paper_id"], float(r["rank"])) for r in rows]
            if allowed_ids is not None:
                matched = [m for m in matched if m[0] in allowed_ids]

            if year_from is not None or year_to is not None:
                ids = [pid for pid, _ in matched]
                meta = self._year_map(conn, ids)
                filtered = []
                for pid, rank in matched:
                    y = meta.get(pid)
                    if y is None:
                        continue
                    if year_from is not None and y < year_from:
                        continue
                    if year_to is not None and y > year_to:
                        continue
                    filtered.append((pid, rank))
                matched = filtered

            if survey_filter:
                titles = self._title_map(conn, [pid for pid, _ in matched])
                matched = [
                    (pid, rank)
                    for pid, rank in matched
                    if is_survey_title(titles.get(pid) or "")
                ]

            if sort == "recency":
                meta = self._date_map(conn, [pid for pid, _ in matched])
                matched.sort(key=lambda x: (meta.get(x[0]) or "", x[0]), reverse=True)
            elif sort == "title":
                titles = self._title_map(conn, [pid for pid, _ in matched])
                matched.sort(key=lambda x: (titles.get(x[0]) or "", x[0]))
            # else keep BM25 order (rank ascending = more relevant for bm25)

            total = len(matched)
            return matched[offset : offset + limit], total
        finally:
            conn.close()

    @staticmethod
    def _year_from_row(update_date: str | None, paper_id: str) -> int | None:
        if update_date and len(update_date) >= 4 and update_date[:4].isdigit():
            return int(update_date[:4])
        m = re.match(r"^(\d{2})(\d{2})\.", paper_id)
        if m:
            yy = int(m.group(1))
            return 1900 + yy if yy > 70 else 2000 + yy
        return None

    def _year_map(self, conn: sqlite3.Connection, ids: list[str]) -> dict[str, int]:
        out: dict[str, int] = {}
        for i in range(0, len(ids), 500):
            chunk = ids[i : i + 500]
            ph = ",".join("?" for _ in chunk)
            for row in conn.execute(
                f"SELECT id, update_date FROM papers WHERE id IN ({ph})", chunk
            ):
                y = self._year_from_row(row["update_date"], row["id"])
                if y is not None:
                    out[row["id"]] = y
        return out

    def _date_map(self, conn: sqlite3.Connection, ids: list[str]) -> dict[str, str]:
        out: dict[str, str] = {}
        for i in range(0, len(ids), 500):
            chunk = ids[i : i + 500]
            ph = ",".join("?" for _ in chunk)
            for row in conn.execute(
                f"SELECT id, update_date FROM papers WHERE id IN ({ph})", chunk
            ):
                out[row["id"]] = row["update_date"] or ""
        return out

    def _title_map(self, conn: sqlite3.Connection, ids: list[str]) -> dict[str, str]:
        out: dict[str, str] = {}
        for i in range(0, len(ids), 500):
            chunk = ids[i : i + 500]
            ph = ",".join("?" for _ in chunk)
            for row in conn.execute(
                f"SELECT id, title FROM papers WHERE id IN ({ph})", chunk
            ):
                out[row["id"]] = (row["title"] or "").lower()
        return out

    def browse(
        self,
        offset: int = 0,
        limit: int = 20,
        sort: str = "recency",
        year_from: int | None = None,
        year_to: int | None = None,
        survey_filter: bool = False,
    ) -> tuple[list[str], int]:
        if not self.ready:
            return [], 0
        order = "update_date DESC, id DESC"
        if sort == "title":
            order = "title ASC, id ASC"
        where = []
        params: list = []
        if year_from is not None:
            where.append("substr(COALESCE(update_date,''), 1, 4) >= ?")
            params.append(str(year_from))
        if year_to is not None:
            where.append("substr(COALESCE(update_date,''), 1, 4) <= ?")
            params.append(str(year_to))
        if survey_filter:
            where.append(SQL_SURVEY_TITLE_CLAUSE)
        clause = f"WHERE {' AND '.join(where)}" if where else ""
        conn = self.connect()
        try:
            total = conn.execute(f"SELECT COUNT(*) FROM papers {clause}", params).fetchone()[0]
            rows = conn.execute(
                f"SELECT id FROM papers {clause} ORDER BY {order} LIMIT ? OFFSET ?",
                [*params, limit, offset],
            ).fetchall()
        finally:
            conn.close()
        return [r["id"] for r in rows], total

    def year_histogram(
        self,
        year_from: int | None = None,
        year_to: int | None = None,
        paper_ids: list[str] | None = None,
    ) -> list[dict]:
        """Return [{year, count}, ...] for sidebar chart."""
        if not self.ready:
            return []
        conn = self.connect()
        try:
            if paper_ids is not None:
                if not paper_ids:
                    return []
                counts: dict[int, int] = {}
                for i in range(0, len(paper_ids), 500):
                    chunk = paper_ids[i : i + 500]
                    ph = ",".join("?" for _ in chunk)
                    for row in conn.execute(
                        f"SELECT id, update_date FROM papers WHERE id IN ({ph})", chunk
                    ):
                        y = self._year_from_row(row["update_date"], row["id"])
                        if y is None:
                            continue
                        if year_from is not None and y < year_from:
                            continue
                        if year_to is not None and y > year_to:
                            continue
                        counts[y] = counts.get(y, 0) + 1
                return [{"year": y, "count": counts[y]} for y in sorted(counts)]

            where = ["update_date IS NOT NULL", "length(update_date) >= 4"]
            params: list = []
            if year_from is not None:
                where.append("substr(update_date, 1, 4) >= ?")
                params.append(str(year_from))
            if year_to is not None:
                where.append("substr(update_date, 1, 4) <= ?")
                params.append(str(year_to))
            rows = conn.execute(
                f"""
                SELECT substr(update_date, 1, 4) AS y, COUNT(*) AS c
                FROM papers
                WHERE {' AND '.join(where)}
                GROUP BY y
                ORDER BY y
                """,
                params,
            ).fetchall()
            out = []
            for r in rows:
                if r["y"] and str(r["y"]).isdigit():
                    out.append({"year": int(r["y"]), "count": int(r["c"])})
            return out
        finally:
            conn.close()

    def count(self) -> int:
        return self.stats()["paper_count"]
