"""Title heuristics for survey/review papers (Phase A; no Neo4j Survey label)."""

from __future__ import annotations

import re

# Prefer "a survey …" / "survey on" over bare "survey of" (astronomy FPs).
SURVEY_TITLE_RE = re.compile(
    r"(?is)(?:"
    r"\ba\s+survey\b|"
    r"\bsurvey\s+on\b|"
    r"\bsystematic\s+review\b|"
    r"\bliterature\s+review\b|"
    r"\bcomprehensive\s+review\b|"
    r"\bcritical\s+review\b"
    r")"
)

# SQLite clause mirroring the regex for browse / SQL filters.
SQL_SURVEY_TITLE_CLAUSE = """(
  lower(COALESCE(title, '')) LIKE 'a survey%'
  OR lower(COALESCE(title, '')) LIKE '% a survey%'
  OR lower(COALESCE(title, '')) LIKE '%survey on%'
  OR lower(COALESCE(title, '')) LIKE '%systematic review%'
  OR lower(COALESCE(title, '')) LIKE '%literature review%'
  OR lower(COALESCE(title, '')) LIKE '%comprehensive review%'
  OR lower(COALESCE(title, '')) LIKE '%critical review%'
)"""


def is_survey_title(title: str | None) -> bool:
    if not title:
        return False
    return bool(SURVEY_TITLE_RE.search(title.replace("\n", " ")))
