from __future__ import annotations

import json
from pathlib import Path

from rwcite.retrieve.retriever.metadata_fetch import (
    iter_date_windows,
    iso_date,
    merge_metadata,
)


def test_iso_date_and_windows():
    assert iso_date("2024-01-15T12:00:00Z") == "2024-01-15"
    wins = iter_date_windows("2024-01-01", "2024-01-10", batch_days=3)
    assert wins[0] == ("2024-01-01", "2024-01-03")
    assert wins[-1][1] == "2024-01-10"
    assert sum(1 for _ in wins) == 4


def test_merge_metadata_add_and_update(tmp_path: Path):
    path = tmp_path / "snap.json"
    path.write_text(
        json.dumps(
            {
                "id": "2101.00001",
                "title": "Old",
                "abstract": "",
                "update_date": "2021-01-01",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    added, updated = merge_metadata(
        path,
        [
            {
                "id": "2101.00001",
                "title": "New",
                "abstract": "a",
                "update_date": "2021-06-01",
            },
            {
                "id": "2201.00002",
                "title": "Brand",
                "abstract": "b",
                "update_date": "2022-01-01",
            },
        ],
        backup=False,
    )
    assert added == 1 and updated == 1
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l]
    by_id = {r["id"]: r for r in rows}
    assert by_id["2101.00001"]["title"] == "New"
    assert "2201.00002" in by_id
