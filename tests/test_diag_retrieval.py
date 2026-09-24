from __future__ import annotations

from pathlib import Path

from rwcite.cli.diag_retrieval import zip_coverage, zip_has


def test_zip_has_matches_tar_gz(tmp_path: Path):
    zdir = tmp_path / "research_papers_zip"
    zdir.mkdir()
    (zdir / "2101.00001.tar.gz").write_bytes(b"x")
    assert zip_has(zdir, "2101.00001")
    assert not zip_has(zdir, "9999.99999")


def test_zip_coverage_counts_hits(tmp_path: Path):
    zdir = tmp_path / "research_papers_zip"
    zdir.mkdir()
    (zdir / "2101.00001.tar.gz").write_bytes(b"x")
    (zdir / "2101.00002.tar.gz").write_bytes(b"y")
    stats = zip_coverage(["2101.00001", "2101.00003"], zdir)
    assert stats["zip_cache_total"] == 2
    assert stats["retrieval_in_zip"] == 1
    assert stats["retrieval_in_zip_frac"] == 0.5
