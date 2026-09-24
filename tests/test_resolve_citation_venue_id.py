"""arXiv ids hidden in venue fields must still resolve to an edge target."""

from __future__ import annotations

from rwcite.latex.utils import resolve_citation_entry

KNOWN = {"2410.24164", "2504.16054"}


def test_journal_field_arxiv_preprint_resolves():
    # Google Scholar's default export for preprints: no eprint field at all.
    entry = {
        "title": r"$\pi_0$: a vision-language-action flow model for general robot control",
        "author": "Kevin Black and others",
        "journal": "arXiv preprint arXiv:2410.24164",
        "year": "2026",
    }
    assert resolve_citation_entry(entry, {}, KNOWN) == "2410.24164"


def test_booktitle_field_arxiv_id_resolves():
    entry = {
        "title": "a vision-language-action model with open-world generalization",
        "booktitle": "arXiv:2504.16054",
    }
    assert resolve_citation_entry(entry, {}, KNOWN) == "2504.16054"


def test_venue_numbers_are_not_read_as_ids():
    # No arXiv marker in the field, so the volume/page digits must be ignored.
    entry = {
        "title": "some conference paper",
        "journal": "Robotics: Science and Systems",
        "volume": "2410.24164",
    }
    assert resolve_citation_entry(entry, {}, KNOWN) is None


def test_explicit_eprint_still_wins():
    entry = {
        "title": "whatever",
        "eprint": "2504.16054",
        "journal": "arXiv preprint arXiv:2410.24164",
    }
    assert resolve_citation_entry(entry, {}, KNOWN) == "2504.16054"
