"""Title-index lookup must survive LaTeX / unicode / SE(3) spelling drift."""

from __future__ import annotations

from rwcite.latex.utils import (
    _normalize_citation_title,
    citation_title_lookup_keys,
    resolve_citation_entry,
)


def test_latex_pi_and_unicode_pi_share_a_key():
    bib = r"$\pi_0.5$: a vision-language-action model with open-world generalization"
    meta = "$π_{0.5}$: a Vision-Language-Action Model with Open-World Generalization"
    assert _normalize_citation_title(bib) == _normalize_citation_title(meta)
    assert "pi" in _normalize_citation_title(bib)
    assert "0.5" in _normalize_citation_title(bib) or "0 5" in _normalize_citation_title(
        bib
    )


def test_escaped_underscore_matches_plain():
    bib = r"dm\_control: software and tasks for continuous control"
    meta = "dm_control: Software and Tasks for Continuous Control"
    assert _normalize_citation_title(bib) == _normalize_citation_title(meta)


def test_se3_spacing_variants_match():
    a = "Deep SE(3)-Equivariant Geometric Reasoning for Precise Placement Tasks"
    b = "deep se (3)-equivariant geometric reasoning for precise placement tasks"
    assert _normalize_citation_title(a) == _normalize_citation_title(b)


def test_math_k_modes_and_curly_apostrophe():
    bib = r"behavior transformers: cloning $ k $ modes with one stone"
    meta = r"Behavior Transformers: Cloning $k$ modes with one stone"
    assert _normalize_citation_title(bib) == _normalize_citation_title(meta)
    bib2 = "fastocc: accelerating 3d occupancy prediction by fusing the 2d bird’s-eye view"
    meta2 = "FastOcc: Accelerating 3D Occupancy Prediction by Fusing the 2D Bird's-Eye View"
    assert _normalize_citation_title(bib2) == _normalize_citation_title(meta2)


def test_resolve_via_improved_title_index():
    meta = "$π_{0.5}$: a Vision-Language-Action Model with Open-World Generalization"
    key = _normalize_citation_title(meta)
    index = {key: "2504.16054"}
    entry = {
        "title": r"$\pi_0.5$: a vision-language-action model with open-world generalization",
        "booktitle": "Proc. CoRL",
        "year": "2025",
    }
    assert resolve_citation_entry(entry, index, {"2504.16054"}) == "2504.16054"


def test_lookup_keys_include_dehyphenated_variant():
    keys = citation_title_lookup_keys("vision-language-action model for robots")
    assert any("vision language action" in k for k in keys)
