from __future__ import annotations

from rwcite.latex.cite_sentence_clean import (
    clean_cite_sentence,
    looks_like_garbage,
    strip_tex_comments,
)


def test_strip_tex_comments_keeps_escaped_percent():
    out = strip_tex_comments(r"about 100\% accurate % trailing")
    assert "accurate" in out
    assert "trailing" not in out
    assert "\\%" in out or "%" in out


def test_clean_removes_label_and_keeps_cite():
    raw = r"\label{sec:x} Prior work~\cite{a,b} showed gains."
    res = clean_cite_sentence(raw)
    assert res.ok
    assert "\\label" not in res.text
    assert "\\cite{" in res.text
    assert "Prior work" in res.text
    assert "a" in res.text and "b" in res.text


def test_clean_truncates_long_cite_key_list():
    keys = ",".join(f"k{i}" for i in range(12))
    raw = rf"See~\cite{{{keys}}} for details."
    res = clean_cite_sentence(raw, max_cite_keys=6)
    assert "cite_truncated" in res.reasons
    inner = res.text[res.text.index("{") + 1 : res.text.index("}")]
    assert len([k for k in inner.split(",") if k.strip()]) == 6


def test_garbage_empty_and_few_letters():
    assert "empty" in looks_like_garbage("")
    assert "too_few_letters" in looks_like_garbage("\\cite{x}")
