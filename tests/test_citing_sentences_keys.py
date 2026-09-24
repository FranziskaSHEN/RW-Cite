"""Sentence cleaning must not change which citations (and so which edges) exist."""

from __future__ import annotations

from rwcite.graph.graph_utils import get_citing_sentences


def _all_keys(content: str, *, clean: bool) -> set[str]:
    keys: set[str] = set()
    for _sent, ks in get_citing_sentences(content, clean=clean).items():
        keys.update(k.strip() for k in ks if k.strip())
    return keys


def test_long_cite_list_keeps_every_key_when_cleaning():
    # clean_cite_sentence caps \cite{...} at max_cite_keys=6; the keys used to be
    # read off that truncated text, which silently dropped graph edges.
    keys = [f"key{i}" for i in range(15)]
    content = "Recent work~\\cite{" + ", ".join(keys) + "} scaled this up."
    assert _all_keys(content, clean=True) == set(keys)
    assert _all_keys(content, clean=True) == _all_keys(content, clean=False)


def test_cite_group_past_max_len_survives_cleaning():
    # clean_cite_sentence also cuts text past max_len=600; a \cite group sitting
    # after that cut must still be resolved.
    filler = "robot learning and embodied navigation are studied widely " * 12
    content = f"{filler} and follow-up analyses~\\cite{{late_one,late_two}} agree."
    got = _all_keys(content, clean=True)
    assert {"late_one", "late_two"} <= got


def test_cleaning_still_cleans_the_stored_sentence():
    content = r"} \label{fig:x} \end{figure*} Prior work~\cite{a,b} showed gains."
    out = get_citing_sentences(content, clean=True)
    sent = next(s for s, ks in out.items() if ks)
    assert "\\label" not in sent
    assert not sent.lstrip().startswith("}")
    assert set(out[sent]) == {"a", "b"}


def test_spans_cleaning_to_same_text_merge_keys():
    # Two raw spans can normalise to the same text; keys must union, not overwrite.
    content = (
        r"\label{a} Prior work~\cite{one} helps. \label{b} Prior work~\cite{two} helps."
    )
    assert {"one", "two"} <= _all_keys(content, clean=True)
