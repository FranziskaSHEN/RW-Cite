"""Clean RR / domain-graph cite sentences extracted from LaTeX.

Used by graph edge write, offline GEXF rewrite, and Cite dataset build.
Goal: strip structural LaTeX debris while keeping readable cite context
(and optional ``\\cite{...}`` macros, with key-list truncation).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


_COMMENT_RE = re.compile(r"(?<!\\)%[^\n]*")
_FIGURE_ENV_RE = re.compile(
    r"\\begin\{figure\*?\}.*?\\end\{figure\*?\}",
    re.IGNORECASE | re.DOTALL,
)
_LABEL_RE = re.compile(r"\\label\*?\{[^{}]*\}")
_VSPACE_RE = re.compile(r"\\(?:v|h)space\*?\{[^{}]*\}")
_IEEE_RE = re.compile(
    r"\\IEEEPARstart\s*\{([^{}]*)\}\s*\{([^{}]*)\}",
    re.IGNORECASE,
)
_SECTION_RE = re.compile(
    r"\\(?:sub)*section\*?\{([^{}]*)\}",
    re.IGNORECASE,
)
_BEGIN_END_RE = re.compile(
    r"\\(?:begin|end)\{[^{}]+\}",
    re.IGNORECASE,
)
_SIMPLE_MACRO_RE = re.compile(
    r"\\(?:noindent|centering|hfill|vfill|bigskip|medskip|smallskip|newpage|clearpage|"
    r"textwidth|linewidth|columnwidth|relax|protect|footnotesize|scriptsize|"
    r"tiny|small|normalsize|large|Large|LARGE|huge|Huge|FloatBarrier|bfseries|textit|"
    r"textbf|emph|textit)\b"
)
_CITE_RE = re.compile(r"\\cite\*?\{([^{}]*)\}")
_LEADING_JUNK_RE = re.compile(
    r"^(?:"
    r"[\}\{\[\]]|"
    r"\\end\{[^{}]+\}|"
    r"\\begin\{[^{}]+\}|"
    r"\\vspace\*?\{[^{}]*\}|"
    r"\\hspace\*?\{[^{}]*\}|"
    r"\\label\*?\{[^{}]*\}|"
    r"\\(?:sub)*section\*?\{[^{}]*\}|"
    r"\\item\b|"
    r"%[^\n]*"
    r")\s*",
    re.IGNORECASE,
)
_MULTI_WS_RE = re.compile(r"\s+")
_ALPHA_RE = re.compile(r"[A-Za-z]")


@dataclass
class CleanResult:
    text: str
    ok: bool
    reasons: list[str] = field(default_factory=list)
    raw_len: int = 0
    clean_len: int = 0

    @property
    def is_garbage(self) -> bool:
        return not self.ok


def strip_tex_comments(s: str) -> str:
    """Remove TeX ``%`` comments; keep ``\\%``."""
    return _COMMENT_RE.sub(" ", s or "")


def _recover_from_percent_comments(raw: str) -> tuple[str, bool]:
    """If live TeX is junk but a ``%``-commented span holds ``\\cite`` prose, recover it."""
    parts = re.split(r"(?<!\\)%", raw or "")
    if len(parts) <= 1:
        return raw or "", False
    scored: list[tuple[int, int, str]] = []
    for i, p in enumerate(parts):
        t = (p or "").strip()
        if not t:
            continue
        score = (10 if "\\cite" in t else 0) + min(50, len(_ALPHA_RE.findall(t)))
        scored.append((score, i, t))
    if not scored:
        return raw or "", False
    scored.sort(reverse=True)
    best_score, _i, best = scored[0]
    live = (parts[0] or "").strip()
    live_letters = len(_ALPHA_RE.findall(live))
    if best_score >= 15 and (live_letters < 8 or "\\cite" not in live):
        return best, True
    return raw or "", False


def _truncate_cite_macros(s: str, *, max_keys: int) -> tuple[str, bool]:
    truncated = False

    def _repl(m: re.Match[str]) -> str:
        nonlocal truncated
        keys = [k.strip() for k in m.group(1).split(",") if k.strip()]
        if len(keys) <= max_keys:
            return m.group(0)
        truncated = True
        return "\\cite{" + ", ".join(keys[:max_keys]) + "}"

    return _CITE_RE.sub(_repl, s), truncated


def _strip_leading_junk(s: str) -> str:
    prev = None
    while prev != s:
        prev = s
        s = _LEADING_JUNK_RE.sub("", s)
    return s


def looks_like_garbage(s: str) -> list[str]:
    """Heuristic flags on a (possibly already cleaned) string."""
    reasons: list[str] = []
    t = (s or "").strip()
    if not t:
        reasons.append("empty")
        return reasons
    if t.startswith("}") or t.startswith("%"):
        reasons.append("lead_junk")
    if "\\label" in t:
        reasons.append("label")
    if re.search(r"\\(?:sub)*section\b", t, re.I):
        reasons.append("section")
    if re.search(r"\\(?:begin|end)\{figure", t, re.I):
        reasons.append("figure")
    if re.search(r"\\IEEEPARstart\b", t, re.I):
        reasons.append("ieeeparstart")
    if len(t) > 900:
        reasons.append("too_long")
    cites = _CITE_RE.findall(t)
    if any(len([k for k in c.split(",") if k.strip()]) > 12 for c in cites):
        reasons.append("cite_spam")
    if len(_ALPHA_RE.findall(t)) < 8:
        reasons.append("too_few_letters")
    return reasons


def clean_cite_sentence(
    raw: str,
    *,
    max_len: int = 600,
    max_cite_keys: int = 6,
    keep_section_title_text: bool = False,
) -> CleanResult:
    """Return cleaned cite sentence.

    ``ok=False`` means the span should not be used as Cite SFT gold
    (edge may still store ``text`` as best-effort for CE-sent coverage).
    """
    raw = raw or ""
    reasons: list[str] = []
    recovered, did_recover = _recover_from_percent_comments(raw)
    if did_recover:
        raw = recovered
        reasons.append("uncommented")
    s = strip_tex_comments(raw)
    # Entire span was a TeX comment line — recover body (common in RW drafts).
    if not s.strip() and raw.lstrip().startswith("%"):
        s = raw.lstrip()[1:]
        s = strip_tex_comments(s)
        reasons.append("uncommented")
    s = _FIGURE_ENV_RE.sub(" ", s)
    s = _LABEL_RE.sub(" ", s)
    s = _VSPACE_RE.sub(" ", s)
    s = _IEEE_RE.sub(lambda m: f"{m.group(1)}{m.group(2)}", s)
    if keep_section_title_text:
        s = _SECTION_RE.sub(lambda m: f" {m.group(1)} ", s)
    else:
        s = _SECTION_RE.sub(" ", s)
    s = _BEGIN_END_RE.sub(" ", s)
    s = _SIMPLE_MACRO_RE.sub(" ", s)
    # Drop leftover TeX commands with one optional brace arg (non-cite).
    s = re.sub(r"\\(?!cite\b)[A-Za-z@]+\*?(\[[^\]]*\])?(\{[^{}]*\})?", " ", s)
    s = _strip_leading_junk(s)
    # Protect \\cite{...} while stripping orphan braces from broken envs.
    cite_slots: list[str] = []

    def _park_cite(m: re.Match[str]) -> str:
        cite_slots.append(m.group(0))
        return f" __CITE{len(cite_slots) - 1}__ "

    s = _CITE_RE.sub(_park_cite, s)
    s = s.replace("{", " ").replace("}", " ")
    for i, c in enumerate(cite_slots):
        s = s.replace(f"__CITE{i}__", c)
    s = re.sub(r"~\s+(\\cite\b)", r"~\1", s)
    s = _MULTI_WS_RE.sub(" ", s).strip(" \t\n\r-–—:;,. ")
    # Re-attach a terminal period if we stripped it and body remains.
    if s and not s.endswith((".", "?", "!")):
        s = s + "."
    s, did_trunc = _truncate_cite_macros(s, max_keys=max_cite_keys)
    if did_trunc:
        reasons.append("cite_truncated")
    if len(s) > max_len:
        # Prefer cutting after a sentence boundary inside the budget.
        cut = s[:max_len]
        for sep in (". ", "? ", "! "):
            idx = cut.rfind(sep)
            if idx >= int(max_len * 0.4):
                cut = cut[: idx + 1]
                break
        s = cut.rstrip()
        if s and not s.endswith((".", "?", "!")):
            s = s + "."
        reasons.append("len_truncated")

    flags = looks_like_garbage(s)
    # lead_junk / label / figure after cleaning => still bad
    hard = {"empty", "lead_junk", "label", "section", "figure", "too_few_letters"}
    hard_hits = [f for f in flags if f in hard]
    reasons.extend(hard_hits)
    ok = not hard_hits and len(s) >= 15
    if not ok and "rejected" not in reasons:
        reasons.append("rejected")
    return CleanResult(text=s if ok or len(s) >= 15 else "", ok=ok, reasons=reasons, raw_len=len(raw), clean_len=len(s))


def is_garbage_cite_sentence(raw: str) -> bool:
    return clean_cite_sentence(raw).is_garbage


def garbage_flags(raw: str) -> list[str]:
    """Flags on *raw* (pre-clean) for reporting."""
    return looks_like_garbage(raw or "")


def _self_check() -> None:
    cases = [
        (
            r"} \label{fig:paradigm} \end{figure*} \subsection{Nav} LLMs navigate~\cite{a,b}.",
            True,
        ),
        (
            r"% \item We extend AI2-THOR~\cite{kolve2017ai2} with stairs.",
            True,
        ),
        (
            r"\label{sec:intro} Humanoid robots~\cite{gu2025humanoid} walk.",
            True,
        ),
        (
            r"\IEEEPARstart{R}{obot} navigation~\cite{x} is hard.",
            True,
        ),
        (
            "Recent VLA models expand language-conditioned learning "
            r"\cite{" + ", ".join(f"k{i}" for i in range(40)) + "}.",
            True,
        ),
        ("}", False),
        ("", False),
    ]
    for raw, expect_ok in cases:
        r = clean_cite_sentence(raw)
        assert r.ok == expect_ok, (raw[:80], r)
        if expect_ok:
            assert "\\label" not in r.text
            assert "\\subsection" not in r.text
            assert not r.text.startswith("}")
            assert not r.text.startswith("%")
    # cite truncation
    long_cite = r"Work~\cite{" + ", ".join(f"a{i}" for i in range(20)) + "} shows X."
    r = clean_cite_sentence(long_cite)
    assert r.ok
    keys = _CITE_RE.search(r.text)
    assert keys is not None
    assert len([k for k in keys.group(1).split(",") if k.strip()]) <= 6
    print("cite_sentence_clean self-check OK")


if __name__ == "__main__":
    _self_check()
