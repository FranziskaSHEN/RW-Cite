"""Reference Recommend: prompt build, parse, align, and v2 Stage A/B helpers."""

from __future__ import annotations

import hashlib
import os
import random
import re
from typing import Any

INSTRUCTION = (
    "Recommend exactly {k} papers from the candidate list that Paper A is most "
    "likely to cite. For each paper, copy the Title exactly as listed in the "
    "candidates, fill arXiv id, and write one Cite sentence describing how "
    "Paper A would cite it. Output in the required Markdown format only."
)

INSTRUCTION_SELECT = (
    "Select exactly {k} papers from the candidate list that Paper A is most "
    "likely to cite. Output only the arXiv ids, one per line, with no other text. "
    "Each id must appear in the candidate list."
)

INSTRUCTION_SELECT_RANK_AWARE = (
    "Select exactly {k} papers from the candidate list that Paper A is most "
    "likely to cite. Output only the arXiv ids, one per line, with no other text. "
    "Each id must appear in the candidate list. Treat the listed rank as a prior: "
    "prefer higher-ranked candidates, and only promote a lower-ranked paper when "
    "it is clearly more relevant by title/abstract meaning."
)

INSTRUCTION_SELECT_ABS = (
    "Select exactly {k} papers from the candidate list that Paper A is most "
    "likely to cite, based on the meaning of each candidate's title and "
    "abstract relative to Paper A. Output only the arXiv ids, one per line, "
    "with no other text. Each id must appear in the candidate list. "
    "Ranks or retrieval scores are not provided; do not assume list order "
    "reflects relevance."
)

INSTRUCTION_SHORTLIST_TITLE = (
    "From the candidate list (titles only), select exactly {k} papers that "
    "Paper A is most likely to cite. Rank them from most to least relevant. "
    "Output only the arXiv ids, one per line, with no other text. Each id must "
    "appear in the candidate list. Do not assume list order reflects relevance."
)

INSTRUCTION_RERANK_ABS = (
    "Reorder the candidate papers by how likely Paper A is to cite each one, "
    "using title and abstract meaning relative to Paper A. Output exactly {k} "
    "arXiv ids, most relevant first, one per line, with no other text. Each id "
    "must appear in the candidate list. Do not assume display order reflects "
    "relevance."
)

INSTRUCTION_CITE_BUNDLE = (
    "For each selected paper below, write one Cite sentence describing how "
    "Paper A would cite it. Each Cite must be distinct: do not reuse or lightly "
    "paraphrase the same sentence across papers; ground each sentence in that "
    "paper's title and abstract. Copy Title and arXiv exactly. Output in the "
    "required Markdown format only."
)

INSTRUCTION_CITE_ONE = (
    "Write one Cite sentence describing how Paper A would cite the selected "
    "paper. Output only the sentence, with no title or labels."
)

INSTRUCTION_POINTWISE_CITE = (
    "Decide whether Paper A should cite the candidate paper, based on title "
    "and abstract meaning. Output exactly one token: cite or skip. No other text."
)

INSTRUCTION_POINTWISE_PREFER = (
    "Decide which of the two candidates Paper A should cite, based on title "
    "and abstract meaning. Output exactly one token: 1 or 2. No other text."
)

INSTRUCTION_ANALYZE = (
    "You are a sophisticated researcher analyzing why Paper A might cite a "
    "candidate paper. Based on titles and abstracts, judge core-citation "
    "likelihood. Output exactly two lines in this format (no other text):\n"
    "Relevance: <core|related|peripheral|none>\n"
    "Reason for Citation: <one short sentence, under 40 words>"
)

INSTRUCTION_RERANK_ANALYSIS = (
    "Reorder the candidate papers by how likely Paper A is to cite each one "
    "as a core citation, using the provided Relevance/Reason analyses. "
    "Output exactly {k} arXiv ids, most relevant first, one per line, with "
    "no other text. Each id must appear in the candidate list. Do not assume "
    "display order reflects relevance."
)

OUTPUT_HEADER = "### Recommended References"

_STOP = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "of",
    "for",
    "in",
    "on",
    "to",
    "with",
    "from",
    "by",
    "is",
    "are",
    "as",
    "at",
    "via",
    "into",
    "using",
    "based",
    "towards",
    "toward",
}


def select_abs_mode(candidates: list[dict[str, Any]] | None = None) -> bool:
    """v3.2: candidate title+abstract in select prompts. Overrides rank-aware."""
    env = (os.environ.get("RR_SELECT_ABS") or "").strip().lower()
    if env in ("1", "true", "yes", "on"):
        return True
    if env in ("0", "false", "no", "off"):
        return False
    if not candidates:
        return False
    return any(bool(c.get("select_abs")) for c in candidates)


def select_abs_chars() -> int:
    raw = (os.environ.get("RR_SELECT_ABS_CHARS") or "300").strip()
    try:
        return max(50, int(raw))
    except ValueError:
        return 300


def select_rank_aware(candidates: list[dict[str, Any]] | None = None) -> bool:
    """Rank/score fields in select prompts (v3.1). Env overrides auto-detect."""
    if select_abs_mode(candidates):
        return False
    env = (os.environ.get("RR_SELECT_RANK_AWARE") or "").strip().lower()
    if env in ("1", "true", "yes", "on"):
        return True
    if env in ("0", "false", "no", "off"):
        return False
    if not candidates:
        return False
    return any(c.get("rank") is not None for c in candidates)


def annotate_candidate_ranks(
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Ensure 1-based rank from list order (ranker order)."""
    out: list[dict[str, Any]] = []
    for i, c in enumerate(candidates):
        d = dict(c)
        if d.get("rank") is None:
            d["rank"] = i + 1
        out.append(d)
    return out


def stable_shuffle_candidates(
    candidates: list[dict[str, Any]],
    source_id: str,
) -> list[dict[str, Any]]:
    """Deterministic shuffle by source_id (v3.2: break position=rank shortcut)."""
    items = [dict(c) for c in candidates]
    seed = int(hashlib.sha1(str(source_id).encode("utf-8")).hexdigest()[:8], 16)
    random.Random(seed).shuffle(items)
    return items


def format_candidates(candidates: list[dict[str, str]]) -> str:
    abs_mode = select_abs_mode(candidates)
    rank_aware = select_rank_aware(candidates)
    abs_chars = select_abs_chars()
    lines = []
    for i, c in enumerate(candidates, start=1):
        cid = (c.get("id") or "").strip()
        title = (c.get("title") or "").strip()
        if abs_mode:
            title = title[:180]
            abs_ = (c.get("abstract") or "").strip().replace("\n", " ")[:abs_chars]
            if abs_:
                lines.append(f"{i}. {cid} | {title} | Abstract: {abs_}")
            else:
                lines.append(f"{i}. {cid} | {title}")
            continue
        if not rank_aware:
            lines.append(f"{i}. {cid} | {title}")
            continue
        rank = c.get("rank") if c.get("rank") is not None else i
        score = c.get("score")
        if score is not None and score != "":
            try:
                score_s = f"{float(score):.4f}"
            except (TypeError, ValueError):
                score_s = str(score)
            lines.append(f"{i}. {cid} | {title} | rank={rank} | score={score_s}")
        else:
            lines.append(f"{i}. {cid} | {title} | rank={rank}")
    return "\n".join(lines)


def format_gold_output(gold: list[dict[str, str]]) -> str:
    blocks = [OUTPUT_HEADER]
    for i, g in enumerate(gold, start=1):
        blocks.append(
            f"{i}. Title: {(g.get('title') or '').strip()}\n"
            f"   arXiv: {(g.get('id') or '').strip()}\n"
            f"   Cite: {(g.get('sentence') or '').strip()}"
        )
    return "\n".join(blocks)


def format_selected_for_cite(selected: list[dict[str, str]]) -> str:
    lines = []
    for i, c in enumerate(selected, start=1):
        abs_ = (c.get("abstract") or "").strip()
        abs_part = f"\n   Abstract: {abs_[:500]}" if abs_ else ""
        lines.append(
            f"{i}. Title: {(c.get('title') or '').strip()}\n"
            f"   arXiv: {(c.get('id') or '').strip()}{abs_part}"
        )
    return "\n".join(lines)


def build_user_content(
    title: str,
    abstract: str,
    candidates: list[dict[str, str]],
    k: int = 10,
) -> str:
    instr = INSTRUCTION.format(k=k)
    cand_block = format_candidates(candidates)
    return (
        f"{instr}\n\n"
        f"Title of Paper A: {title.strip()}\n"
        f"Abstract of Paper A: {abstract.strip()}\n\n"
        f"Candidate papers:\n{cand_block}\n"
    )


def build_select_user_content(
    title: str,
    abstract: str,
    candidates: list[dict[str, str]],
    k: int = 10,
) -> str:
    if select_abs_mode(candidates):
        instr = INSTRUCTION_SELECT_ABS
    elif select_rank_aware(candidates):
        instr = INSTRUCTION_SELECT_RANK_AWARE
    else:
        instr = INSTRUCTION_SELECT
    src_abs = abstract.strip()
    if select_abs_mode(candidates):
        src_abs = src_abs[:800]
    return (
        f"{instr.format(k=k)}\n\n"
        f"Title of Paper A: {title.strip()}\n"
        f"Abstract of Paper A: {src_abs}\n\n"
        f"Candidate papers:\n{format_candidates(candidates)}\n"
    )


def format_candidates_title_only(candidates: list[dict[str, str]]) -> str:
    lines = []
    for i, c in enumerate(candidates, start=1):
        cid = (c.get("id") or "").strip()
        title = (c.get("title") or "").strip()[:180]
        lines.append(f"{i}. {cid} | {title}")
    return "\n".join(lines)


def format_candidates_with_abs(
    candidates: list[dict[str, str]], *, abs_chars: int = 400
) -> str:
    lines = []
    for i, c in enumerate(candidates, start=1):
        cid = (c.get("id") or "").strip()
        title = (c.get("title") or "").strip()[:180]
        abs_ = (c.get("abstract") or "").strip().replace("\n", " ")[:abs_chars]
        if abs_:
            lines.append(f"{i}. {cid} | {title} | Abstract: {abs_}")
        else:
            lines.append(f"{i}. {cid} | {title}")
    return "\n".join(lines)


def build_shortlist_user_content(
    title: str,
    abstract: str,
    candidates: list[dict[str, str]],
    k: int = 20,
) -> str:
    src_abs = (abstract or "").strip()[:800]
    return (
        f"{INSTRUCTION_SHORTLIST_TITLE.format(k=k)}\n\n"
        f"Title of Paper A: {title.strip()}\n"
        f"Abstract of Paper A: {src_abs}\n\n"
        f"Candidate papers:\n{format_candidates_title_only(candidates)}\n"
    )


def build_rerank_user_content(
    title: str,
    abstract: str,
    candidates: list[dict[str, str]],
    k: int = 20,
    *,
    abs_chars: int = 400,
) -> str:
    src_abs = (abstract or "").strip()[:800]
    return (
        f"{INSTRUCTION_RERANK_ABS.format(k=k)}\n\n"
        f"Title of Paper A: {title.strip()}\n"
        f"Abstract of Paper A: {src_abs}\n\n"
        f"Candidate papers:\n"
        f"{format_candidates_with_abs(candidates, abs_chars=abs_chars)}\n"
    )


def format_analysis_block(relevance: str, reason: str) -> str:
    rel = (relevance or "none").strip().lower()
    rea = (reason or "").strip().replace("\n", " ")
    return f"Relevance: {rel}\nReason for Citation: {rea}"


def build_analyze_user_content(
    title: str,
    abstract: str,
    paper: dict[str, str],
    *,
    paper_abs_chars: int = 800,
    cand_abs_chars: int = 400,
) -> str:
    a_abs = (abstract or "").strip()[: max(50, int(paper_abs_chars))]
    c_abs = (paper.get("abstract") or "").strip()[: max(50, int(cand_abs_chars))]
    return (
        f"{INSTRUCTION_ANALYZE}\n\n"
        f"Title of Paper A: {title.strip()}\n"
        f"Abstract of Paper A: {a_abs}\n\n"
        f"Candidate Title: {(paper.get('title') or '').strip()}\n"
        f"Candidate arXiv: {(paper.get('id') or '').strip()}\n"
        f"Candidate Abstract: {c_abs}\n"
    )


def format_candidates_with_analysis(
    candidates: list[dict[str, str]],
) -> str:
    lines = []
    for i, c in enumerate(candidates, start=1):
        cid = (c.get("id") or "").strip()
        title = (c.get("title") or "").strip()[:180]
        analysis = (c.get("analysis") or "").strip().replace("\n", " | ")
        if not analysis:
            rel = (c.get("relevance") or "").strip()
            reason = (c.get("reason") or "").strip()
            if rel or reason:
                analysis = format_analysis_block(rel, reason).replace("\n", " | ")
        lines.append(f"{i}. {cid} | {title} | Analysis: {analysis}")
    return "\n".join(lines)


def build_rerank_with_analysis_user_content(
    title: str,
    abstract: str,
    candidates: list[dict[str, str]],
    k: int = 30,
) -> str:
    src_abs = (abstract or "").strip()[:800]
    return (
        f"{INSTRUCTION_RERANK_ANALYSIS.format(k=k)}\n\n"
        f"Title of Paper A: {title.strip()}\n"
        f"Abstract of Paper A: {src_abs}\n\n"
        f"Candidate papers:\n"
        f"{format_candidates_with_analysis(candidates)}\n"
    )


def parse_analysis(text: str) -> dict[str, Any]:
    """Parse Relevance / Reason for Citation block.

    Returns keys: relevance, reason, ok (bool).
    """
    if not text:
        return {"relevance": "", "reason": "", "ok": False}
    rel = ""
    reason = ""
    m_rel = re.search(
        r"Relevance\s*:\s*(core|related|peripheral|none)\b",
        text,
        flags=re.IGNORECASE,
    )
    if m_rel:
        rel = m_rel.group(1).strip().lower()
    m_rea = re.search(
        r"Reason(?:\s+for\s+Citation)?\s*:\s*(.+)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if m_rea:
        reason = m_rea.group(1).strip().split("\n")[0].strip()
    return {"relevance": rel, "reason": reason, "ok": bool(rel) and bool(reason)}


def merge_rerank_with_pool(
    ordered: list[str],
    pool_ids: list[str],
    *,
    n_out: int = 30,
) -> list[str]:
    """S2 ordered ids first, then remaining pool order; truncate to n_out."""
    seen: set[str] = set()
    out: list[str] = []
    for pid in ordered:
        if pid and pid not in seen:
            out.append(pid)
            seen.add(pid)
        if len(out) >= n_out:
            return out[:n_out]
    for pid in pool_ids:
        if pid and pid not in seen:
            out.append(pid)
            seen.add(pid)
        if len(out) >= n_out:
            break
    return out[:n_out]


def chat_messages_shortlist(
    title: str,
    abstract: str,
    candidates: list[dict[str, str]],
    k: int = 20,
) -> list[dict[str, str]]:
    return [
        {
            "role": "user",
            "content": build_shortlist_user_content(title, abstract, candidates, k=k),
        }
    ]


def chat_messages_rerank(
    title: str,
    abstract: str,
    candidates: list[dict[str, str]],
    k: int = 20,
    *,
    abs_chars: int = 400,
) -> list[dict[str, str]]:
    return [
        {
            "role": "user",
            "content": build_rerank_user_content(
                title, abstract, candidates, k=k, abs_chars=abs_chars
            ),
        }
    ]


def build_cite_bundle_user_content(
    title: str,
    abstract: str,
    selected: list[dict[str, str]],
) -> str:
    return (
        f"{INSTRUCTION_CITE_BUNDLE}\n\n"
        f"Title of Paper A: {title.strip()}\n"
        f"Abstract of Paper A: {abstract.strip()}\n\n"
        f"Selected papers:\n{format_selected_for_cite(selected)}\n"
    )


def build_cite_one_user_content(
    title: str,
    abstract: str,
    paper: dict[str, str],
) -> str:
    abs_ = (paper.get("abstract") or "").strip()[:500]
    return (
        f"{INSTRUCTION_CITE_ONE}\n\n"
        f"Title of Paper A: {title.strip()}\n"
        f"Abstract of Paper A: {abstract.strip()}\n\n"
        f"Selected paper Title: {(paper.get('title') or '').strip()}\n"
        f"Selected paper arXiv: {(paper.get('id') or '').strip()}\n"
        f"Selected paper Abstract: {abs_}\n"
    )


def build_pointwise_user_content(
    title: str,
    abstract: str,
    candidate: dict[str, Any],
    *,
    paper_abs_chars: int = 800,
    cand_abs_chars: int = 500,
) -> str:
    """v5.1 pointwise cite/skip prompt (single candidate)."""
    a_abs = (abstract or "").strip()[: max(50, int(paper_abs_chars))]
    c_abs = (candidate.get("abstract") or "").strip()[: max(50, int(cand_abs_chars))]
    return (
        f"{INSTRUCTION_POINTWISE_CITE}\n\n"
        f"Title of Paper A: {title.strip()}\n"
        f"Abstract of Paper A: {a_abs}\n\n"
        f"Candidate Title: {(candidate.get('title') or '').strip()}\n"
        f"Candidate arXiv: {(candidate.get('id') or '').strip()}\n"
        f"Candidate Abstract: {c_abs}\n"
    )


def build_pointwise_prefer_user_content(
    title: str,
    abstract: str,
    cand1: dict[str, Any],
    cand2: dict[str, Any],
    *,
    paper_abs_chars: int = 800,
    cand_abs_chars: int = 500,
) -> str:
    """v5.2 pairwise prefer prompt: output 1 or 2."""
    a_abs = (abstract or "").strip()[: max(50, int(paper_abs_chars))]

    def _blk(c: dict[str, Any], idx: int) -> str:
        c_abs = (c.get("abstract") or "").strip()[: max(50, int(cand_abs_chars))]
        return (
            f"Candidate {idx} Title: {(c.get('title') or '').strip()}\n"
            f"Candidate {idx} arXiv: {(c.get('id') or '').strip()}\n"
            f"Candidate {idx} Abstract: {c_abs}\n"
        )

    return (
        f"{INSTRUCTION_POINTWISE_PREFER}\n\n"
        f"Title of Paper A: {title.strip()}\n"
        f"Abstract of Paper A: {a_abs}\n\n"
        f"{_blk(cand1, 1)}\n"
        f"{_blk(cand2, 2)}"
    )


def chat_messages_infer(
    title: str,
    abstract: str,
    candidates: list[dict[str, str]],
    k: int = 10,
) -> list[dict[str, str]]:
    return [
        {
            "role": "user",
            "content": build_user_content(title, abstract, candidates, k=k),
        }
    ]


def chat_messages_select(
    title: str,
    abstract: str,
    candidates: list[dict[str, str]],
    k: int = 10,
) -> list[dict[str, str]]:
    return [
        {
            "role": "user",
            "content": build_select_user_content(title, abstract, candidates, k=k),
        }
    ]


def chat_messages_cite_bundle(
    title: str,
    abstract: str,
    selected: list[dict[str, str]],
) -> list[dict[str, str]]:
    return [
        {
            "role": "user",
            "content": build_cite_bundle_user_content(title, abstract, selected),
        }
    ]


def chat_messages_cite_one(
    title: str,
    abstract: str,
    paper: dict[str, str],
) -> list[dict[str, str]]:
    return [
        {
            "role": "user",
            "content": build_cite_one_user_content(title, abstract, paper),
        }
    ]


def parse_bundle(text: str) -> list[dict[str, str]]:
    """Parse model Markdown into list of {title, id, sentence}."""
    if not text or not text.strip():
        return []
    items: list[dict[str, str]] = []
    chunks = re.split(r"\n(?=\d+\.\s*Title\s*:)", text.strip())
    for chunk in chunks:
        m_title = re.search(r"Title\s*:\s*(.+)", chunk, flags=re.IGNORECASE)
        m_id = re.search(r"arXiv\s*:\s*([^\n]+)", chunk, flags=re.IGNORECASE)
        m_cite = re.search(r"Cite\s*:\s*(.+)", chunk, flags=re.IGNORECASE | re.DOTALL)
        if not m_title:
            continue
        title = m_title.group(1).strip()
        cite = ""
        if m_cite:
            cite = m_cite.group(1).strip().split("\n")[0].strip()
        pid = m_id.group(1).strip() if m_id else ""
        items.append({"title": title, "id": pid, "sentence": cite})
    return items


_ARXIV_RE = re.compile(
    r"(?:arXiv\s*:\s*)?(\d{4}\.\d{4,5}(?:v\d+)?|[a-z\-]+(?:\.[A-Z]{2})?/\d{7})",
    re.IGNORECASE,
)


def parse_id_list(text: str, candidates: list[dict[str, str]], k: int = 10) -> list[str]:
    """Parse Stage-A id / index lines into candidate ids (order preserved, unique)."""
    if not text:
        return []
    by_id = {(c.get("id") or "").strip(): c for c in candidates if c.get("id")}
    by_idx = {i + 1: c for i, c in enumerate(candidates)}
    out: list[str] = []
    seen: set[str] = set()
    for raw in text.strip().splitlines():
        line = raw.strip()
        if not line:
            continue
        # bare index or "1." / "1)" list forms
        m_idx = re.match(r"^(\d{1,2})(?:[\.\)\]]|\s|$)", line)
        if m_idx and not _ARXIV_RE.search(line):
            idx = int(m_idx.group(1))
            c = by_idx.get(idx)
            if c and c["id"] not in seen:
                seen.add(c["id"])
                out.append(c["id"])
            if len(out) >= k:
                break
            continue
        # strip bullets / leading junk
        line = re.sub(r"^[\-\*]+\s*", "", line).strip()
        m = _ARXIV_RE.search(line)
        if not m:
            continue
        pid = m.group(1).strip()
        # normalize version suffix for match
        if pid not in by_id and pid.rstrip("v0123456789") in by_id:
            pid = pid  # keep; try base
        base = re.sub(r"v\d+$", "", pid)
        if pid in by_id:
            pick = pid
        elif base in by_id:
            pick = base
        else:
            continue
        if pick not in seen:
            seen.add(pick)
            out.append(pick)
        if len(out) >= k:
            break
    return out[:k]


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").lower()).strip()


def title_significant_tokens(title: str) -> set[str]:
    toks = re.findall(r"[a-z0-9]+", (title or "").lower())
    return {t for t in toks if len(t) > 2 and t not in _STOP}


def cite_consistent_with_title(cite: str, title: str) -> bool:
    """L1: cite should overlap title tokens or not invent a conflicting \\cite key story."""
    cite = (cite or "").strip()
    if not cite:
        return False
    title_toks = title_significant_tokens(title)
    if not title_toks:
        return True
    cite_l = cite.lower()
    # direct token overlap
    cite_toks = set(re.findall(r"[a-z0-9]+", cite_l))
    if title_toks & cite_toks:
        return True
    # \\cite{key} keys
    keys = re.findall(r"\\cite\{([^}]+)\}", cite)
    for key in keys:
        parts = re.split(r"[,;]", key)
        for p in parts:
            p = p.strip().lower()
            ptoks = set(re.findall(r"[a-z0-9]+", p))
            if ptoks & title_toks:
                return True
    # acronym-like ALLCAPS words in title vs cite
    acrs = re.findall(r"\b[A-Z][A-Z0-9\-]{1,}\b", title or "")
    for a in acrs:
        if a.lower() in cite_l:
            return True
    return False


def cite_token_jaccard(a: str, b: str) -> float:
    """Token Jaccard on lowercased alnum tokens (for cross-cite near-dup)."""
    ta = set(re.findall(r"[a-z0-9]+", (a or "").lower()))
    tb = set(re.findall(r"[a-z0-9]+", (b or "").lower()))
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / max(1, len(ta | tb))


def cites_near_duplicate(a: str, b: str, *, thr: float = 0.8) -> bool:
    na = _norm(a)
    nb = _norm(b)
    if not na or not nb:
        return False
    if na == nb:
        return True
    return cite_token_jaccard(a, b) >= thr


def pairwise_cite_distinct_rate(
    sentences: list[str],
    *,
    thr: float = 0.8,
) -> float:
    n = len(sentences)
    if n < 2:
        return 1.0
    ok = 0
    tot = 0
    for i in range(n):
        for j in range(i + 1, n):
            tot += 1
            if not cites_near_duplicate(sentences[i], sentences[j], thr=thr):
                ok += 1
    return ok / max(1, tot)


def align_to_candidates(
    items: list[dict[str, str]],
    candidates: list[dict[str, str]],
) -> list[dict[str, str]]:
    """Map each item to a unique candidate by title (then id); drop failures."""
    by_title = {_norm(c["title"]): c for c in candidates if c.get("title")}
    by_id = {(c.get("id") or "").strip(): c for c in candidates if c.get("id")}
    used: set[str] = set()
    aligned: list[dict[str, str]] = []
    for it in items:
        cand = None
        tid = (it.get("id") or "").strip()
        if tid and tid in by_id and by_id[tid]["id"] not in used:
            cand = by_id[tid]
        else:
            nt = _norm(it.get("title") or "")
            if nt in by_title and by_title[nt]["id"] not in used:
                cand = by_title[nt]
            else:
                best = None
                best_score = 0
                for c in candidates:
                    if c["id"] in used:
                        continue
                    ct = _norm(c["title"])
                    if not ct or not nt:
                        continue
                    if nt in ct or ct in nt:
                        score = min(len(nt), len(ct))
                        if score > best_score:
                            best_score = score
                            best = c
                cand = best
        if not cand:
            continue
        used.add(cand["id"])
        aligned.append(
            {
                "id": cand["id"],
                "title": cand["title"],
                "sentence": (it.get("sentence") or "").strip(),
                "abstract": (cand.get("abstract") or it.get("abstract") or "").strip(),
            }
        )
    return aligned


def pad_to_k(
    selected: list[dict[str, str]],
    candidates: list[dict[str, str]],
    k: int = 10,
) -> list[dict[str, str]]:
    """Pad with unused candidates in retrieval order (no invented ids)."""
    out = list(selected[:k])
    used = {c["id"] for c in out}
    for c in candidates:
        if len(out) >= k:
            break
        cid = (c.get("id") or "").strip()
        if not cid or cid in used:
            continue
        used.add(cid)
        out.append(
            {
                "id": cid,
                "title": (c.get("title") or cid).strip(),
                "sentence": "",
                "abstract": (c.get("abstract") or "").strip(),
                "padded": True,
            }
        )
    return out[:k]


def prepare_candidates(
    raw: list[dict[str, Any]],
    *,
    exclude_id: str = "",
    n: int = 30,
) -> list[dict[str, str]]:
    """Exclude self, dedupe near-duplicate titles, keep retrieval order (MMR-lite)."""
    exclude_id = (exclude_id or "").strip()
    seen_ids: set[str] = set()
    seen_titles: set[str] = set()
    out: list[dict[str, str]] = []
    for r in raw:
        cid = (r.get("paper_id") or r.get("id") or "").strip()
        title = (r.get("title") or "").strip()
        if not cid or not title:
            continue
        if exclude_id and cid == exclude_id:
            continue
        if cid in seen_ids:
            continue
        nt = _norm(title)
        # near-dup: identical normalized title
        if nt in seen_titles:
            continue
        # light containment dedupe against kept titles
        dup = False
        for kept in seen_titles:
            if nt in kept or kept in nt:
                if min(len(nt), len(kept)) >= 24:
                    dup = True
                    break
        if dup:
            continue
        seen_ids.add(cid)
        seen_titles.add(nt)
        out.append(
            {
                "id": cid,
                "title": title,
                "abstract": (r.get("abstract") or "").strip()[:500],
                "score": r.get("score"),
            }
        )
        if len(out) >= n:
            break
    return out


def rr_cand_mode() -> str:
    """Legacy name kept for meta; always universe+ranker (slot_mix/dense removed)."""
    return "universe_ranker"


def rr_cand_params() -> dict[str, int]:
    """Env-tunable pool sizes (see docs/REFERENCE_RECOMMEND.md)."""

    def _i(name: str, default: int) -> int:
        raw = (os.environ.get(name) or "").strip()
        if not raw:
            return default
        try:
            return max(0, int(raw))
        except ValueError:
            return default

    return {
        "m": _i("RR_CAND_M", 150),
        "n": _i("RR_CAND_N", 30),
        # kept for backward-compatible env reads; unused by new builder
        "n_dense": _i("RR_CAND_N_DENSE", 30),
        "n_proxy": _i("RR_CAND_N_PROXY", 0),
        "p_hubs": _i("RR_CAND_P", 250),
        "min_outdeg": _i("RR_CAND_MIN_OUTDEG", 2),
    }


def normalize_arxiv_id(pid: str) -> str:
    s = (pid or "").strip().replace("arxiv:", "")
    if "v" in s[4:]:  # version suffix e.g. 2501.11858v2
        base, _, ver = s.rpartition("v")
        if ver.isdigit():
            return base
    return s


def _title_excluded(title: str, exclude_title_norm: str) -> bool:
    if not exclude_title_norm:
        return False
    nt = _norm(title)
    if not nt:
        return False
    if nt == exclude_title_norm:
        return True
    if nt in exclude_title_norm or exclude_title_norm in nt:
        if min(len(nt), len(exclude_title_norm)) >= 24:
            return True
    return False


def _hit_to_cand(r: dict[str, Any]) -> dict[str, str] | None:
    cid = normalize_arxiv_id(r.get("paper_id") or r.get("id") or "")
    title = (r.get("title") or "").strip()
    if not cid or not title:
        return None
    return {
        "id": cid,
        "title": title,
        "abstract": (r.get("abstract") or "").strip()[:500],
        "score": r.get("score"),
        "source": "dense",
    }


def build_rr_candidate_pool(
    dense_hits: list[dict[str, Any]],
    graph: Any,
    *,
    exclude_title: str = "",
    exclude_id: str = "",
    mode: str | None = None,
    n: int | None = None,
    n_dense: int | None = None,
    n_proxy: int | None = None,
    p_hubs: int | None = None,
    min_outdeg: int | None = None,
    title: str | None = None,
    abstract: str | None = None,
    search_fn: Any = None,
    return_meta: bool = False,
) -> list[dict[str, str]] | tuple[list[dict[str, str]], dict[str, Any]]:
    """Cold-start pool: large universe → independent ranker → Top-N (default 30).

    ``dense_hits`` / ``mode`` / slot params are ignored (API kept for callers).
    Prefer passing ``search_fn`` + title/abstract so multi-query + PRF can run.
    Does **not** use the query paper's own out-edges.
    """
    from rwcite.ranker.rr_ranker import rank_universe, rr_ranker_params
    from rwcite.ranker.rr_universe import build_rr_universe

    params = rr_cand_params()
    n = int(n if n is not None else params["n"] or rr_ranker_params()["n"])
    excl = normalize_arxiv_id(exclude_id)
    q_title = (title if title is not None else exclude_title) or ""
    q_abs = abstract or ""

    def _search_from_hits(_query: str, _limit: int) -> list[dict[str, Any]]:
        # Fallback when only a single dense hit list is available.
        return list(dense_hits or [])

    sf = search_fn or _search_from_hits
    pack = build_rr_universe(
        q_title,
        q_abs,
        graph,
        search_fn=sf,
        exclude_id=excl,
        multi_query=search_fn is not None,
        prf=search_fn is not None,
    )
    ranked = rank_universe(
        title=q_title,
        abstract=q_abs,
        universe_pack=pack,
        graph=graph,
        n=n,
    )
    # title near-dup guard within Top-N
    out: list[dict[str, str]] = []
    used_titles: set[str] = set()
    for c in ranked:
        nt = _norm(c.get("title") or "")
        if nt and nt in used_titles:
            continue
        if nt and any(
            min(len(nt), len(k)) >= 24 and (nt in k or k in nt) for k in used_titles
        ):
            continue
        if nt:
            used_titles.add(nt)
        out.append(
            {
                "id": c["id"],
                "title": c.get("title") or "",
                "abstract": c.get("abstract") or "",
                "score": c.get("score"),
                "source": c.get("source") or "ranked",
            }
        )
        if len(out) >= n:
            break
    meta = {
        "cand_mode": "universe_ranker",
        "universe_size": pack.get("meta", {}).get("universe_size", 0),
        "pool_size": len(out),
        "universe_ids": [c["id"] for c in pack.get("universe") or []],
    }
    if return_meta:
        return out, meta
    return out


def _dedupe_ranked(ranked_ids: list[str], limit: int | None = None) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for x in ranked_ids:
        pid = normalize_arxiv_id(x)
        if not pid or pid in seen:
            continue
        seen.add(pid)
        out.append(pid)
        if limit is not None and len(out) >= limit:
            break
    return out


def hits_at_n(
    ranked_ids: list[str],
    gold_ids: list[str],
    n: int,
) -> int:
    """hits@N = |Top-N ∩ G| (count) on a rank-ordered id list."""
    n = max(1, int(n))
    gold = {normalize_arxiv_id(x) for x in gold_ids if x}
    if not gold:
        return 0
    return len(set(_dedupe_ranked(ranked_ids, limit=n)) & gold)


def recall_at_g(
    ranked_ids: list[str],
    gold_ids: list[str],
) -> float:
    """R@G: |Top-|G| ∩ G| / |G| using a rank-ordered id list."""
    gold = [normalize_arxiv_id(x) for x in gold_ids if x]
    gold_set = set(gold)
    n = len(gold_set)
    if n == 0:
        return 0.0
    ranked = _dedupe_ranked(ranked_ids, limit=n)
    return len(set(ranked) & gold_set) / n


def decompose_pool_metrics(
    pred_ids: list[str],
    gold_ids: list[str],
    pool_ids: list[str] | None,
) -> dict[str, Any]:
    """pool_recall / select_recall / end-to-end recall (cold-start diagnostics)."""
    pred = {normalize_arxiv_id(x) for x in pred_ids if x}
    gold = {normalize_arxiv_id(x) for x in gold_ids if x}
    pool = {normalize_arxiv_id(x) for x in (pool_ids or []) if x}
    gold_in_pool = gold & pool if pool else set()
    pool_recall = (len(gold_in_pool) / len(gold)) if gold else 0.0
    if pool and gold_in_pool:
        select_recall: float | None = len(pred & gold_in_pool) / len(gold_in_pool)
    elif pool and gold and not gold_in_pool:
        select_recall = None
    else:
        select_recall = None
    # R@K (product Top-K) vs R@G (Top-|G| on ranked pool when available).
    e2e_at_k = (len(pred & gold) / len(gold)) if gold else 0.0
    ranked_for_g = list(pool_ids or []) or list(pred_ids)
    return {
        "pool_size": len(pool),
        "gold_in_pool": len(gold_in_pool),
        "pool_recall": pool_recall,
        "select_recall": select_recall,
        "recall_at_k": e2e_at_k,
        "recall_at_g": recall_at_g(ranked_for_g, list(gold)),
        "precision_at_k": (len(pred & gold) / len(pred)) if pred else 0.0,
        "hit_count": len(pred & gold),
    }


def render_bundle(items: list[dict[str, str]]) -> str:
    return format_gold_output(items)


def parse_title_abstract(message: str) -> tuple[str, str]:
    """Extract title/abstract from Tools-style message."""
    title, abstract = "", ""
    m_t = re.search(
        r"Title of Paper(?: A)?\s*:\s*(.+?)(?:\n|Abstract)",
        message,
        flags=re.IGNORECASE | re.DOTALL,
    )
    m_a = re.search(
        r"Abstract of Paper(?: A)?\s*:\s*(.+?)(?:\n\nCandidate|\nCandidate|\Z)",
        message,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if m_t:
        title = m_t.group(1).strip()
    if m_a:
        abstract = m_a.group(1).strip()
    if not title and not abstract:
        abstract = message.strip()
    return title, abstract


def metrics_sets(
    pred_ids: list[str], gold_ids: list[str]
) -> dict[str, Any]:
    ps, gs = set(pred_ids), set(gold_ids)
    inter = ps & gs
    prec = len(inter) / len(ps) if ps else 0.0
    rec = len(inter) / len(gs) if gs else 0.0
    return {
        "precision": prec,
        "recall": rec,
        "hit_count": len(inter),
        "pred_count": len(ps),
        "gold_count": len(gs),
    }


def _cite_bertscore_for_hits(
    pairs: list[tuple[str, str]],
) -> tuple[dict[str, float | None], list[float | None]]:
    """Mean BERTScore over (pred_cite, gold_cite) hit pairs; per-pair F1 list."""
    empty = {
        "cite_bertscore_precision": None,
        "cite_bertscore_recall": None,
        "cite_bertscore_f1": None,
        "cite_scored_pairs": 0,
    }
    if not pairs:
        return empty, []
    try:
        from rwcite.runtime.services.bertscore_service import get_bertscore_service

        rows = get_bertscore_service().score_many(
            [a for a, _ in pairs], [b for _, b in pairs]
        )
    except Exception:
        return empty, [None] * len(pairs)
    if not rows:
        return empty, []
    return {
        "cite_bertscore_precision": sum(r["precision"] for r in rows) / len(rows),
        "cite_bertscore_recall": sum(r["recall"] for r in rows) / len(rows),
        "cite_bertscore_f1": sum(r["f1"] for r in rows) / len(rows),
        "cite_scored_pairs": len(rows),
    }, [float(r["f1"]) for r in rows]


def score_prediction(
    candidate_text: str,
    reference_text: str,
    k: int = 10,
    pool_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Score generated RR Markdown against gold Markdown (paper Tools / API).

    Predictions are capped at K (hits@K / P@K). Gold uses the **full** parsed
    reference set — do not truncate gold to K (that undercounts hits@K when the
    paper has many outgoing edges). Cite quality: mean BERTScore F1 on hit pairs
    (pred Cite vs gold Cite for the same arXiv id).
    """
    k = max(1, min(10, int(k)))
    pred = parse_bundle(candidate_text)[:k]
    gold = parse_bundle(reference_text)  # full gold; not [:k]
    gold_pool = [
        {
            "id": normalize_arxiv_id(g.get("id") or ""),
            "title": (g.get("title") or "").strip(),
            "sentence": (g.get("sentence") or "").strip(),
        }
        for g in gold
        if (g.get("id") or g.get("title"))
    ]
    aligned = align_to_candidates(pred, gold_pool) if gold_pool else []
    pred_ids = [
        normalize_arxiv_id(p.get("id") or "")
        for p in pred
        if (p.get("id") or "").strip()
    ]
    gold_ids = [
        normalize_arxiv_id(g.get("id") or "")
        for g in gold
        if (g.get("id") or "").strip()
    ]
    gold_by_id: dict[str, dict[str, str]] = {}
    for g in gold:
        gid = normalize_arxiv_id(g.get("id") or "")
        if not gid or gid in gold_by_id:
            continue
        gold_by_id[gid] = {
            "id": gid,
            "title": (g.get("title") or "").strip(),
            "sentence": (g.get("sentence") or "").strip(),
        }

    sets = metrics_sets(pred_ids, gold_ids)
    cite_total = len(pred)
    cite_nonempty = sum(1 for p in pred if (p.get("sentence") or "").strip())
    cite_ok = sum(
        1
        for p in pred
        if cite_consistent_with_title(p.get("sentence") or "", p.get("title") or "")
    )

    # Hit pairing for cite quality + UI color groups (match_idx).
    hit_order: list[str] = []
    seen_hits: set[str] = set()
    for pid in pred_ids:
        if pid and pid in gold_by_id and pid not in seen_hits:
            seen_hits.add(pid)
            hit_order.append(pid)
    match_idx_by_id = {pid: i for i, pid in enumerate(hit_order)}

    cite_pairs: list[tuple[str, str]] = []
    cite_pair_ids: list[str] = []
    for pid in hit_order:
        pc = ""
        for p in pred:
            if normalize_arxiv_id(p.get("id") or "") == pid:
                pc = (p.get("sentence") or "").strip()
                break
        gc = (gold_by_id[pid].get("sentence") or "").strip()
        if pc and gc:
            cite_pairs.append((pc, gc))
            cite_pair_ids.append(pid)
    cite_agg, cite_f1s = _cite_bertscore_for_hits(cite_pairs)
    cite_f1_by_id = {pid: f1 for pid, f1 in zip(cite_pair_ids, cite_f1s)}

    pred_items: list[dict[str, Any]] = []
    for i, p in enumerate(pred):
        pid = normalize_arxiv_id(p.get("id") or "")
        title = (p.get("title") or "").strip()
        sent = (p.get("sentence") or "").strip()
        hit = bool(pid and pid in match_idx_by_id)
        pred_items.append(
            {
                "rank": i + 1,
                "id": pid,
                "title": title,
                "sentence": sent,
                "hit": hit,
                "match_idx": match_idx_by_id.get(pid, -1),
                "cite_ok": cite_consistent_with_title(sent, title) if sent else False,
                "cite_f1": cite_f1_by_id.get(pid),
            }
        )

    gold_items: list[dict[str, Any]] = []
    for i, g in enumerate(gold):
        gid = normalize_arxiv_id(g.get("id") or "")
        hit = bool(gid and gid in match_idx_by_id)
        gold_items.append(
            {
                "rank": i + 1,
                "id": gid,
                "title": (g.get("title") or "").strip(),
                "sentence": (g.get("sentence") or "").strip(),
                "hit": hit,
                "match_idx": match_idx_by_id.get(gid, -1),
                "cite_f1": cite_f1_by_id.get(gid) if hit else None,
            }
        )

    # Ranked list for pool hits@N / R@G (prefer cold-start pool order).
    ranked_for_hits = (
        [normalize_arxiv_id(x) for x in pool_ids if x]
        if pool_ids is not None
        else list(pred_ids)
    )
    h10 = hits_at_n(ranked_for_hits, gold_ids, 10)
    h30 = hits_at_n(ranked_for_hits, gold_ids, 30)
    h50 = hits_at_n(ranked_for_hits, gold_ids, 50)
    r_at_g = recall_at_g(ranked_for_hits, gold_ids)
    out: dict[str, Any] = {
        "format_ok": len(pred) > 0,
        "parsed_count": len(pred),
        "gold_count": len(gold_ids),
        "title_hit": (len(aligned) / len(pred)) if pred else 0.0,
        "precision_at_k": sets["precision"],
        "recall_at_k": sets["recall"],
        "recall_at_g": r_at_g,
        # Product Top-K hits (Cite bundle); pool hits@N below for ranker ladder.
        "hit_count": sets["hit_count"],
        "hits_at_k": sets["hit_count"],
        "hits_at_10": h10,
        "hits_at_30": h30,
        "hits_at_50": h50,
        "cite_nonempty_rate": (cite_nonempty / cite_total) if cite_total else 0.0,
        "cite_consistency_rate": (cite_ok / cite_total) if cite_total else 0.0,
        **cite_agg,
        "pred_ids": pred_ids,
        "gold_ids": gold_ids,
        "pred_items": pred_items,
        "gold_items": gold_items,
        "k": k,
    }
    if pool_ids is not None:
        decomp = decompose_pool_metrics(pred_ids, gold_ids, pool_ids)
        out["pool_ids"] = [normalize_arxiv_id(x) for x in pool_ids if x]
        out["pool_size"] = decomp["pool_size"]
        out["gold_in_pool"] = decomp["gold_in_pool"]
        out["pool_recall"] = decomp["pool_recall"]
        out["select_recall"] = decomp["select_recall"]
        out["precision_at_k"] = decomp["precision_at_k"]
        out["recall_at_k"] = decomp["recall_at_k"]
        out["recall_at_g"] = decomp["recall_at_g"]
        out["hit_count"] = decomp["hit_count"]
        out["hits_at_k"] = decomp["hit_count"]
        # Recompute hits@N on normalized pool order (same as ranked_for_hits).
        out["hits_at_10"] = hits_at_n(out["pool_ids"], gold_ids, 10)
        out["hits_at_30"] = hits_at_n(out["pool_ids"], gold_ids, 30)
        out["hits_at_50"] = hits_at_n(out["pool_ids"], gold_ids, 50)
    return out
