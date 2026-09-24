import regex
import re

import networkx as nx

from rwcite.latex.utils import fix_citations

# Starred / optional-arg section headings, e.g. \section*{Intro} \section[x]{Intro}
_SECTION_CMD_RE = regex.compile(
    r"\\section\*?(?:\[(?:.*?)\])*\{((?:[^{}]+|\{(?1)\})*)\}(?:\[(?:.*?)\])*",
    regex.DOTALL,
)
_SECTION_END_RE = re.compile(
    r"\\appendix\b|\\begin\{thebibliography\}|\\bibliographystyle\b|\\end\{document\}",
    re.I,
)

_RELATED_EXACT = {
    "literature review",
    "related work",
    "related works",
    "prior work",
    "prior works",
    "related research",
    "research overview",
    "previous work",
    "previous works",
    "review of the literature",
    "review of related literature",
    "survey of related work",
    "survey of related works",
    "background",
    "research background",
    "review of prior research",
    "literature survey",
    "overview of literature",
    "existing literature",
    "review of existing work",
    "review of existing works",
    "review of previous studies",
    "review of prior literature",
    "summary of related research",
    "survey of existing literature",
    "survey of literature",
    "existing research overview",
    "prior literature review",
    "related literature",
    "related studies",
    "state of the art",
    "state-of-the-art",
}

_RELATED_FUZZY = (
    "related work",
    "related works",
    "literature review",
    "prior work",
    "previous work",
    "related research",
    "related literature",
    "background and related",
    "state of the art",
    "state-of-the-art",
)


def retrieve_text_cite(text, command):
    # Allow \command* and optional [...] before/after the mandatory {...}
    base_pattern = (
        r'\\'
        + command
        + r"\*?(?:\[(?:.*?)\])*\{((?:[^{}]+|\{(?1)\})*)\}(?:\[(?:.*?)\])*"
    )

    def extract_text_inside_curly_braces(text):
        pattern = r"\{((?:[^{}]|(?R))*)\}"

        match = regex.search(pattern, text)

        if match:
            return match.group(1)
        else:
            return ""

    found_texts = []
    for match in regex.finditer(base_pattern, text):
        temp_substring = text[match.span()[0] : match.span()[1]]
        found_texts.append(extract_text_inside_curly_braces(temp_substring))

    return found_texts


def _normalize_section_title(title: str) -> str:
    """Strip labels / markup so 'Introduction\\label{x}' matches introduction."""
    if not title:
        return ""
    t = str(title)
    t = re.sub(r"\\label\s*\{[^}]*\}", " ", t)
    t = re.sub(r"\\[a-zA-Z]+\*?\s*", " ", t)
    t = re.sub(r"[{}$~^_&%#]", " ", t)
    t = re.sub(r"\s+", " ", t).strip().lower()
    t = re.sub(r"^[\d\.\)]+\s*", "", t)
    return t.strip(" :-–—|")


def _is_intro_title(title: str) -> bool:
    n = _normalize_section_title(title)
    if not n:
        return False
    if n == "introduction":
        return True
    # e.g. "Introduction and Motivating Work"
    if n.startswith("introduction"):
        return True
    return False


def _is_related_title(title: str) -> bool:
    n = _normalize_section_title(title)
    if not n:
        return False
    if n in _RELATED_EXACT:
        return True
    return any(key in n for key in _RELATED_FUZZY)


def list_section_spans(content: str) -> list[tuple[int, int, str]]:
    """Return (header_start, body_start, title) for each \\section / \\section*."""
    out: list[tuple[int, int, str]] = []
    for match in _SECTION_CMD_RE.finditer(content or ""):
        title = match.group(1) or ""
        out.append((match.start(), match.end(), title))
    return out


def _section_body_by_pred(content: str, pred) -> str:
    spans = list_section_spans(content)
    if not spans:
        return ""
    idx = next((i for i, (_, _, title) in enumerate(spans) if pred(title)), None)
    if idx is None:
        return ""
    body_start = spans[idx][1]
    if idx + 1 < len(spans):
        body_end = spans[idx + 1][0]
    else:
        m_end = _SECTION_END_RE.search(content, body_start)
        body_end = m_end.start() if m_end else len(content)
    return content[body_start:body_end]


def get_citing_sentences(content, *, clean: bool = True):
    # Raw arXiv / natbib often uses \citep/\citet; normalize before scanning.
    content = fix_citations(content or "")
    content_new = re.sub(r'[\n]+', ' ', content) # keep only one \n
    content_new = re.sub(r'e\.g\.' , 'eg', content_new)
    content_new = re.sub(r'i\.e\.' , 'eg', content_new)
    content_new = re.sub(r'etc\.' , 'etc', content_new)
    content_new = re.sub(r' +', ' ', content_new)
    sentences = [sentence + '.' for sentence in content_new.split('.')]
    citing_sentences = [s for s in sentences if '\\cite' in s]
    results = {}
    if clean:
        try:
            from rwcite.latex.cite_sentence_clean import clean_cite_sentence
        except ImportError:  # pragma: no cover
            clean_cite_sentence = None  # type: ignore
    else:
        clean_cite_sentence = None  # type: ignore
    for s in citing_sentences:
        span = s
        if clean_cite_sentence is not None:
            res = clean_cite_sentence(s)
            # Prefer cleaned text when usable; else keep raw span for key resolve.
            if res.text and ("\\cite" in res.text or "\\cite" in s):
                span = res.text if "\\cite" in res.text else s
        # Keys always come from the raw span. clean_cite_sentence caps \cite{...}
        # key lists (max_cite_keys) and cuts text past max_len, so reading keys off
        # the cleaned text silently drops citations — and with them graph edges.
        # Cleaning decides how the sentence reads, never which edges exist.
        citations = retrieve_text_cite(s, 'cite')
        final_citations = []
        for cite in citations:
            final_citations.extend(cite.split(','))
        # Key the map by cleaned span when possible so edges store clean text.
        # Distinct raw spans can clean to the same text; merge instead of drop.
        key = span if final_citations else s
        if key in results:
            have = set(results[key])
            results[key].extend(c for c in final_citations if c not in have)
        else:
            results[key] = final_citations
    return results


def get_intro(content):
    return _section_body_by_pred(content or "", _is_intro_title)


def get_related_works(content):
    return _section_body_by_pred(content or "", _is_related_title)


_BODY_END_RE = re.compile(
    r"\\begin\{thebibliography\}|"
    r"\\bibliography\b|"
    r"\\bibliographystyle\b|"
    r"\\section\*?\{[^}]*(?:acknowledg|appendix|supplement)|"
    r"\\subsection\*?\{[^}]*(?:acknowledg)|"
    r"\\end\{document\}",
    re.I,
)


def get_body_cite_fallback(content: str) -> str:
    """Letter/PRL-style papers often have no \\section{Introduction}.

    Use main body after frontmatter (maketitle / abstract) until bibliography
    or acknowledgements, so Intro+RW cite extraction still sees \\cite.
    """
    c = content or ""
    if not c.strip():
        return ""
    start = 0
    for pat in (
        r"\\maketitle\b",
        r"\\end\{abstract\}",
        r"\\begin\{document\}",
    ):
        m = re.search(pat, c, re.I)
        if m:
            start = max(start, m.end())
    # Skip leftover abstract if we anchored at begin{document}
    m_abs = re.search(r"\\begin\{abstract\}.*?\\end\{abstract\}", c[start:], re.I | re.S)
    if m_abs and m_abs.start() < 2000:
        start = start + m_abs.end()
    body = c[start:]
    m_end = _BODY_END_RE.search(body)
    if m_end:
        body = body[: m_end.start()]
    # Cap extremely long bodies (full monographs)
    if len(body) > 80_000:
        body = body[:80_000]
    return body


def get_intro_related_for_rr(content: str) -> tuple[str, str]:
    """Intro + Related-Work bodies; body fallback when both section extracts empty."""
    intro = get_intro(content) or ""
    related = get_related_works(content) or ""
    if (len(intro) + len(related)) < 50:
        fb = get_body_cite_fallback(content)
        if len(fb) >= 50:
            return fb, ""
    return intro, related


def normalize_section_text(text):
    if text is None:
        return ""
    text = str(text)
    return text if text.strip() else ""


def rename_graph_keys(data_graph):
    return {
        (
            "/".join(re.match(r"([a-z-]+)([0-9]+)", key, re.I).groups())
            if re.match(r"([a-z-]+)([0-9]+)", key, re.I)
            else key
        ): value
        for key, value in data_graph.items()
    }


def format_concepts(paper_id, id2topics):
    if paper_id not in id2topics:
        return ""
    return ", ".join(
        list(set(item for sublist in id2topics[paper_id] for item in sublist))
    )


def build_gexf_node_attrs(paper_id, renamed_data, papers, id2topics, keep_content=False):
    entry = renamed_data.get(paper_id, {})
    attrs = {
        "title": papers[paper_id]["title"],
        "abstract": papers[paper_id]["abstract"],
        "introduction": normalize_section_text(entry.get("Introduction", "")),
        "related": normalize_section_text(entry.get("Related Work", "")),
        "concepts": format_concepts(paper_id, id2topics),
    }
    if keep_content:
        attrs["content"] = entry.get("content", "")
    return attrs


def upsert_gexf_node(g, paper_id, renamed_data, papers, id2topics, keep_content=False):
    if paper_id not in papers:
        return False

    attrs = build_gexf_node_attrs(
        paper_id, renamed_data, papers, id2topics, keep_content
    )
    if paper_id not in g:
        g.add_node(paper_id, **attrs)
        return True

    node = g.nodes[paper_id]
    for field in ("introduction", "related"):
        if normalize_section_text(attrs.get(field)) and not normalize_section_text(
            node.get(field)
        ):
            node[field] = attrs[field]
    if keep_content:
        if normalize_section_text(attrs.get("content")) and not normalize_section_text(
            node.get("content")
        ):
            node["content"] = attrs["content"]
    if not normalize_section_text(node.get("concepts")) and attrs.get("concepts"):
        node["concepts"] = attrs["concepts"]
    return True


def build_citation_graph(renamed_data, papers, id2topics, keep_content=False):
    g = nx.DiGraph()
    for key in renamed_data:
        upsert_gexf_node(g, key, renamed_data, papers, id2topics, keep_content)
        for citation in renamed_data[key].get("Citations", []):
            source, target, metadata = citation
            sentence = metadata.get("sentence", "")
            upsert_gexf_node(g, target, renamed_data, papers, id2topics, keep_content)
            if source in g and target in g:
                g.add_edge(source, target, sentence=sentence)
    g.remove_nodes_from(list(nx.isolates(g)))
    return g
