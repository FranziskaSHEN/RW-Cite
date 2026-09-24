import regex
import re

import networkx as nx


def retrieve_text_cite(text, command):
    base_pattern = (
        r'\\' + command + r"(?:\[(?:.*?)\])*\{((?:[^{}]+|\{(?1)\})*)\}(?:\[(?:.*?)\])*"
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


def get_citing_sentences(content):
    content_new = re.sub(r'[\n]+', ' ', content) # keep only one \n
    content_new = re.sub(r'e\.g\.' , 'eg', content_new)
    content_new = re.sub(r'i\.e\.' , 'eg', content_new)
    content_new = re.sub(r'etc\.' , 'etc', content_new)
    content_new = re.sub(r' +', ' ', content_new)
    sentences = [sentence + '.' for sentence in content_new.split('.')]
    citing_sentences = [s for s in sentences if '\\cite' in s]
    results = {}
    for s in citing_sentences:
        citations = retrieve_text_cite(s, 'cite')
        final_citations = []
        for cite in citations:
            final_citations.extend(cite.split(','))
        results[s] = final_citations
    return results


def get_intro(content):
    sections = retrieve_text_cite(content, 'section')
    if sections == []:
        return ''
    try_intro = [x for x in sections if x.strip().lower() == 'introduction']
    if try_intro == []:
        return ''
    else:
        to_find = try_intro[0]
        ind = sections.index(to_find)
    if ind + 1 < len(sections):
        start_marker = f'\\section{{{sections[ind]}}}'
        end_marker = f'\\section{{{sections[ind+1]}}}'
        start_point = content.find(start_marker)
        end_point = content.find(end_marker)
        return content[start_point+len(start_marker):end_point]
    else:
        return ''


def get_related_works(content):
    sections = retrieve_text_cite(content, 'section')
    if sections == []:
        return ''
    possible_related = [
        "Literature Review",
        "Related Work",
        "Related Works",
        "Prior Work",
        "Prior Works",
        "Related Research",
        "Research Overview",
        "Previous Work",
        "Previous Works",
        "Review of the Literature",
        "Review of Related Literature",
        "Survey of Related Work",
        "Survey of Related Works",
        "Background",
        "Research Background",
        "Review of Prior Research",
        "Literature Survey",
        "Overview of Literature",
        "Existing Literature",
        "Review of Existing Work",
        "Review of Existing Works",
        "Review of Previous Studies",
        "Review of Prior Literature",
        "Summary of Related Research",
        "Survey of Existing Literature",
        "Survey of Literature",
        "Existing Research Overview",
        "Prior Literature Review"
    ]
    possible_sections = [x for x in sections if any([True for y in possible_related if y.lower() == x.strip().lower()])]
    if possible_sections == []:
        return ''
    else:
        to_find = possible_sections[0]
        ind = sections.index(to_find)

    if ind + 1 < len(sections):
        start_marker = f'\\section{{{sections[ind]}}}'
        end_marker = f'\\section{{{sections[ind+1]}}}'
        start_point = content.find(start_marker)
        end_point = content.find(end_marker)
        return content[start_point+len(start_marker):end_point]

    else:
        return ''


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
