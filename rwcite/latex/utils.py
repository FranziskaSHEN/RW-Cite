import contextlib
import logging
import os
from pathlib import Path
import re
import sys
import yaml
import regex
import shutil
import subprocess
import numpy as np
import bibtexparser
import bibtexparser.model as bibtex_model
import networkx as nx
from langdetect import LangDetectException, detect
from charset_normalizer import from_path

_SKIP_BIB_BLOCKS = (
    bibtex_model.DuplicateBlockKeyBlock,
    bibtex_model.DuplicateFieldKeyBlock,
    bibtex_model.ParsingFailedBlock,
    bibtex_model.ImplicitComment,
    bibtex_model.ExplicitComment,
    bibtex_model.String,
    bibtex_model.Preamble,
    bibtex_model.MiddlewareErrorBlock,
)
# Allow whitespace between \\bibitem and optional [...] / mandatory {key}
# (REVTeX/natbib: "\\bibitem [{...}]{key}"). Negative lookahead skips
# \\bibitemStop / \\bibitemNoStop helpers.
_BIBITEM_PATTERN = re.compile(
    r"\\bibitem(?![A-Za-z])\s*(?:\[[^\]]*\])?\s*\{([^}]+)\}"
)
_LATEX_QUOTED_TITLE = re.compile(r'``(.*?)' + "''", re.DOTALL)
_BIBINFO_FIELD_RE = re.compile(
    r"\\bibinfo\s*\{\s*([a-zA-Z]+)\s*\}\s*\{",
    re.I,
)


def _repo_config(rel: str) -> str:
    """Resolve configs/* against RWCITE_ROOT (cwd fallback)."""
    root = os.environ.get("RWCITE_ROOT") or os.getcwd()
    return str(Path(root) / rel)


# ACL dumps (~40MB anthology.bib) are not paper-local bibliographies; parsing them
# floods logs and can pollute cite-key → title resolution.
_MAX_BIB_FILE_BYTES = 1_500_000
_SKIP_BIB_NAME_RE = re.compile(r"(^|[/\\])anthology\.bib$", re.I)


@contextlib.contextmanager
def _quiet_bibtexparser():
    """Suppress bibtexparser WARNING spam (DuplicateBlockKey / ParsingFailed)."""
    root = logging.getLogger("bibtexparser")
    prev = root.level
    root.setLevel(logging.ERROR)
    try:
        yield
    finally:
        root.setLevel(prev)


def should_skip_bibliography_file(path: str, size: int | None = None) -> bool:
    """Skip non-local mega-bibs (e.g. ACL anthology.bib inside arXiv sources)."""
    if _SKIP_BIB_NAME_RE.search(path.replace("\\", "/")):
        return True
    if size is None:
        try:
            size = os.path.getsize(path)
        except OSError:
            return False
    return size > _MAX_BIB_FILE_BYTES


def _parse_bibtex_string(content: str):
    with _quiet_bibtexparser():
        return bibtexparser.parse_string(content)


def is_venv():
    return (hasattr(sys, 'real_prefix') or
            (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix))

def read_yaml_file(file_path):
    with open(file_path, 'r') as file:
        try:
            data = yaml.safe_load(file)
            return data
        except yaml.YAMLError as e:
            print(f"Error reading YAML file: {e}")

def read_tex_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        tex_content = file.read()
    return tex_content

def write_tex_file(file_path, s):
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(s)

def get_core(s):
    start = '\\begin{document}'
    end = '\\end{document}'
    beginning_doc = s.find(start)
    end_doc = s.rfind(end)
    return s[beginning_doc+len(start):end_doc]


def retrieve_text(text, command, keep_text=False):
    """Removes '\\command{*}' from the string 'text'.

    Regex `base_pattern` used to match balanced parentheses taken from:
    https://stackoverflow.com/questions/546433/regular-expression-to-match-balanced-parentheses/35271017#35271017
    """
    base_pattern = (
        r'\\' + command + r"(?:\[(?:.*?)\])*\{((?:[^{}]+|\{(?1)\})*)\}(?:\[(?:.*?)\])*"
    )

    def extract_text_inside_curly_braces(text):
        """Extract text inside of {} from command string"""
        pattern = r"\{((?:[^{}]|(?R))*)\}"

        match = regex.search(pattern, text)

        if match:
            return match.group(1)
        else:
            return ""

    # Loops in case of nested commands that need to retain text, e.g. \red{hello \red{world}}.
    while True:
        all_substitutions = []
        has_match = False
        for match in regex.finditer(base_pattern, text):
            # In case there are only spaces or nothing up to the following newline,
            # adds a percent, not to alter the newlines.
            has_match = True

            if not keep_text:
                new_substring = ""
            else:
                temp_substring = text[match.span()[0] : match.span()[1]]
                return extract_text_inside_curly_braces(temp_substring)

            if match.span()[1] < len(text):
                next_newline = text[match.span()[1] :].find("\n")
                if next_newline != -1:
                    text_until_newline = text[
                        match.span()[1] : match.span()[1] + next_newline
                    ]
                    if (
                        not text_until_newline or text_until_newline.isspace()
                    ) and not keep_text:
                        new_substring = "%"
            all_substitutions.append((match.span()[0], match.span()[1], new_substring))

        for start, end, new_substring in reversed(all_substitutions):
            text = text[:start] + new_substring + text[end:]

        if not keep_text or not has_match:
            break


def reduce_linebreaks(s):
    return re.sub(r'(\n[ \t]*)+(\n[ \t]*)+', '\n\n', s)


def replace_percentage(s):
    return re.sub(r'% *\n', '\n', s)   


def reduce_spaces(s):
    return re.sub(' +', ' ', s)


def delete_urls(s):
    return re.sub(r'http\S+', '', s)


def remove_tilde(s):
    s1 = re.sub(r'[~ ]\.', '.', s)
    s2 = re.sub(r'[~ ],', ',', s1)
    return re.sub(r'{}', '', s2)


def remove_verbatim_words(s):
    with open(_repo_config("configs/latex_commands.yaml"), "r") as stream:
        read_config = yaml.safe_load(stream)
    
    for command in read_config['verbatim_to_delete']:
        s = s.replace(command, '')

    for command in read_config['two_arguments']:
        pattern = r'\\' + command + r'{[^}]*}' + r'{[^}]*}'
        s = re.sub(pattern, '', s)

    for command in read_config['three_arguments']:
        pattern = r'\\' + command + r'{[^}]*}' + r'{[^}]*}' + r'{[^}]*}'
        s = re.sub(pattern, '', s)

    for command in read_config['two_arguments_elaborate']:
        s = remove_multargument(s, '\\' + command, 2)

    for command in read_config['three_arguments_elaborate']:
        s = remove_multargument(s, '\\' + command, 3)

    for command in read_config['replace_comments']:
        pattern = r'\\' + command
        s = re.sub(pattern, '%', s)
    
    s = re.sub(
      r'\\end{[\s]*abstract[\s]*}',
      '',
      s,
      flags=re.IGNORECASE
    )

    s = re.sub(
      r'\\begin{[\s]*abstract[\s]*}',
      'Abstract\n\n',
      s,
      flags=re.IGNORECASE
    )
    return s


def yes_or_no(s):
    return 1 if "Yes" == s[0:3] else 0 if "No" == s[0:2] else -1


def get_main(directory):
    file_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            file_paths.append(file_path)
    latex_paths = [f for f in file_paths if f.endswith('.tex')]
    number_tex = len(latex_paths)
    if number_tex == 0:
        return None
    if number_tex == 1:
        return latex_paths[0]
    adjacency = np.zeros((number_tex, number_tex))
    keys = [os.path.basename(path) for path in latex_paths]
    reg_ex = r'\\input{(.*?)}|\\include{(.*?)}|\\import{(.*?)}|\\subfile{(.*?)}|\\include[*]{(.*?)}|}'
    for i,file in enumerate(latex_paths):
        content = read_tex_file(file)
        find_pattern_input = re.findall(reg_ex, content)
        find_pattern_input = [tup for tup in find_pattern_input if not all(element == "" for element in tup)]
        number_matches = len(find_pattern_input)
        if number_matches == 0:
            continue
        else:
            content = replace_imports(file, content)
        reg_ex_clean = r'\\input{(.*?)}|\\include{(.*?)}'
        find_pattern_input = re.findall(reg_ex_clean, content)
        number_matches = len(find_pattern_input)  
        for j in range(number_matches):
            match = find_pattern_input[j]
            non_empty_match = [t for t in match if t]
            for non_empty in non_empty_match:
                base_match = os.path.basename(non_empty)
                if not base_match.endswith('.tex'):
                    base_match = base_match + '.tex'
                    if base_match not in keys:
                        continue
                ind = keys.index(base_match)
                adjacency[i][ind] = 1
    G = nx.from_numpy_array(adjacency, create_using=nx.DiGraph)
    connected_components = list(nx.weakly_connected_components(G))
    size_connected = [len(x) for x in connected_components]
    maximum_size = max(size_connected)
    biggest_connected = [x for x in connected_components if len(x) == maximum_size]
    if len(biggest_connected)>1:
        roots = [n for connected in biggest_connected for n in connected if not list(G.predecessors(n))]
        _check = []
        for r in roots:
            try: 
                _check.append(check_begin(latex_paths[r]))
            except Exception as e:
                _check.append(False)
        potentials_files = [latex_paths[x] for x, y in zip(roots, _check) if y == True]
        if potentials_files:
            sizes_files = [os.path.getsize(x) for x in potentials_files]
            return potentials_files[sizes_files.index(max(sizes_files))]
        return _fallback_main_tex(latex_paths)
        
    else:
        roots = [n for n in biggest_connected[0] if not list(G.predecessors(n))]
        if roots:
            return latex_paths[roots[0]]
        return _fallback_main_tex(latex_paths)


def initial_clean(directory, config):
    # Prefer project cleaning config whenever present (avoids cleaner crashes /
    # inconsistent defaults when callers pass config=False).
    cleaning_cfg = _repo_config("configs/cleaning_config.yaml")
    config_cmd = ''
    if config is True or os.path.exists(cleaning_cfg):
        if not os.path.exists(cleaning_cfg):
            raise FileNotFoundError(
                f"{cleaning_cfg} required for arxiv_latex_cleaner "
                "(provide it through the RW-Cite configuration directory)"
            )
        config_cmd = f'--config {cleaning_cfg}'
    base_dir = directory[:directory.rfind('/')]
    temp_dir = base_dir + '_temp' + '/'
    cleaned_directory = base_dir + '_arXiv'

    for stale_dir in (temp_dir, cleaned_directory):
        if os.path.exists(stale_dir):
            shutil.rmtree(stale_dir)

    shutil.copytree(directory, temp_dir)
    try:
        command_res = os.system('arxiv_latex_cleaner --keep_bib {} {}'.format(directory, config_cmd))
        if command_res != 0:
            raise Exception('Error cleaning')
        else:
            shutil.rmtree(temp_dir)
        
    except Exception as e:
        shutil.rmtree(directory, ignore_errors=True)
        if os.path.exists(temp_dir):
            os.rename(temp_dir, directory)
        file_paths = []
        for root, _, files in os.walk(directory):
            for file in files:
                file_path = os.path.join(root, file)
                file_paths.append(file_path) 
        latex_paths = [f for f in file_paths if f.endswith('.tex')]
        for p in latex_paths:
            results = from_path(p)
            with open(p, 'w', encoding='utf-8') as f:
                f.write(str(results.best()))
        os.system('arxiv_latex_cleaner --keep_bib {} {}'.format(directory, config_cmd))
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    if not os.path.exists(cleaned_directory):
        # Cleaner failed to emit *_arXiv; keep the working directory as-is.
        if not os.path.exists(directory):
            raise FileNotFoundError(
                f"arxiv_latex_cleaner produced no output for {directory}"
            )
        return
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.rename(cleaned_directory, directory)


def _fallback_main_tex(latex_paths):
    candidates = [path for path in latex_paths if check_begin(path)]
    if candidates:
        return max(candidates, key=os.path.getsize)
    return latex_paths[0] if latex_paths else None


def check_begin(directory):
    content = read_tex_file(directory)
    if not re.findall(r'\\begin{document}', content):
        return False
    try:
        return detect(content) == 'en'
    except LangDetectException:
        return len(content.strip()) > 100


_MONTH_NAMES = (
    "january|february|march|april|may|june|july|august|september|october|"
    "november|december|jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec"
)
_VENUE_LIKE_RE = re.compile(
    r"^(arxiv(\s+preprint)?|proceedings|journal|corr|coRR|ieee|acm|neurips|"
    r"nips|icml|iclr|cvpr|iccv|eccv|aaai|acl|emnlp|naacl|nature|science|"
    r"advances in neural|in advances|in proceedings|workshop|transactions|"
    r"conference|symposium|volume\b|vol\b|pp\b|pages\b)",
    re.I,
)
_ARXIV_ID_IN_TEXT_RE = re.compile(
    r"(?:arxiv\.org/(?:abs|pdf)/|arxiv\s*preprint\s*arxiv\s*:?\s*|"
    r"arxiv\s*:\s*|arxiv\s+)"
    r"(\d{4}\.\d{4,5}|[a-z\-]+(?:\.[A-Z]{2})?/\d{7})(?:v\d+)?",
    re.I,
)
_BARE_ARXIV_ID_RE = re.compile(
    r"\b(\d{4}\.\d{4,5})(?:v\d+)?\b"
)


# Unicode / LaTeX greek (and a few symbols) → ASCII tokens so bib titles like
# ``$\pi_0.5$: ...`` and metadata ``$π_{0.5}$: ...`` land on the same key.
_GREEK_ASCII: tuple[tuple[str, str], ...] = (
    ("α", "alpha"),
    ("β", "beta"),
    ("γ", "gamma"),
    ("δ", "delta"),
    ("ε", "epsilon"),
    ("θ", "theta"),
    ("λ", "lambda"),
    ("μ", "mu"),
    ("π", "pi"),
    ("σ", "sigma"),
    ("φ", "phi"),
    ("ϕ", "phi"),
    ("ψ", "psi"),
    ("ω", "omega"),
    ("×", "x"),
)


def _normalize_citation_title(title):
    """Normalize a bib/metadata title for exact-key lookup in the title index.

    Must be identical for the side that *builds* the index (arXiv OAI titles)
    and the side that *queries* it (bib entries). Handles LaTeX math, escaped
    underscores, unicode greek, and SE(3)-style group notation that otherwise
    silently drop resolvable edges.
    """
    s = str(title or "")
    s = re.sub(r"[\n\r]+", " ", s)
    # Keep math body; drop the dollars. Do this before command stripping so
    # ``$\pi_0.5$`` / ``$π_{0.5}$`` become comparable.
    s = re.sub(r"\$([^$]*)\$", r" \1 ", s)
    # Escaped punctuation commonly left in bibtex titles
    s = s.replace("\\_", "_").replace("\\&", " and ").replace("\\%", " percent ")
    s = s.replace("\\,", " ").replace("\\;", " ").replace("\\!", "")
    s = s.replace("\\emph", "").replace("\\emp", "").replace("\\em", "")
    # ``\text{...}`` / ``\mathrm{...}`` keep inner text
    for cmd in (
        "text",
        "textrm",
        "textit",
        "textbf",
        "texttt",
        "textsc",
        "mathrm",
        "mathbf",
        "mathit",
        "mathsf",
        "operatorname",
        "mbox",
        "hbox",
    ):
        s = re.sub(rf"\\{cmd}\s*\{{([^{{}}]*)\}}", r" \1 ", s)
    # Sub/superscripts: ``_{0.5}`` / ``^2`` → keep the body
    s = re.sub(r"_\s*\{([^{}]*)\}", r" \1 ", s)
    s = re.sub(r"\^\s*\{([^{}]*)\}", r" \1 ", s)
    s = re.sub(r"_(\w+)", r" \1 ", s)
    s = re.sub(r"\^(\w+)", r" \1 ", s)
    # LaTeX greek commands before generic ``\cmd`` wipe
    for name in (
        "alpha",
        "beta",
        "gamma",
        "delta",
        "epsilon",
        "varepsilon",
        "theta",
        "lambda",
        "mu",
        "pi",
        "sigma",
        "phi",
        "varphi",
        "psi",
        "omega",
        "times",
        "cdot",
    ):
        repl = "x" if name == "times" else (" " if name == "cdot" else name)
        s = re.sub(rf"\\{name}\b", f" {repl} ", s)
    for uni, ascii_name in _GREEK_ASCII:
        s = s.replace(uni, f" {ascii_name} ")
    # Drop remaining TeX commands / braces / math glue
    s = re.sub(r"\\[a-zA-Z]+\*?", " ", s)
    s = s.replace("{", " ").replace("}", " ")
    s = s.replace("``", " ").replace("''", " ").replace("`", " ")
    # Curly / typographic apostrophes → ASCII
    s = s.replace("\u2019", "'").replace("\u2018", "'").replace("\u02bc", "'")
    s = s.replace("\u201c", " ").replace("\u201d", " ")
    # Group notation: ``SE(3)`` / ``SE (3)`` / ``SO(2)`` → ``se 3``
    s = re.sub(r"\b([A-Za-z]{1,4})\s*\(\s*(\d+)\s*\)", r"\1 \2", s)
    # Strip most punctuation; keep word chars, spaces, hyphens
    s = re.sub(r"[^\w\s\-']", " ", s, flags=re.UNICODE)
    s = s.replace("'", " ")
    s = re.sub(r"\s+", " ", s).strip(" .-").lower()
    return s


def _extract_balanced_brace_group(text: str, open_idx: int) -> str | None:
    """Return insides of `{...}` starting at open_idx (must point at '{')."""
    if open_idx < 0 or open_idx >= len(text) or text[open_idx] != "{":
        return None
    depth = 0
    for j in range(open_idx, len(text)):
        ch = text[j]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[open_idx + 1 : j]
    return None


def _extract_bibinfo_fields(body: str) -> dict[str, str]:
    """Parse REVTeX/natbib `\\bibinfo {field} {value}` pairs from a bibitem body."""
    out: dict[str, str] = {}
    if not body:
        return out
    for m in _BIBINFO_FIELD_RE.finditer(body):
        field = (m.group(1) or "").strip().lower()
        val = _extract_balanced_brace_group(body, m.end() - 1)
        if not field or val is None:
            continue
        val = re.sub(r"\s+", " ", val).strip()
        if val:
            out[field] = val
    return out


# Fields that may carry an arXiv id, in priority order. Google Scholar exports
# preprints as `journal = {arXiv preprint arXiv:2410.24164}` with no eprint field,
# and such entries usually have LaTeX-heavy titles ("$\pi_0$: ...") that the title
# index cannot match — so skipping the venue fields loses the edge entirely.
# extract_arxiv_id_from_text only accepts a bare id when the field also mentions
# arXiv, which keeps volume/page numbers from being read as ids.
ID_BEARING_FIELDS: tuple[str, ...] = (
    "eprint",
    "arxiv",
    "arxiv_id",
    "arxivId",
    "archiveprefix",
    "url",
    "doi",
    "note",
    "howpublished",
    "journal",
    "journaltitle",
    "booktitle",
    "series",
    "publisher",
    "volume",
    "title",
)


def extract_arxiv_id_from_text(text: str) -> str | None:
    """Best-effort arXiv id from bib fields / bbl venue lines."""
    if not text:
        return None
    s = str(text)
    m = _ARXIV_ID_IN_TEXT_RE.search(s)
    if m:
        return m.group(1)
    # "arxiv preprint arxiv:2212.06817" after normalization may be bare
    if "arxiv" in s.lower() or "eprint" in s.lower():
        m2 = _BARE_ARXIV_ID_RE.search(s)
        if m2:
            return m2.group(1)
    return None


def citation_title_lookup_keys(title: str) -> list[str]:
    """Generate normalized title variants for metadata index lookup."""
    if not title:
        return []
    base = _normalize_citation_title(title)
    keys: list[str] = []
    seen: set[str] = set()

    def add(x: str) -> None:
        x = re.sub(r"\s+", " ", (x or "").strip(" .-"))
        if not x or len(x.split()) < 2:
            return
        if x in seen:
            return
        seen.add(x)
        keys.append(x)

    add(base)
    # drop trailing month + year / year / leftover month
    no_my = re.sub(rf"\b(?:{_MONTH_NAMES})\s+\d{{4}}$", "", base, flags=re.I)
    add(no_my)
    no_year = re.sub(r"\b\d{4}$", "", base).strip()
    add(no_year)
    add(re.sub(rf"\b(?:{_MONTH_NAMES})$", "", no_year, flags=re.I))
    # drop leading arxiv preprint boilerplate
    add(re.sub(r"^arxiv(\s+preprint)?(\s+arxiv)?(\s*:?\s*\d{4}\.\d{4,5})?\s*", "", base, flags=re.I))
    # remove leftover "arxiv:NNNN.NNNNN" tokens
    add(re.sub(r"\barxiv\s*:?\s*\d{4}\.\d{4,5}\b", "", base, flags=re.I))
    # hyphen / space collapse: ``vision-language-action`` ↔ ``vision language action``
    add(base.replace("-", " "))
    add(re.sub(r"(?<=\w)-(?=\w)", "", base))
    parts = base.split()
    if len(parts) >= 3 and parts[0] in {"a", "an", "the"}:
        add(" ".join(parts[1:]))
    return keys

def _strip_tex_markup_keep_text(s: str) -> str:
    s = re.sub(r"\\emph\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\{\\em\s+([^}]*)\}", r"\1", s)
    s = re.sub(r"\\[a-zA-Z]+\*?\s*", " ", s)
    s = re.sub(r"[{}]", "", s)
    s = re.sub(r"\s+", " ", s).strip(" .")
    return s


def _looks_like_venue(text: str) -> bool:
    n = _normalize_citation_title(text)
    if not n:
        return True
    if _VENUE_LIKE_RE.search(n):
        return True
    if extract_arxiv_id_from_text(n) and len(n.split()) <= 6:
        return True
    return False


def _looks_like_title(text: str) -> bool:
    plain = _strip_tex_markup_keep_text(text)
    words = [w for w in re.split(r"\s+", plain) if w]
    if len(words) < 2:
        return False
    low = plain.lower()
    # REVTeX author/journal soup leaked into title heuristics
    if any(
        tok in low
        for tok in (
            "bibnamefont",
            "bibfnamefont",
            "bibinfo",
            "bibfield",
            "bibitem",
        )
    ):
        return False
    if _looks_like_venue(plain):
        return False
    # "schoelkopf science 339 1169 (2013)" style venue lines
    if re.search(r"\b(nature|science|prl|pra|prb|prx)\b.*\b\d{3,4}\b", low):
        if len(words) <= 8:
            return False
    return True


def _extract_title_from_bibitem_body(body):
    """Extract title from a \\bibitem body (natbib \\newblock / REVTeX bibinfo)."""
    raw = body or ""
    # --- REVTeX / APS: \\bibinfo {title} {....} ---
    bibinfo = _extract_bibinfo_fields(raw)
    if bibinfo.get("title") and _looks_like_title(bibinfo["title"]):
        return _normalize_citation_title(_strip_tex_markup_keep_text(bibinfo["title"]))
    # Structured REVTeX entry with journal/author but no title: do not invent
    # a title from the author block (avoids bibnamefont garbage).
    if bibinfo and not bibinfo.get("title") and (
        "journal" in bibinfo or "volume" in bibinfo
    ):
        match = _LATEX_QUOTED_TITLE.search(raw)
        if match and _looks_like_title(match.group(1)):
            return _normalize_citation_title(match.group(1))
        return None

    # --- preferred: natbib newblock structure (authors / title / venue) ---
    blocks = [b.strip() for b in re.split(r"\\newblock\s*", raw) if b.strip()]
    if len(blocks) >= 2:
        for cand in blocks[1:]:
            if _looks_like_title(cand):
                return _normalize_citation_title(_strip_tex_markup_keep_text(cand))
        # sometimes title is wrapped in emph inside block 1
        for cand in blocks[1:]:
            m = re.search(r"\\emph\{([^}]+)\}", cand)
            if m and _looks_like_title(m.group(1)):
                return _normalize_citation_title(m.group(1))

    flat = re.sub(r"\\newblock\s*", " ", raw)
    flat = re.sub(r"\s+", " ", flat).strip()

    match = _LATEX_QUOTED_TITLE.search(flat)
    if match and _looks_like_title(match.group(1)):
        return _normalize_citation_title(match.group(1))

    for pat in (r"\\emph\{([^}]+)\}", r"\{\\em\s+([^}]+)\}"):
        for m in re.finditer(pat, flat):
            if _looks_like_title(m.group(1)):
                return _normalize_citation_title(m.group(1))

    match = re.search(r",\s*in:\s", flat, re.I)
    if match:
        prefix = flat[: match.start()]
        if "," in prefix:
            cand = prefix.split(",", 1)[1]
            if _looks_like_title(cand):
                return _normalize_citation_title(cand)

    match = re.search(r",\s*\\emph\{", flat)
    if match:
        prefix = flat[: match.start()]
        if "," in prefix:
            cand = prefix.split(",", 1)[1]
            if _looks_like_title(cand):
                return _normalize_citation_title(cand)

    # APA / plain-text: "Authors (2024). Title. Venue"
    match = re.search(
        r"\(\d{4}[a-z]?\)\.\s+(.+?)(?:\.\s+[A-Z\d]|$)",
        flat,
    )
    if match:
        title = match.group(1).strip().rstrip(".")
        if _looks_like_title(title):
            return _normalize_citation_title(title)

    # Fallback: sentence-like segments that are not venue-like
    parts = [p.strip() for p in flat.split(".") if p.strip()]
    for part in parts[1:]:
        words = part.split()
        if len(words) >= 4 and not re.fullmatch(r"\d+", words[0]) and _looks_like_title(part):
            return _normalize_citation_title(part)
    return None


def parse_thebibliography(content):
    library = {}
    matches = list(_BIBITEM_PATTERN.finditer(content))
    for index, match in enumerate(matches):
        key = match.group(1).strip()
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(content)
        if index + 1 == len(matches):
            end_marker = content.find('\\end{thebibliography}', start)
            if end_marker != -1:
                end = end_marker
        body = content[start:end].strip()
        entry: dict = {}
        bibinfo = _extract_bibinfo_fields(body)
        title = _extract_title_from_bibitem_body(body)
        if title:
            entry["title"] = title
        # Prefer structured bibinfo ids when present
        for fld in ("eprint", "doi", "url", "howpublished", "note"):
            if bibinfo.get(fld) and fld not in entry:
                entry[fld] = bibinfo[fld]
        aid = extract_arxiv_id_from_text(body)
        if aid:
            entry["eprint"] = aid
        elif entry.get("eprint"):
            aid2 = extract_arxiv_id_from_text(str(entry["eprint"]))
            if aid2:
                entry["eprint"] = aid2
        if entry:
            library[key] = entry
    return library


def resolve_citation_entry(
    entry: dict | None,
    title_index: dict[str, str],
    known_ids: set[str] | None = None,
) -> str | None:
    """Map a bib/bbl library entry to an arXiv id (or None)."""
    if not entry:
        return None

    def _accept(pid: str | None) -> str | None:
        if not pid:
            return None
        pid = str(pid).strip()
        if not pid:
            return None
        if known_ids is not None and pid not in known_ids:
            # still accept modern ids that look valid; known_ids miss is rare
            if not (
                re.match(r"^\d{4}\.\d{4,5}$", pid)
                or re.match(r"^[a-z\-]+(?:\.[A-Z]{2})?/\d{7}$", pid, re.I)
            ):
                return None
        return pid

    # 1) explicit identifiers
    for fld in ID_BEARING_FIELDS:
        val = entry.get(fld)
        if not val:
            continue
        # archivePrefix alone is not an id
        if fld == "archiveprefix" and str(val).lower() in {"arxiv", "arXiv"}:
            continue
        aid = extract_arxiv_id_from_text(str(val))
        ok = _accept(aid)
        if ok:
            return ok
        # eprint field sometimes is already a bare id
        if fld in {"eprint", "arxiv", "arxiv_id", "arxivId"}:
            bare = str(val).strip().replace("arXiv:", "").replace("arxiv:", "")
            bare = re.sub(r"v\d+$", "", bare)
            ok = _accept(bare if (known_ids is None or bare in known_ids) else None)
            if ok:
                return ok
            if re.match(r"^\d{4}\.\d{4,5}$", bare):
                return bare

    # 2) title index with soft variants
    title = entry.get("title") or ""
    for key in citation_title_lookup_keys(title):
        tid = title_index.get(key)
        if tid:
            return str(tid)
    return None


def parse_thebibliography_file(path):
    with open(path, 'r', encoding='utf-8', errors='replace') as handle:
        return parse_thebibliography(handle.read())


def _iter_bibtex_entries(blocks):
    for block in blocks:
        if isinstance(block, bibtex_model.Entry):
            yield block


def post_processing(extracted_dir, file):
    _dir = os.path.dirname(file) + '/'
    perl_expand(file)
    file = _dir + 'merged_latexpand.tex'
    try:
        de_macro(file)
        file = _dir + 'merged_latexpand-clean.tex'
    except Exception as e:
        pass
    try:
        def_handle(file)
    except Exception as e:
        pass
    try:
        declare_operator(file) # has additional add-ons 
    except Exception as e:
        pass
    try:
        de_macro(file)
        file = _dir + os.path.splitext(os.path.basename(file))[0] + '-clean' + '.tex'
    except Exception as e:
        pass
    initial_clean(_dir, config=True)
    initial_clean(_dir, config=False)
    tex_content = read_tex_file(file)
    final_tex = reduce_spaces(
        delete_urls(
            remove_tilde(
                reduce_linebreaks(
                    replace_percentage(
                        remove_verbatim_words(
                            tex_content
                        )
                    )
                )
            )
        )
    ).strip()
    shutil.rmtree(extracted_dir)
    os.makedirs(extracted_dir)
    write_tex_file(extracted_dir + 'final_cleaned.tex', final_tex)
    initial_clean(extracted_dir, config=False)    
    return extracted_dir + 'final_cleaned.tex'


def perl_expand(file):
    # Save the current working directory
    oldpwd = os.getcwd()
    target_dir = os.path.abspath(os.path.dirname(file)) + os.sep
    target = os.path.join(target_dir, 'latexpand')
    src = str(Path(__file__).resolve().parent / 'latexpand')
    # Copy the `latexpand` script to the target directory
    shutil.copyfile(src, target)
    # Change to the target directory
    os.chdir(target_dir)

    merged_path = os.path.join(target_dir, 'merged_latexpand.tex')
    with open('merged_latexpand.tex', 'w') as output_file:
        result = subprocess.run(
            ['perl', 'latexpand', os.path.basename(file)],
            stdout=output_file,
            stderr=subprocess.PIPE,
            text=True,
        )
    os.chdir(oldpwd)

    if result.returncode != 0:
        raise RuntimeError(
            f"latexpand failed (exit {result.returncode}): {result.stderr.strip()}"
        )
    if os.path.getsize(merged_path) == 0:
        raise RuntimeError(
            "latexpand produced an empty merged_latexpand.tex; "
            f"stderr: {result.stderr.strip()}"
        )


def de_macro(file):
    # Save the current working directory\
    oldpwd = os.getcwd()
    target_dir = os.path.dirname(file) + '/'
    # Construct the target path
    target = os.path.join(target_dir, 'de-macro.py')
    src = str(Path(__file__).resolve().parent / 'de-macro.py')

    # Copy the `de-macro.py` script to the target directory
    shutil.copyfile(src, target)
    # Change to the target directory
    os.chdir(target_dir)

    # Run the de-macro script without os.system and capture errors
    try:
        subprocess.run(['python3', 'de-macro.py', os.path.basename(file)],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    except subprocess.CalledProcessError as e:
        raise Exception(f"Error de-macro: {e}") from e
    finally:
        # Always return to the original directory
        os.chdir(oldpwd)


def def_handle(file):
    h = os.system('python3 {} {} --output {}'.format(str(Path(__file__).resolve().parent / 'def_handle.py'), file, file))
    if h != 0:
        raise Exception('Error def handle')


def declare_operator(file):
    s = read_tex_file(file)
    ## Operators
    pattern = r'\\DeclareMathOperator'
    s = re.sub(pattern, r'\\newcommand', s)
    pattern = {
    r'\\newcommand\*': r'\\newcommand',
    r'\\providecommand\*': r'\\newcommand',
    r'\\providecommand': r'\\newcommand',
    r'\\renewcommand\*': r'\\renewcommand',
    r'\\newenvironment\*': r'\\newenvironment',
    r'\\renewenvironment\*': r'\\renewenvironment'
    }
    s = re.sub(r'\\end +', r'\\end', s)
    for key in pattern:
        s = re.sub(key, pattern[key], s)
    ## Title
    start = '\\begin{document}'
    beginning_doc = s.find(start)
    pattern = {
            r'\\icmltitlerunning\*': r'\\title',
            r'\\icmltitlerunning': r'\\title',
            r'\\inlinetitle\*': r'\\title',
            r'\\icmltitle\*': r'\\title',
            r'\\inlinetitle': r'\\title',
            r'\\icmltitle': r'\\title',
            r'\\titlerunning\*': r'\\title',
            r'\\titlerunning': r'\\title',
            r'\\toctitle': r'\\title',
            r'\\title\*': r'\\title',
            r'\\TITLE\*': r'\\title',
            r'\\TITLE': r'\\title',
            r'\\Title\*': r'\\title',
            r'\\Title': r'\\title',
        }
    for key in pattern:
        s = re.sub(key, pattern[key], s)
    find_potential = s.find('\\title') 

    ## Remove \\
    title_content = retrieve_text(s, 'title', keep_text = True)
    if title_content != None:
        cleaned_title = re.sub(r'\\\\', ' ', title_content)
        cleaned_title = re.sub(r'\n',' ', cleaned_title)
        cleaned_title = re.sub(r'\~',' ', cleaned_title)
        s = s.replace(title_content, cleaned_title)
        if find_potential != -1 and find_potential < beginning_doc:
            s = s.replace('\\maketitle', cleaned_title)

    ##  Cite and ref commands 
    pattern = {
        r'\\citep\*': r'\\cite',
        r'\\citet\*': r'\\cite',
        r'\\citep': r'\\cite',
        r'\\citet': r'\\cite',
        r'\\cite\*': r'\\cite',
        r'\\citealt\*': r'\\cite',
        r'\\citealt': r'\\cite',
        r'\\citealtp\*': r'\\cite',
        r'\\citealp': r'\\cite',
        r'\\citeyear\*': r'\\cite',
        r'\\citeyear': r'\\cite',
        r'\\citeauthor\*': r'\\cite',
        r'\\citeauthor': r'\\cite',
        r'\\citenum\*': r'\\cite',
        r'\\citenum': r'\\cite',
        r'\\cref': r'\\ref',
        r'\\Cref': r'\\ref',
        r'\\factref': r'\\ref',
        r'\\appref': r'\\ref',
        r'\\thmref': r'\\ref',
        r'\\secref': r'\\ref',
        r'\\lemref': r'\\ref',
        r'\\corref': r'\\ref',
        r'\\eqref': r'\\ref',
        r'\\autoref': r'\\ref',
        r'begin{thm}': r'begin{theorem}',
        r'begin{lem}': r'begin{lemma}',
        r'begin{cor}': r'begin{corollary}',
        r'begin{exm}': r'begin{example}',
        r'begin{defi}': r'begin{definition}',
        r'begin{rem}': r'begin{remark}',
        r'begin{prop}': r'begin{proposition}',
        r'end{thm}': r'end{theorem}',
        r'end{lem}': r'end{lemma}',
        r'end{cor}': r'end{corollary}',
        r'end{exm}': r'end{example}',
        r'end{defi}': r'end{definition}',
        r'end{rem}': r'end{remark}',
        r'end{prop}': r'end{proposition}',
    }

    for key in pattern:
        s = re.sub(key, pattern[key], s)

    
    pattern = {
        r'subsubsection':  r'section',
        r'subsubsection ': r'section',
        r'subsubsection\*':  r'section',
        r'subsubsection\* ':  r'section',
        r'subsection': r'section',
        r'subsection ':  r'section',
        r'subsection\*': r'section',
        r'subsection\* ': r'section',
        r'section ':  r'section',
        r'section\*': r'section',
        r'section\* ': r'section',
        r'chapter':  r'section',
        r'chapter ': r'section',
        r'chapter\*':  r'section',
        r'chapter\* ':  r'section',
        r'mysubsubsection': r'section',
        r'mysubsection':  r'section',
        r'mysection':  r'section',
    }

    for key in pattern:
        s = re.sub(key, pattern[key], s)

    # In case any new commands for appendix/appendices 
    s = re.sub(r'newcommand{\\appendix}', '', s)
    s = re.sub(r'newcommand{\\appendices}', '', s)
    s = get_core(s)
    
    ## In case of double titles being defined 
    title_content = retrieve_text(s, 'title', keep_text = True)
    if title_content != None:
        cleaned_title = re.sub(r'\\\\', ' ', title_content)
        cleaned_title = re.sub(r'\n',' ', cleaned_title)
        cleaned_title = re.sub(r'\~',' ', cleaned_title)
        s = s.replace(title_content, cleaned_title)
    write_tex_file(file, s)
    

def replace_imports(file, s):
    regex_p1 = r'\\import{(.*?)}{(.*?)}'
    s = re.sub(regex_p1, r"\\input{\1\2}", s)
    regex_p2 = r'\\subfile{(.*?)}'
    s = re.sub(regex_p2, r"\\input{\1}", s)
    regex_p3 = r'\\include[*]{(.*?)}'
    s = re.sub(regex_p3, r"\\input{\1}", s)
    write_tex_file(file, s)
    return s


def remove_multargument(s, target, k):
    ind = s.find(target)
    while ind != -1:
        start_ind = ind + len(target)
        stack_open = 0
        stack_close = 0
        track_arg  = 0
        for i, char in enumerate(s[start_ind:]):
            if char == '{':
                stack_open += 1
            if char == '}':
                stack_close += 1
            if stack_open !=0 and stack_close !=0:
                if stack_open == stack_close:
                    track_arg += 1
                    stack_open = 0
                    stack_close = 0
            if track_arg == k:
                break
        s = s[:ind] + s[start_ind + i + 1:]
        ind = s.find(target)
    return s


def fix_citations(s):
    # Longer / starred forms first so they are not partially rewritten.
    pattern = {
        r'\\citep\*': r'\\cite',
        r'\\citet\*': r'\\cite',
        r'\\citep': r'\\cite',
        r'\\citet': r'\\cite',
        r'\\cite\*': r'\\cite',
        r'\\citealt\*': r'\\cite',
        r'\\citealt': r'\\cite',
        r'\\citealtp\*': r'\\cite',
        r'\\citealp\*': r'\\cite',
        r'\\citealp': r'\\cite',
        r'\\citeyear\*': r'\\cite',
        r'\\citeyear': r'\\cite',
        r'\\citeauthor\*': r'\\cite',
        r'\\citeauthor': r'\\cite',
        r'\\citenum\*': r'\\cite',
        r'\\citenum': r'\\cite',
        r'\\parencite\*': r'\\cite',
        r'\\parencite': r'\\cite',
        r'\\textcite\*': r'\\cite',
        r'\\textcite': r'\\cite',
        r'\\footcite\*': r'\\cite',
        r'\\footcite': r'\\cite',
        r'\\smartcite\*': r'\\cite',
        r'\\smartcite': r'\\cite',
        r'\\autocite\*': r'\\cite',
        r'\\autocite': r'\\cite',
    }
    for key in pattern:
        s = re.sub(key, pattern[key], s)
    return s


_MAIN_TEX_NAMES = {
    "main.tex",
    "paper.tex",
    "ms.tex",
    "root.tex",
    "article.tex",
    "arxiv.tex",
    "manuscript.tex",
}
_BAD_TEX_NAME_RE = re.compile(
    r"(appendix|suppl|supplement|response|rebuttal|checklist|table|figure|fig_|anon)",
    re.I,
)
_INPUT_INCLUDE_RE = re.compile(
    r"\\(?:input|include)\*?\{([^}]+)\}"
)


def _read_tex_bytes(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            return handle.read()
    except OSError:
        return ""


def score_main_tex_candidate(path: str, content: str | None = None) -> int:
    """Higher score ⇒ likelier main paper body (not appendix / table dump)."""
    name = os.path.basename(path).lower()
    if content is None:
        content = _read_tex_bytes(path)
    score = 0
    if name in _MAIN_TEX_NAMES:
        score += 100
    if name == "final_cleaned.tex":
        score -= 20
    if _BAD_TEX_NAME_RE.search(name):
        score -= 80
    if re.search(r"\\documentclass\b", content):
        score += 40
    if re.search(r"\\begin\{document\}", content):
        score += 30
    nsec = len(re.findall(r"\\section\*?\{", content))
    score += min(nsec, 20) * 4
    ninput = len(_INPUT_INCLUDE_RE.findall(content))
    score += min(ninput, 12) * 2
    low = content.lower()
    if "introduction" in low:
        score += 15
    if "related work" in low or "related works" in low:
        score += 10
    # Prefer moderate size over enormous generated tables
    size = len(content)
    if size < 1500:
        score -= 30
    elif size > 400_000:
        score -= 40
    return score


def select_main_tex(directory: str) -> str | None:
    """Pick best root .tex under directory (non-recursive filenames first)."""
    if not directory or not os.path.isdir(directory):
        return None
    candidates: list[str] = []
    for root, _dirs, files in os.walk(directory):
        for fname in files:
            if not fname.lower().endswith(".tex"):
                continue
            if fname.lower() == "final_cleaned.tex":
                continue
            candidates.append(os.path.join(root, fname))
    if not candidates:
        return None
    ranked = sorted(
        candidates,
        key=lambda p: (score_main_tex_candidate(p), -len(p)),
        reverse=True,
    )
    return ranked[0]


def _resolve_tex_input(base_dir: str, ref: str) -> str | None:
    ref = (ref or "").strip().strip('"').replace("\\", "/")
    if not ref or ref.startswith("/"):
        return None
    # strip leading ./
    while ref.startswith("./"):
        ref = ref[2:]
    trials = []
    if ref.lower().endswith(".tex"):
        trials.append(ref)
    else:
        trials.extend([ref + ".tex", ref])
    for rel in trials:
        path = os.path.normpath(os.path.join(base_dir, rel))
        if os.path.isfile(path):
            return path
    return None


def expand_tex_inputs(
    content: str,
    base_dir: str,
    *,
    max_depth: int = 8,
    _stack: set[str] | None = None,
) -> str:
    """Recursively inline \\input/\\include (best-effort; skips missing files)."""
    if not content or max_depth <= 0:
        return content or ""
    stack = _stack if _stack is not None else set()

    def repl(match: re.Match) -> str:
        ref = match.group(1)
        path = _resolve_tex_input(base_dir, ref)
        if path is None:
            return match.group(0)
        ap = os.path.abspath(path)
        if ap in stack:
            return ""
        body = _read_tex_bytes(path)
        if not body:
            return match.group(0)
        stack.add(ap)
        try:
            expanded = expand_tex_inputs(
                body, os.path.dirname(path), max_depth=max_depth - 1, _stack=stack
            )
        finally:
            stack.discard(ap)
        return "\n" + expanded + "\n"

    return _INPUT_INCLUDE_RE.sub(repl, content)


def prepare_rr_tex(content: str, paper_dir: str) -> str:
    """Expand inputs + light normalize for Intro/RW cite extraction."""
    if not content:
        return ""
    if _INPUT_INCLUDE_RE.search(content):
        content = expand_tex_inputs(content, paper_dir)
    return content

def find_bib(directory):
    file_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            file_paths.append(file_path) 
    bib_paths = [f for f in file_paths if f.endswith('.bib')]
    return bib_paths

def create_bib_from_bbl(bibfile):
    if should_skip_bibliography_file(bibfile):
        return {}
    with open(bibfile, 'r', encoding='utf-8', errors='replace') as handle:
        content = handle.read()
    if '\\bibitem' in content:
        return parse_thebibliography(content)

    library_raw = _parse_bibtex_string(content)
    library = {}
    for block in _iter_bibtex_entries(library_raw.blocks):
        fields = {}
        for field in block.fields:
            fields[field.key] = field.value

        if 'title' in fields:
            fields['title'] = _normalize_citation_title(fields['title'])
        elif 'note' in fields:
            field_content = fields['note'].replace("\n", " ")
            field_content = re.sub(" +", " ", field_content)
            if field_content.find("``") != -1 and field_content.find("''") != -1:
                title = field_content[
                    field_content.find("``") + 2 : field_content.find("''")
                ]
                fields['title'] = _normalize_citation_title(title)
            elif field_content.count("\\newblock") == 2:
                field_content = field_content.replace("\\newblock", "``", 1)
                field_content = field_content.replace("\\newblock", "''", 1)
                if field_content.find("``") != -1 and field_content.find("''") != -1:
                    title = field_content[
                        field_content.find("``") + 2 : field_content.find("''")
                    ]
                    fields['title'] = _normalize_citation_title(title)
        if 'title' not in fields:
            continue
        library[block.key] = fields
    return library


def create_bib(bibfile):
    if should_skip_bibliography_file(bibfile):
        return {}
    with open(bibfile, 'r', encoding='utf-8', errors='replace') as handle:
        content = handle.read()
    library_raw = _parse_bibtex_string(content)

    library = {}
    for block in _iter_bibtex_entries(library_raw.blocks):
        fields = {}
        for field in block.fields:
            key = (field.key or "").lower()
            val = field.value.replace('{', '').replace('}', '')
            fields[key] = val
            if key == 'title':
                fields[key] = _normalize_citation_title(field.value)
        # promote arxiv id into eprint when only url/note carries it
        if not fields.get("eprint"):
            for fld in ("url", "doi", "note", "howpublished", "title"):
                aid = extract_arxiv_id_from_text(fields.get(fld, ""))
                if aid:
                    fields["eprint"] = aid
                    break
        if "title" not in fields and "eprint" not in fields:
            continue
        library[block.key] = fields
    return library


def find_bbl(directory):
    file_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            file_paths.append(file_path) 
    bib_paths = [f for f in file_paths if f.endswith('.bbl')]
    return bib_paths


def get_library_bib(bib_files):
    final_library = {}
    for bib_file in bib_files:
        final_library.update(create_bib(bib_file))
    return final_library


def get_library_bbl(bbl_files):
    final_library = {}
    for bbl_file in bbl_files:
        final_library.update(create_bib_from_bbl(bbl_file))
    return final_library
