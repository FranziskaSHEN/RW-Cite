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
_BIBITEM_PATTERN = re.compile(r'\\bibitem(?:\[[^\]]*\])?\{([^}]+)\}')
_LATEX_QUOTED_TITLE = re.compile(r'``(.*?)' + "''", re.DOTALL)


def _repo_config(rel: str) -> str:
    """Resolve configs/* against RWCITE_ROOT (cwd fallback)."""
    root = os.environ.get("RWCITE_ROOT") or os.getcwd()
    return str(Path(root) / rel)


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


def _normalize_citation_title(title):
    title = re.sub(r'[\n]+', ' ', title)
    title = re.sub(r' +', ' ', title)
    return (
        title.replace("\\emph", "")
        .replace("\\emp", "")
        .replace("\\em", "")
        .replace(",", "")
        .replace("{", "")
        .replace("}", "")
        .replace("``", "")
        .replace("''", "")
        .strip(".")
        .strip()
        .strip(".")
        .lower()
    )


def _extract_title_from_bibitem_body(body):
    body = re.sub(r'\\newblock\s*', ' ', body)
    match = _LATEX_QUOTED_TITLE.search(body)
    if match:
        return _normalize_citation_title(match.group(1))
    match = re.search(r'\\emph\{([^}]+)\}', body)
    if match:
        return _normalize_citation_title(match.group(1))
    match = re.search(r'\{\\em\s+([^}]+)\}', body)
    if match:
        return _normalize_citation_title(match.group(1))
    match = re.search(r',\s*in:\s', body, re.I)
    if match:
        prefix = body[:match.start()]
        if ',' in prefix:
            return _normalize_citation_title(prefix.split(',', 1)[1])
    match = re.search(r',\s*\\emph\{', body)
    if match:
        prefix = body[:match.start()]
        if ',' in prefix:
            return _normalize_citation_title(prefix.split(',', 1)[1])
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
        title = _extract_title_from_bibitem_body(content[start:end].strip())
        if title:
            library[key] = {'title': title}
    return library


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
    r'\\citenum': r'\\cite'
    }
    for key in pattern:
        s = re.sub(key, pattern[key], s)
    return s

def find_bib(directory):
    file_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            file_paths.append(file_path) 
    bib_paths = [f for f in file_paths if f.endswith('.bib')]
    return bib_paths

def create_bib_from_bbl(bibfile):
    with open(bibfile, 'r', encoding='utf-8', errors='replace') as handle:
        content = handle.read()
    if '\\bibitem' in content:
        return parse_thebibliography(content)

    library_raw = bibtexparser.parse_string(content)
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
    with open(bibfile, 'r', encoding='utf-8', errors='replace') as handle:
        content = handle.read()
    library_raw = bibtexparser.parse_string(content)

    library = {}
    for block in _iter_bibtex_entries(library_raw.blocks):
        fields = {}
        for field in block.fields:
            fields[field.key] = field.value.replace('{', '').replace('}', '')
            if field.key == 'title':
                fields[field.key] = _normalize_citation_title(field.value)
        if 'title' not in fields:
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
