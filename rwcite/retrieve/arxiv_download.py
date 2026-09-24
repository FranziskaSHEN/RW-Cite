"""ArXiv source download + LaTeX clean for domain graph build (no Gradio/train)."""
from __future__ import annotations

import gzip
import json
import os
import re
import shutil
import tarfile
import time
import urllib.request
from types import SimpleNamespace

from tqdm import tqdm

from rwcite.graph.graph_utils import (
    get_citing_sentences,
    get_intro,
    get_related_works,
)
from rwcite.retrieve.utils.utils import (
    check_begin,
    find_bbl,
    find_bib,
    get_library_bbl,
    get_library_bib,
    get_main,
    initial_clean,
    post_processing,
    read_tex_file,
)

# Mutable session state (set by graph pipeline before download)
config: dict = {}
config_path: str = ""
working_dir: str = ""
save_zip_directory: str = ""
save_directory: str = ""
save_description: str = ""
save_path: str = ""
save_graph: str = ""
gexf_file: str = ""
retrieval_nodes_path: str = ""
failed_downloads_path: str = "datasets/failed_downloads.json"

def normalize_arxiv_id(paper_id: str) -> tuple[str, str]:
    """Return (download_id, storage_key) for arXiv paper ids."""
    download_id = paper_id
    storage_key = paper_id
    if re.search(r"[a-zA-Z]", paper_id) is not None:
        storage_key = "".join(paper_id.split("/"))
    return download_id, storage_key


def _load_existing_download_state():
    graph = {}
    results = {
        "Number of papers": 0,
        "Number of latex papers": 0,
        "Number of bib files": 0,
        "Number of bbl files": 0,
        "Number of inline files": 0,
        "Number of introductions found": 0,
        "Number of related works found": 0,
        "Number of succesful finding of extracts": 0,
    }
    if os.path.exists(save_graph):
        with open(save_graph, encoding="utf-8") as f:
            graph = json.load(f)
    if os.path.exists(save_path):
        with open(save_path, encoding="utf-8") as f:
            results.update(json.load(f))
    return graph, results


def _count_citation_edges(graph: dict) -> int:
    return sum(len(entry.get("Citations", [])) for entry in graph.values())


FAILED_DOWNLOADS_PATH = "datasets/failed_downloads.json"


def _failed_downloads_path() -> str:
    return globals().get("failed_downloads_path", FAILED_DOWNLOADS_PATH)


def _load_failed_downloads() -> set[str]:
    path = _failed_downloads_path()
    if not os.path.exists(path):
        return set()
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return set(data.keys())


def _remove_from_retrieval_list(paper_id: str) -> None:
    path = globals().get("retrieval_nodes_path", "datasets/retrieval_nodes.json")
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    download_id, storage_key = normalize_arxiv_id(paper_id)
    for key in (paper_id, download_id, storage_key):
        data.pop(key, None)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)


def _mark_paper_failed(paper_id: str, reason: str) -> None:
    path = _failed_downloads_path()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    record = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            record = json.load(f)
    download_id, storage_key = normalize_arxiv_id(paper_id)
    entry = {"reason": reason, "time": time.strftime("%Y-%m-%dT%H:%M:%S")}
    for key in (paper_id, download_id, storage_key):
        record[key] = entry
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)
    _remove_from_retrieval_list(paper_id)
    print(f"Removed failed paper from retrieval list: {download_id} ({reason})", flush=True)


def filter_retrieval_nodes_file() -> int:
    """Drop failed paper ids from retrieval_nodes.json. Returns removed count."""
    path = globals().get("retrieval_nodes_path", "datasets/retrieval_nodes.json")
    failed_ids = _load_failed_downloads()
    if not failed_ids or not os.path.exists(path):
        return 0
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    before = len(data)
    filtered = {}
    for paper_id, value in data.items():
        _, storage_key = normalize_arxiv_id(paper_id)
        if paper_id in failed_ids or storage_key in failed_ids:
            continue
        filtered[paper_id] = value
    if len(filtered) != before:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(filtered, f)
    return before - len(filtered)


def _is_valid_tar_archive(tar_file_path: str) -> bool:
    if not os.path.exists(tar_file_path) or os.path.getsize(tar_file_path) == 0:
        return False
    try:
        with tarfile.open(tar_file_path) as tar:
            tar.getmembers()
        return True
    except Exception:
        return False


def _has_local_download(tar_file_path: str, extracted_dir: str) -> bool:
    if os.path.isdir(extracted_dir) and os.listdir(extracted_dir):
        return True
    return _is_valid_tar_archive(tar_file_path)


def _download_arxiv_source(
    download_id: str,
    tar_file_path: str,
    timeout_sec: int,
    max_bytes: int | None,
    retries: int,
) -> bool:
    url = "https://arxiv.org/src/" + download_id
    for attempt in range(1, retries + 1):
        tmp_path = tar_file_path + ".partial"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "RW-Cite/1.0"})
            with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
                content_length = resp.headers.get("Content-Length")
                if max_bytes and content_length and int(content_length) > max_bytes:
                    _mark_paper_failed(download_id, "oversized_src")
                    print(
                        f"Skipping oversized paper {download_id}: "
                        f"{int(content_length) / 1e6:.1f} MB > limit",
                        flush=True,
                    )
                    return False
                total = 0
                with open(tmp_path, "wb") as out:
                    while True:
                        chunk = resp.read(1024 * 1024)
                        if not chunk:
                            break
                        total += len(chunk)
                        if max_bytes and total > max_bytes:
                            raise ValueError(
                                f"download exceeded max size ({max_bytes} bytes)"
                            )
                        out.write(chunk)
            os.replace(tmp_path, tar_file_path)
            if not _is_valid_tar_archive(tar_file_path):
                raise tarfile.ReadError("downloaded tar failed validation")
            return True
        except Exception as exc:
            for path in (tmp_path, tar_file_path):
                if os.path.exists(path):
                    os.remove(path)
            if attempt >= retries:
                _mark_paper_failed(download_id, f"download_failed:{type(exc).__name__}")
            print(
                f"Couldn't download paper {download_id} "
                f"(attempt {attempt}/{retries}): {type(exc).__name__}: {exc}",
                flush=True,
            )
            if attempt < retries:
                time.sleep(min(attempt * 2, 10))
    return False


def _extract_paper_archive(tar_file_path: str, extracted_dir: str, storage_key: str) -> bool:
    if os.path.isdir(extracted_dir) and os.listdir(extracted_dir):
        return True
    if not _is_valid_tar_archive(tar_file_path):
        if os.path.exists(tar_file_path):
            os.remove(tar_file_path)
        return False
    if os.path.exists(extracted_dir):
        shutil.rmtree(extracted_dir)
    os.makedirs(extracted_dir, exist_ok=True)
    try:
        with tarfile.open(tar_file_path) as tar:
            tar.extractall(extracted_dir)
        return True
    except Exception:
        try:
            with gzip.open(tar_file_path, "rb") as f:
                file_content = f.read()
            with open(extracted_dir + storage_key + ".tex", "w", encoding="utf-8") as f:
                f.write(file_content.decode())
            return True
        except Exception:
            print(f"Could not extract paper id: {storage_key}", flush=True)
            if os.path.exists(tar_file_path):
                os.remove(tar_file_path)
            _mark_paper_failed(storage_key, "extract_failed")
            return False


# Fetch and store arXiv source files
def fetch_arxiv_papers(papers_to_download):
    # Download the arXiv metadata file if it doesn't exist
    dataset = 'datasets/arxiv-metadata-oai-snapshot.json'
    data = []
    if not os.path.exists(dataset):
        os.system("wget https://huggingface.co/spaces/ddiddu/simsearch/resolve/main/arxiv-metadata-oai-snapshot.json -P ./datasets")

    with open(dataset, 'r') as f:
        for line in f:
            data.append(json.loads(line))

    papers = [d for d in data]
    paper_ids = [d['id'] for d in data]
    paper_titles = [
        (
            re.sub(r' +', ' ', re.sub(r'[\n]+', ' ', paper['title']))
            .replace("\\emph", "")
            .replace("\\emp", "")
            .replace("\\em", "")
            .replace(",", "")
            .replace("{", "")
            .replace("}", "")
            .strip(".")
            .strip()
            .strip(".")
            .lower()
        )
        for paper in papers
    ]
    paper_dict = {
        k:v
        for k,v in zip(paper_titles, paper_ids)
    }


    total_papers = len(papers_to_download)
    download_progress_bar=lambda _x: None
    skip_if_downloaded = config["data_downloading"]["processing"].get("skip_if_downloaded", True)
    graph, results = _load_existing_download_state()
    failed_ids = _load_failed_downloads()
    if failed_ids:
        papers_to_download = [
            paper_id
            for paper_id in papers_to_download
            if paper_id not in failed_ids
            and normalize_arxiv_id(paper_id)[1] not in failed_ids
        ]
        total_papers = len(papers_to_download)
    num_papers = 0
    num_edges = _count_citation_edges(graph)
    skipped_processed = 0
    skipped_download = 0
    newly_downloaded = 0
    
    llm_resp = []
    t, iter_ind = 0, 0

    arxiv_rate_lim = config['data_downloading']['processing']['arxiv_rate_limit']
    download_timeout = config['data_downloading']['processing'].get('download_timeout_sec', 120)
    max_tar_bytes = config['data_downloading']['processing'].get('max_tar_bytes', 104857600)
    download_retries = config['data_downloading']['processing'].get('download_retries', 2)

    for paper_name in tqdm(papers_to_download):
        paper_name_download, storage_key = normalize_arxiv_id(paper_name)
        tar_file_path = save_zip_directory + storage_key + '.tar.gz'
        extracted_dir = save_directory + storage_key + '/'

        if storage_key in graph:
            skipped_processed += 1
            num_papers += 1
            pass  # progress
            continue

        print(
            "Number of papers processed: {} | edges: {} | prev_iter_s: {:.1f} | now: {}".format(
                num_papers, num_edges, time.time()-t, paper_name_download
            ),
            flush=True,
        )
        t = time.time()
        num_papers += 1
        results["Number of papers"] += 1

        downloaded_this_iter = False
        t1 = time.time()
        if skip_if_downloaded and _has_local_download(tar_file_path, extracted_dir):
            skipped_download += 1
        elif not _download_arxiv_source(
            paper_name_download,
            tar_file_path,
            timeout_sec=download_timeout,
            max_bytes=max_tar_bytes,
            retries=download_retries,
        ):
            continue
        else:
            newly_downloaded += 1
            downloaded_this_iter = True

        if not _extract_paper_archive(tar_file_path, extracted_dir, storage_key):
            continue

        try:
            # Perform initial cleaning and get the main TeX file
            initial_clean(extracted_dir, config=False)
            main_file = get_main(extracted_dir)

            # If no main TeX file is found, remove the downloaded archive and continue
            if main_file == None:
                print("No tex files found", flush=True)
                if downloaded_this_iter and os.path.exists(tar_file_path):
                    os.remove(tar_file_path)
                _mark_paper_failed(paper_name_download, "no_tex_files")
                continue

            # Check if the main TeX file contains a valid LaTeX document
            h = check_begin(main_file)
            if h == True:
                results["Number of latex papers"] += 1
                # Flag to check for internal bibliography
                check_internal = 0
                # Dictionary to store bibliographic references
                final_library = {}

                # Identify bibliography files (.bib or .bbl)
                bib_files = find_bib(extracted_dir)
                if bib_files == []:
                    bbl_files = find_bbl(extracted_dir)
                    if bbl_files == []:
                        # No external bibliography found
                        check_internal = 1
                    else:
                        final_library = get_library_bbl(bbl_files)
                        results["Number of bbl files"] += 1
                else:
                    results["Number of bib files"] += 1
                    final_library = get_library_bib(bib_files)

                # Apply post-processing to clean the TeX document
                main_file = post_processing(extracted_dir, main_file)

                # Read the cleaned LaTeX document content
                descr = main_file
                content = read_tex_file(descr)

                # If configured, store the raw content in the graph
                if config['data_downloading']['processing']['keep_unstructured_content']:
                    graph[storage_key] = {'content': content}
                else:
                    graph[storage_key] = {}

                # Check for inline bibliography within the LaTeX document
                if check_internal == 1:
                    beginning_bib = '\\begin{thebibliography}'
                    end_bib = '\\end{thebibliography}'

                    if content.find(beginning_bib) != -1 and content.find(end_bib) != -1:
                        bibliography = content[content.find(beginning_bib):content.find(end_bib) + len(end_bib)]
                        save_bbl = os.path.join(extracted_dir, "bibliography.bbl")

                        results["Number of inline files"] += 1
                        with open(save_bbl, "w") as f:
                            f.write(bibliography)

                        final_library = get_library_bbl([save_bbl])

                # If no valid bibliography is found, skip processing citations
                if final_library == {}:
                    print("No library found...")
                    continue

                # Extract relevant sections such as "Related Work" and "Introduction"
                related_works = get_related_works(content)
                if related_works  != '':
                    graph[storage_key]['Related Work'] = related_works
                    results["Number of related works found"] += 1
                
                intro = get_intro(content)
                if intro  != '':
                    graph[storage_key]['Introduction'] = intro
                    results["Number of introductions found"] += 1

                # Extract citation sentences from the introduction and related works
                sentences_citing = get_citing_sentences(intro + '\n' + related_works)
                
                # Map citations to corresponding papers
                raw_sentences_citing = {}
                for k,v in sentences_citing.items():
                    new_values = []
                    for item in v:
                        try:
                            new_values.append(paper_dict[final_library[item]['title']])
                        except Exception as e:
                            pass
                    if new_values != []:
                        raw_sentences_citing[k] = new_values

                # Construct citation edges
                edges_set = []
                for k,v in raw_sentences_citing.items():
                    for item in v:
                        edges_set.append((paper_name_download, item, {"sentence":k}))

                iter_ind +=1
                if len(edges_set) !=0:
                    results["Number of succesful finding of extracts"] += 1
                    graph[storage_key]['Citations'] = edges_set
                    num_edges += len(edges_set)

                # Save progress after every 10 iterations
                if iter_ind % 10 == 0:
                    print("Saving graph now")
                    with open(save_path, 'w') as f:
                        json.dump(results, f)
                    with open(save_graph, 'w') as f:
                        json.dump(graph, f)
        
        except Exception as e:
            print(
                "Could not get main paper {}: {}: {}".format(
                    storage_key, type(e).__name__, e
                )
            )

        # Update the progress bar after processing each paper
        pass  # progress

        if downloaded_this_iter:
            t2 = time.time()
            elapsed_time = t2 - t1
            if elapsed_time < arxiv_rate_lim:
                time.sleep(arxiv_rate_lim - elapsed_time)


    # Final saving of processed data
    with open(save_graph, 'w') as f:
        json.dump(graph, f)
    with open(save_path, 'w') as f:
        json.dump(results, f)


    # Log final completion message
    llm_resp.append(
        "Incremental summary: skipped_processed={} skipped_download={} newly_downloaded={}".format(
            skipped_processed, skipped_download, newly_downloaded
        )
    )
    llm_resp.append("Successfully downloaded and cleaned {} papers.".format(results["Number of latex papers"]))
    return "\n".join(llm_resp)


