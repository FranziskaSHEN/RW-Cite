#!/usr/bin/env python3
r"""Rebuild RR citation graph (native seed / expand).

Uses ``rwcite.retrieve`` helpers and domain-local zip when available.
Optional readonly zip via ``RWCITE_DATA_ROOT`` / ``RWCITE_ASSET_SOURCE``.

Pipeline:
  1) Seed from existing GEXF (readonly) — keep historical edges/nodes.
  2) Re-parse Intro+RW \cite using final_cleaned.tex + bibliography from
     optional readonly tar.gz zip dir (or local .bbl/.bib).
  3) Domain closure: missing citee arXiv ids enter the graph via metadata;
     optional expand rounds download/parse more papers (zip or arxiv.org).
  4) Write new GEXF (default description/test_graph_rr.gexf).

Does not truncate gold (dataset concern).
"""

from __future__ import annotations

import argparse
import io
import json
import random
import re
import shutil
import sys
import tarfile
import tempfile
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import networkx as nx
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
from rwcite.graph.graph_utils import (  # noqa: E402
    get_citing_sentences,
    get_intro,
    get_intro_related_for_rr,
    get_related_works,
)
from rwcite.latex.utils import (  # noqa: E402
    ID_BEARING_FIELDS,
    _normalize_citation_title,
    create_bib,
    create_bib_from_bbl,
    parse_thebibliography,
    prepare_rr_tex,
    read_tex_file,
    resolve_citation_entry,
    select_main_tex,
    should_skip_bibliography_file,
)

_ARXIV_ID_RE = re.compile(
    r"^(?:\d{4}\.\d{4,5}|[a-z-]+(?:\.[A-Z]{2})?/\d{7})$", re.I
)


def _norm_id(pid: str) -> str:
    return str(pid).strip()


def _paper_dir(papers_dir: Path, pid: str) -> Path:
    pid = _norm_id(pid)
    for cand in (papers_dir / pid, papers_dir / pid.replace("/", "")):
        if cand.is_dir():
            return cand
    return papers_dir / pid.replace("/", "")


def _zip_path(zip_dir: Path | None, pid: str) -> Path | None:
    if zip_dir is None or not zip_dir.is_dir():
        return None
    pid = _norm_id(pid)
    for name in (f"{pid}.tar.gz", f"{pid.replace('/', '')}.tar.gz"):
        p = zip_dir / name
        if p.is_file():
            return p
    return None


DEFAULT_METADATA = "datasets/arxiv-metadata-oai-snapshot.json"

# path -> (title_index, known_ids, by_id)
_META_SOURCE_CACHE: dict[str, tuple[dict[str, str], set[str], dict[str, dict[str, str]]]] = {}


def _resolve_metadata_path(meta_path: Path | str | None) -> Path:
    p = Path(meta_path) if meta_path else ROOT / DEFAULT_METADATA
    if not p.is_absolute():
        p = ROOT / p
    return p


def _load_jsonl_metadata(
    meta_path: Path,
) -> tuple[dict[str, str], set[str], dict[str, dict[str, str]]]:
    """Scan arXiv OAI JSONL once: title index + id→{title,abstract}."""
    print(f"loading metadata from {meta_path} ...", flush=True)
    title_index: dict[str, str] = {}
    known_ids: set[str] = set()
    by_id: dict[str, dict[str, str]] = {}
    n = 0
    with open(meta_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            pid_s = _norm_id(entry.get("id") or "")
            if not pid_s:
                continue
            title = entry.get("title") or ""
            abstract = entry.get("abstract") or ""
            known_ids.add(pid_s)
            by_id[pid_s] = {"title": title, "abstract": abstract}
            nt = _normalize_citation_title(title)
            if nt and len(nt.split()) >= 2:
                title_index[nt] = pid_s
            n += 1
            if n % 500_000 == 0:
                print(
                    f"  scanned {n} papers, index={len(title_index)}",
                    flush=True,
                )
    print(
        f"metadata ready: {len(title_index)} title keys / {len(known_ids)} ids "
        f"from {n} rows",
        flush=True,
    )
    return title_index, known_ids, by_id


def _get_meta_source(
    meta_path: Path | str | None,
) -> tuple[Path, dict[str, str], set[str], dict[str, dict[str, str]]]:
    path = _resolve_metadata_path(meta_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"missing arXiv metadata JSONL: {path} "
            "(run scripts/run_metadata_fetch.sh or bootstrap_assets.sh)"
        )
    key = str(path.resolve())
    cached = _META_SOURCE_CACHE.get(key)
    if cached is None:
        cached = _load_jsonl_metadata(path)
        _META_SOURCE_CACHE[key] = cached
    title_index, known_ids, by_id = cached
    return path, title_index, known_ids, by_id


def load_title_index(
    meta_path: Path | str | None = None,
) -> tuple[dict[str, str], set[str]]:
    """Return (normalized title -> arxiv id, known id set) from JSONL snapshot."""
    _, title_index, known_ids, _ = _get_meta_source(meta_path)
    return title_index, known_ids


def load_meta_rows(
    meta_path: Path | str | None, ids: Iterable[str]
) -> dict[str, dict[str, str]]:
    need = [_norm_id(i) for i in ids]
    if not need:
        return {}
    _, _, _, by_id = _get_meta_source(meta_path)
    return {i: by_id[i] for i in need if i in by_id}


def library_from_tex_and_dir(tex_content: str, paper_dir: Path) -> dict[str, dict]:
    lib: dict[str, dict] = {}
    if "\\bibitem" in tex_content or "thebibliography" in tex_content:
        lib.update(parse_thebibliography(tex_content))
    for bbl in paper_dir.glob("*.bbl"):
        try:
            lib.update(create_bib_from_bbl(str(bbl)))
        except Exception:  # noqa: BLE001
            pass
    for bib in paper_dir.glob("*.bib"):
        try:
            lib.update(create_bib(str(bib)))
        except Exception:  # noqa: BLE001
            pass
    return lib


def library_from_zip(zip_path: Path) -> dict[str, dict]:
    lib: dict[str, dict] = {}
    try:
        with tarfile.open(zip_path, "r:gz") as tf:
            for m in tf.getmembers():
                name = m.name.lower()
                if not m.isfile():
                    continue
                if not (name.endswith(".bbl") or name.endswith(".bib")):
                    continue
                # Skip ACL anthology dumps / other mega-bibs before decoding.
                if should_skip_bibliography_file(m.name, size=int(m.size or 0)):
                    continue
                f = tf.extractfile(m)
                if f is None:
                    continue
                raw = f.read().decode("utf-8", errors="replace")
                try:
                    if name.endswith(".bbl") or "\\bibitem" in raw:
                        # write temp for create_bib_from_bbl API
                        with tempfile.NamedTemporaryFile(
                            "w", suffix=".bbl", delete=False, encoding="utf-8"
                        ) as tmp:
                            tmp.write(raw)
                            tmp_path = tmp.name
                        lib.update(create_bib_from_bbl(tmp_path))
                        Path(tmp_path).unlink(missing_ok=True)
                    else:
                        with tempfile.NamedTemporaryFile(
                            "w", suffix=".bib", delete=False, encoding="utf-8"
                        ) as tmp:
                            tmp.write(raw)
                            tmp_path = tmp.name
                        lib.update(create_bib(tmp_path))
                        Path(tmp_path).unlink(missing_ok=True)
                except Exception:  # noqa: BLE001
                    continue
    except Exception as e:  # noqa: BLE001
        print(f"  zip library fail {zip_path.name}: {e}", flush=True)
    return lib


def extract_edges_for_paper(
    pid: str,
    papers_dir: Path,
    zip_dir: Path | None,
    title_index: dict[str, str],
    known_ids: set[str] | None = None,
) -> tuple[list[tuple[str, str, str]], dict[str, Any]]:
    """Return edges (src, tgt, sentence) and stats."""
    stats = {
        "has_tex": False,
        "lib_keys": 0,
        "cite_keys": 0,
        "resolved": 0,
        "unresolved": 0,
        "resolved_by_id": 0,
        "resolved_by_title": 0,
    }
    pdir = _paper_dir(papers_dir, pid)
    tex_path = pdir / "final_cleaned.tex"
    if not tex_path.is_file():
        return [], stats
    stats["has_tex"] = True
    try:
        content = read_tex_file(str(tex_path))
    except Exception:  # noqa: BLE001
        return [], stats
    # Raw / partially cleaned sources may still \input{introduction} etc.
    content = prepare_rr_tex(content, str(pdir))

    lib = library_from_tex_and_dir(content, pdir)
    zpath = _zip_path(zip_dir, pid)
    if zpath is not None:
        lib.update(library_from_zip(zpath))
    # also scan nested bib/bbl under materialized tree
    for bbl in pdir.rglob("*.bbl"):
        try:
            lib.update(create_bib_from_bbl(str(bbl)))
        except Exception:  # noqa: BLE001
            pass
    for bib in pdir.rglob("*.bib"):
        try:
            lib.update(create_bib(str(bib)))
        except Exception:  # noqa: BLE001
            pass
    stats["lib_keys"] = len(lib)
    if not lib:
        return [], stats

    intro, related = get_intro_related_for_rr(content)
    citing = get_citing_sentences(intro + "\n" + related)
    edges: list[tuple[str, str, str]] = []
    seen: set[tuple[str, str]] = set()
    src = _norm_id(pid)
    for sent, keys in citing.items():
        for key in keys:
            stats["cite_keys"] += 1
            key = str(key).strip()
            entry = lib.get(key)
            # allow eprint-only entries (no title)
            if not entry or not (entry.get("title") or entry.get("eprint") or entry.get("url")):
                stats["unresolved"] += 1
                continue
            tid_via_id = resolve_citation_entry(
                {
                    k: entry.get(k)
                    for k in ID_BEARING_FIELDS
                    if k != "title" and entry.get(k)
                }
                or None,
                {},  # id-only pass
                known_ids,
            )
            tid = tid_via_id or resolve_citation_entry(entry, title_index, known_ids)
            if not tid:
                stats["unresolved"] += 1
                continue
            tid = _norm_id(tid)
            if tid == src:
                continue
            pair = (src, tid)
            if pair in seen:
                continue
            seen.add(pair)
            sent_s = str(sent).strip()
            try:
                from rwcite.latex.cite_sentence_clean import clean_cite_sentence

                cres = clean_cite_sentence(sent_s)
                if cres.text:
                    sent_s = cres.text
            except Exception:  # noqa: BLE001
                pass
            edges.append((src, tid, sent_s))
            stats["resolved"] += 1
            if tid_via_id:
                stats["resolved_by_id"] += 1
            else:
                stats["resolved_by_title"] += 1
    return edges, stats

def seed_graph_from_gexf(path: Path) -> nx.DiGraph:
    g0 = nx.read_gexf(str(path), node_type=None, relabel=False, version="1.2draft")
    g = nx.DiGraph()
    for n, attrs in g0.nodes(data=True):
        g.add_node(
            _norm_id(n),
            title=attrs.get("title") or "",
            abstract=attrs.get("abstract") or "",
            introduction=attrs.get("introduction") or "",
            related=attrs.get("related") or "",
            concepts=attrs.get("concepts") or "",
            label=attrs.get("label") or "",
        )
    for u, v, attrs in g0.edges(data=True):
        g.add_edge(
            _norm_id(u),
            _norm_id(v),
            sentence=attrs.get("sentence") or attrs.get("label") or "",
        )
    return g


def ensure_nodes(
    g: nx.DiGraph,
    ids: Iterable[str],
    meta_path: Path | str | None,
    papers_cache: dict[str, dict],
) -> int:
    need = [i for i in {_norm_id(x) for x in ids} if i not in g]
    if not need:
        return 0
    missing_meta = [i for i in need if i not in papers_cache]
    if missing_meta:
        papers_cache.update(load_meta_rows(meta_path, missing_meta))
    added = 0
    for i in need:
        meta = papers_cache.get(i)
        if meta is None:
            # still add stub so edge can exist if we know id
            g.add_node(i, title=i, abstract="", introduction="", related="", concepts="", label="")
            added += 1
            continue
        g.add_node(
            i,
            title=meta.get("title") or i,
            abstract=meta.get("abstract") or "",
            introduction="",
            related="",
            concepts="",
            label="",
        )
        added += 1
    return added


def add_edges(g: nx.DiGraph, edges: list[tuple[str, str, str]]) -> tuple[int, int]:
    new_e = 0
    for s, t, sent in edges:
        if s not in g or t not in g:
            continue
        if g.has_edge(s, t):
            # keep existing sentence if non-empty
            if not (g.edges[s, t].get("sentence") or "").strip() and sent:
                g.edges[s, t]["sentence"] = sent
            continue
        g.add_edge(s, t, sentence=sent)
        new_e += 1
    return new_e, len(edges)


def download_arxiv_src(
    pid: str, dest_tar: Path, *, timeout: float = 90.0
) -> bool:
    """Fetch arXiv source tarball. ``timeout`` avoids indefinite hangs on dead sockets."""
    url = f"https://arxiv.org/src/{pid}"
    try:
        dest_tar.parent.mkdir(parents=True, exist_ok=True)
        if dest_tar.exists():
            dest_tar.unlink(missing_ok=True)
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "RW-Cite-enrich/1.0 (mailto:devnull@local)"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp, open(
            dest_tar, "wb"
        ) as out:
            shutil.copyfileobj(resp, out, length=1024 * 256)
        ok = dest_tar.is_file() and dest_tar.stat().st_size > 0
        if not ok and dest_tar.exists():
            dest_tar.unlink(missing_ok=True)
        return ok
    except Exception as e:  # noqa: BLE001
        print(f"  download fail {pid}: {e}", flush=True)
        try:
            dest_tar.unlink(missing_ok=True)
        except OSError:
            pass
        return False


def materialize_paper_from_zip(
    pid: str, zip_path: Path, papers_dir: Path
) -> bool:
    """Extract tex/bbl/bib into papers_dir/pid and build expanded final_cleaned.tex."""
    pdir = _paper_dir(papers_dir, pid)
    pdir.mkdir(parents=True, exist_ok=True)
    try:
        with tarfile.open(zip_path, "r:gz") as tf:
            for m in tf.getmembers():
                if not m.isfile():
                    continue
                # keep relative path when possible so \\input{dir/x} resolves
                name = m.name
                if name.startswith("./"):
                    name = name[2:]
                # skip macOS junk / absolute-ish
                if not name or name.startswith("/") or "__MACOSX" in name:
                    continue
                base = Path(name).name
                low = base.lower()
                if not (
                    low.endswith(".tex")
                    or low.endswith(".bbl")
                    or low.endswith(".bib")
                ):
                    continue
                f = tf.extractfile(m)
                if f is None:
                    continue
                data = f.read()
                out = pdir / Path(name)
                # flatten if path escapes
                try:
                    out.resolve().relative_to(pdir.resolve())
                except Exception:  # noqa: BLE001
                    out = pdir / base
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_bytes(data)

        main = select_main_tex(str(pdir))
        if main is None:
            # fallback: largest non-cleaned tex
            texs = sorted(
                (p for p in pdir.rglob("*.tex") if p.name != "final_cleaned.tex"),
                key=lambda p: p.stat().st_size,
                reverse=True,
            )
            main = str(texs[0]) if texs else None
        if main is None:
            return False
        raw = read_tex_file(main)
        expanded = prepare_rr_tex(raw, str(Path(main).parent))
        # also allow inputs relative to paper root
        if expanded == raw or "\\input" in expanded or "\\include" in expanded:
            expanded = prepare_rr_tex(raw, str(pdir))
        (pdir / "final_cleaned.tex").write_text(expanded, encoding="utf-8")
        return (pdir / "final_cleaned.tex").exists()
    except Exception as e:  # noqa: BLE001
        print(f"  materialize fail {pid}: {e}", flush=True)
        return False


def graph_stats(g: nx.DiGraph) -> dict[str, Any]:
    ods = [g.out_degree(n) for n in g.nodes]
    ids = [g.in_degree(n) for n in g.nodes]
    elig = [n for n in g.nodes if g.out_degree(n) >= 10]
    edged = [n for n in g.nodes if g.out_degree(n) >= 1]
    sent_ok = sum(
        1
        for _, _, a in g.edges(data=True)
        if str(a.get("sentence") or "").strip()
    )
    import numpy as np

    def pct(arr, p):
        if not arr:
            return 0.0
        return float(np.percentile(arr, p))

    return {
        "nodes": g.number_of_nodes(),
        "edges": g.number_of_edges(),
        "sources_outdeg_ge1": len(edged),
        "sources_outdeg_ge10": len(elig),
        "outdeg_p50_elig10": pct([g.out_degree(n) for n in elig], 50) if elig else 0,
        "outdeg_p90_elig10": pct([g.out_degree(n) for n in elig], 90) if elig else 0,
        "outdeg_max": max(ods) if ods else 0,
        "indeg_p50_all": pct(ids, 50),
        "edges_with_sentence": sent_ok,
        "edges_with_sentence_rate": sent_ok / max(1, g.number_of_edges()),
    }


def expand_budget(
    n_candidates: int,
    *,
    expand_frac: float,
    max_expand_per_round: int,
    min_expand_per_round: int = 0,
) -> int:
    """How many zero-outdeg citees to expand this round.

    Primary control is ``expand_frac`` × |candidates|. ``max_expand_per_round``
    is an optional hard cap (0 = no cap). ``min_expand_per_round`` floors tiny
    graphs when frac rounds down to 0.
    """
    if n_candidates <= 0:
        return 0
    if expand_frac > 0:
        n = int(round(n_candidates * expand_frac))
    else:
        # Legacy: fixed count when frac disabled
        n = max_expand_per_round if max_expand_per_round > 0 else 0
    if min_expand_per_round > 0:
        n = max(n, min_expand_per_round)
    if max_expand_per_round > 0:
        n = min(n, max_expand_per_round)
    return max(0, min(n, n_candidates))


def _expand_sim_scores(
    queries: Sequence[str],
    candidate_ids: Sequence[str],
    *,
    embedder: str = "models/base/bge-large-en-v1.5",
) -> dict[str, float]:
    """Best cosine of each candidate to domain query prototypes (topic embeds)."""
    from transformers import AutoModel, AutoTokenizer

    from rwcite.retrieve.retriever.corpus_utils import load_merged_embeddings
    from rwcite.graph.domain_retrieve import _encode_queries, merge_prototype_scores

    q_list = [str(q).strip() for q in queries if str(q).strip()]
    if not q_list or not candidate_ids:
        return {}
    load_kwargs = {"local_files_only": True}
    try:
        tokenizer = AutoTokenizer.from_pretrained(embedder, **load_kwargs)
        model = AutoModel.from_pretrained(embedder, **load_kwargs)
    except OSError:
        tokenizer = AutoTokenizer.from_pretrained(embedder)
        model = AutoModel.from_pretrained(embedder)
    model = model.to(device="cuda", dtype=torch.float16)
    q_embs = _encode_queries(tokenizer, model, q_list)
    merged_ids, all_embs = load_merged_embeddings(
        use_hf_embeds=True, cache_dir="datasets/topic_level_embeds"
    )
    id_to_emb = dict(zip(merged_ids, all_embs))
    out: dict[str, float] = {}
    need = [_norm_id(c) for c in candidate_ids]
    present = [c for c in need if c in id_to_emb]
    if present:
        mat = np.stack([id_to_emb[c] for c in present])
        best = merge_prototype_scores(q_embs, mat)
        for c, sc in zip(present, best):
            out[c] = float(sc)
    for c in need:
        out.setdefault(c, -1.0)
    return out


def build_native_seed_graph(
    paper_ids: Sequence[str],
    papers_dir: Path | str,
    *,
    zip_dir: Path | str | None = None,
    metadata_path: Path | str | None = None,
    metadata_db: Path | str | None = None,  # alias of metadata_path
) -> tuple[nx.DiGraph, dict[str, Any]]:
    """Seed GEXF via native extractor + ensure_nodes.

    Node insertion: retrieval order for sources, then citees on first edge.
    """
    papers_dir = Path(papers_dir)
    if not papers_dir.is_absolute():
        papers_dir = ROOT / papers_dir
    meta_path = metadata_path if metadata_path is not None else metadata_db
    meta_path = _resolve_metadata_path(meta_path)
    zip_dir_p: Path | None = None
    if zip_dir is not None and str(zip_dir).strip():
        zip_dir_p = Path(zip_dir)
        if not zip_dir_p.is_absolute():
            zip_dir_p = ROOT / zip_dir_p

    title_index, known_ids = load_title_index(meta_path)
    papers_cache: dict[str, dict] = {}
    g = nx.DiGraph()
    ids = [_norm_id(p) for p in paper_ids]
    ensure_nodes(g, ids, meta_path, papers_cache)

    new_edges = 0
    parse_stats: dict[str, int] = defaultdict(int)
    n_with_tex = 0
    for i, pid in enumerate(ids):
        edges, st = extract_edges_for_paper(
            pid, papers_dir, zip_dir_p, title_index, known_ids
        )
        for k, v in st.items():
            parse_stats[k] += int(v) if isinstance(v, (int, bool)) else 0
        if st.get("has_tex"):
            n_with_tex += 1
        if edges:
            ensure_nodes(g, [t for _, t, _ in edges], meta_path, papers_cache)
            ne, _ = add_edges(g, edges)
            new_edges += ne
        if (i + 1) % 200 == 0 or i + 1 == len(ids):
            print(
                f"  native seed {i+1}/{len(ids)} edges={new_edges} "
                f"nodes={g.number_of_nodes()} with_tex={n_with_tex}",
                flush=True,
            )
    iso = list(nx.isolates(g))
    g.remove_nodes_from(iso)
    report = {
        "seed_mode": "native",
        "n_retrieval_ids": len(ids),
        "n_with_tex": n_with_tex,
        "new_edges": new_edges,
        "removed_isolates": len(iso),
        "parse_stats": dict(parse_stats),
        "stats": graph_stats(g),
    }
    return g, report


def enrich_citation_graph(
    gexf_in: Path | str,
    gexf_out: Path | str,
    papers_dir: Path | str,
    *,
    zip_dir: Path | str | None = None,
    metadata_path: Path | str | None = None,
    metadata_db: Path | str | None = None,  # alias of metadata_path
    expand_rounds: int = 2,
    expand_frac: float = 0.2,
    max_expand_per_round: int = 0,
    min_expand_per_round: int = 0,
    max_reparse: int = 0,
    allow_download: bool = False,
    seed: int = 0,
    report_path: Path | str | None = None,
    expand_sim_tau: float = 0.0,
    expand_queries: Sequence[str] | None = None,
    expand_embedder: str = "models/base/bge-large-en-v1.5",
) -> dict[str, Any]:
    """Reparse Intro/RW cites + optional citee expand; write enriched GEXF.

    Callable from the domain graph pipeline or CLI. Paths may be relative to ROOT.
    Expand size defaults to ``expand_frac`` of zero-outdeg candidates per round.

    ``max_reparse``: ``0`` = all papers with tex (legacy); ``-1`` = skip reparse
    (default after native seed); ``>0`` = subsample that many.
    ``expand_sim_tau``: if >0, keep expand candidates with topic-embed cosine to
    ``expand_queries`` ≥ tau (missing corpus emb → rejected).
    """
    gexf_in = Path(gexf_in)
    gexf_out = Path(gexf_out)
    papers_dir = Path(papers_dir)
    if not gexf_in.is_absolute():
        gexf_in = ROOT / gexf_in
    if not gexf_out.is_absolute():
        gexf_out = ROOT / gexf_out
    if not papers_dir.is_absolute():
        papers_dir = ROOT / papers_dir

    meta_path = metadata_path if metadata_path is not None else metadata_db
    meta_path = _resolve_metadata_path(meta_path)

    if zip_dir is not None and str(zip_dir).strip():
        zip_dir_p: Path | None = Path(zip_dir)
        if not zip_dir_p.is_absolute():
            zip_dir_p = ROOT / zip_dir_p
    else:
        zip_dir_p = None
        # Prefer domain-local zip; optional readonly fallback via RWCITE_DATA_ROOT
        local_zip = papers_dir.parent / "research_papers_zip"
        if local_zip.is_dir():
            zip_dir_p = local_zip
            print(f"using domain zip-dir {zip_dir_p}", flush=True)
        else:
            import os

            data_root = os.environ.get("RWCITE_DATA_ROOT") or os.environ.get(
                "RWCITE_ASSET_SOURCE"
            )
            if data_root:
                guess = (
                    Path(data_root)
                    / papers_dir.parent.name
                    / "research_papers_zip"
                )
                if guess.is_dir():
                    zip_dir_p = guess
                    print(f"using readonly zip-dir {zip_dir_p}", flush=True)

    report: dict[str, Any] = {"steps": []}

    print(f"seed from {gexf_in}", flush=True)
    g = seed_graph_from_gexf(gexf_in)
    old_stats = graph_stats(g)
    report["old"] = old_stats
    print("old", old_stats, flush=True)

    title_index, known_ids = load_title_index(meta_path)
    papers_cache: dict[str, dict] = {}

    skip_reparse = int(max_reparse) < 0
    tex_pids: list[str] = []
    new_edges_total = 0
    parse_stats: dict[str, int] = defaultdict(int)
    if skip_reparse:
        print("reparse skipped (max_reparse<0; native seed path)", flush=True)
        report["steps"].append(
            {
                "name": "reparse_existing",
                "papers": 0,
                "new_edges": 0,
                "skipped": True,
                "reason": "max_reparse<0",
            }
        )
    else:
        tex_pids = sorted(
            d.name
            for d in papers_dir.iterdir()
            if d.is_dir() and (d / "final_cleaned.tex").is_file()
        )
        rng = random.Random(seed)
        if max_reparse and max_reparse < len(tex_pids):
            rng.shuffle(tex_pids)
            tex_pids = tex_pids[:max_reparse]
        print(f"reparse {len(tex_pids)} papers ...", flush=True)

        for i, pid in enumerate(tex_pids):
            edges, st = extract_edges_for_paper(
                pid, papers_dir, zip_dir_p, title_index, known_ids
            )
            for k, v in st.items():
                parse_stats[k] += int(v) if isinstance(v, (int, bool)) else 0
            if edges:
                tgts = [t for _, t, _ in edges]
                ensure_nodes(g, [pid] + tgts, meta_path, papers_cache)
                ne, _ = add_edges(g, edges)
                new_edges_total += ne
            if (i + 1) % 200 == 0 or i + 1 == len(tex_pids):
                print(
                    f"  reparse {i+1}/{len(tex_pids)} new_edges={new_edges_total} "
                    f"resolved={parse_stats['resolved']} by_id={parse_stats['resolved_by_id']}",
                    flush=True,
                )
        report["steps"].append(
            {
                "name": "reparse_existing",
                "papers": len(tex_pids),
                "new_edges": new_edges_total,
                "parse_stats": dict(parse_stats),
            }
        )
    print("after reparse", graph_stats(g), flush=True)

    q_expand = [str(q).strip() for q in (expand_queries or []) if str(q).strip()]
    use_sim_gate = float(expand_sim_tau) > 0 and bool(q_expand)
    report["expand_sim_tau"] = float(expand_sim_tau)
    report["expand_queries"] = q_expand

    for rnd in range(expand_rounds):
        candidates = [
            n for n in g.nodes if g.in_degree(n) >= 1 and g.out_degree(n) == 0
        ]
        candidates.sort(key=lambda n: g.in_degree(n), reverse=True)
        n_cand_raw = len(candidates)
        sim_map: dict[str, float] = {}
        n_sim_filtered = 0
        if use_sim_gate:
            sim_map = _expand_sim_scores(
                q_expand, candidates, embedder=expand_embedder
            )
            before = len(candidates)
            candidates = [
                n for n in candidates if sim_map.get(_norm_id(n), -1.0) >= expand_sim_tau
            ]
            n_sim_filtered = before - len(candidates)
            candidates.sort(
                key=lambda n: (
                    sim_map.get(_norm_id(n), -1.0),
                    g.in_degree(n),
                ),
                reverse=True,
            )
        budget = expand_budget(
            len(candidates),
            expand_frac=expand_frac,
            max_expand_per_round=max_expand_per_round,
            min_expand_per_round=min_expand_per_round,
        )
        todo: list[str] = []
        for n in candidates:
            if len(todo) >= budget:
                break
            pdir = _paper_dir(papers_dir, n)
            z = _zip_path(zip_dir_p, n)
            if (pdir / "final_cleaned.tex").exists() or z is not None or allow_download:
                todo.append(n)
        print(
            f"expand round {rnd+1}: consider {len(todo)} / {len(candidates)} "
            f"zero-outdeg citees (raw={n_cand_raw}, sim_drop={n_sim_filtered}) "
            f"(frac={expand_frac}, budget={budget}"
            f"{f', cap={max_expand_per_round}' if max_expand_per_round else ''}"
            f"{f', sim_tau={expand_sim_tau}' if use_sim_gate else ''})",
            flush=True,
        )
        round_new = 0
        materialized = 0
        for j, pid in enumerate(todo):
            pdir = _paper_dir(papers_dir, pid)
            if not (pdir / "final_cleaned.tex").exists():
                z = _zip_path(zip_dir_p, pid)
                if z is not None:
                    if materialize_paper_from_zip(pid, z, papers_dir):
                        materialized += 1
                elif allow_download:
                    tar = (
                        papers_dir.parent
                        / "research_papers_zip_expand"
                        / f"{pid.replace('/', '')}.tar.gz"
                    )
                    print(f"  downloading {pid} ...", flush=True)
                    if download_arxiv_src(pid, tar):
                        if materialize_paper_from_zip(pid, tar, papers_dir):
                            materialized += 1
            edges, st = extract_edges_for_paper(
                pid, papers_dir, zip_dir_p, title_index, known_ids
            )
            if edges:
                ensure_nodes(
                    g, [pid] + [t for _, t, _ in edges], meta_path, papers_cache
                )
                ne, _ = add_edges(g, edges)
                round_new += ne
            if (j + 1) % 50 == 0:
                print(
                    f"  expand {j+1}/{len(todo)} new_edges={round_new} "
                    f"materialized={materialized}",
                    flush=True,
                )
        report["steps"].append(
            {
                "name": f"expand_round_{rnd+1}",
                "candidates_raw": n_cand_raw,
                "candidates": len(candidates),
                "sim_filtered_out": n_sim_filtered,
                "budget": budget,
                "expand_frac": expand_frac,
                "todo": len(todo),
                "materialized": materialized,
                "new_edges": round_new,
            }
        )
        print(f"after expand {rnd+1}", graph_stats(g), flush=True)

    iso = list(nx.isolates(g))
    g.remove_nodes_from(iso)
    new_stats = graph_stats(g)
    report["new"] = new_stats
    report["removed_isolates"] = len(iso)

    gexf_out.parent.mkdir(parents=True, exist_ok=True)
    nx.write_gexf(g, str(gexf_out))
    print(
        f"wrote {gexf_out} nodes={g.number_of_nodes()} edges={g.number_of_edges()}",
        flush=True,
    )

    goals = {
        "more_edges_than_old": new_stats["edges"] > old_stats["edges"],
        "more_or_equal_nodes": new_stats["nodes"] >= old_stats["nodes"],
        "sentence_rate_ge_0_95": new_stats["edges_with_sentence_rate"] >= 0.95,
        "elig10_sources_ge_old": new_stats["sources_outdeg_ge10"]
        >= old_stats["sources_outdeg_ge10"],
        "full_gold_ready_note": (
            "dataset must use all successors; graph provides untruncated out-edges"
        ),
    }
    report["design_goals"] = goals
    try:
        report["gexf_out"] = str(gexf_out.relative_to(ROOT))
    except ValueError:
        report["gexf_out"] = str(gexf_out)
    report["zip_dir"] = str(zip_dir_p) if zip_dir_p else ""
    report["allow_download"] = bool(allow_download)

    if report_path is not None:
        rp = Path(report_path)
        if not rp.is_absolute():
            rp = ROOT / rp
        rp.parent.mkdir(parents=True, exist_ok=True)
        rp.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"report → {rp}", flush=True)

    print(
        json.dumps(
            {"design_goals": goals, "old": old_stats, "new": new_stats}, indent=2
        )
    )
    return report


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Enrich a seed GEXF via AS reparse + optional expand (standalone)"
    )
    ap.add_argument(
        "--gexf-in",
        default="superconducting_quantum_computing_retrieval/description/test_graph.gexf",
    )
    ap.add_argument(
        "--gexf-out",
        default="superconducting_quantum_computing_retrieval/description/test_graph_rr.gexf",
    )
    ap.add_argument(
        "--papers-dir",
        default="superconducting_quantum_computing_retrieval/research_papers",
    )
    ap.add_argument(
        "--zip-dir",
        default="",
        help="readonly tar.gz dir; empty=auto domain zip or RWCITE_ASSET_SOURCE",
    )
    ap.add_argument(
        "--metadata",
        "--metadata-db",
        dest="metadata_path",
        default=DEFAULT_METADATA,
        help="arXiv OAI JSONL snapshot (title/abstract index)",
    )
    ap.add_argument("--expand-rounds", type=int, default=2)
    ap.add_argument(
        "--expand-frac",
        type=float,
        default=0.2,
        help="Fraction of zero-outdeg candidates to expand per round (default 0.2). "
        "Set 0 to use --max-expand-per-round as a fixed count.",
    )
    ap.add_argument(
        "--max-expand-per-round",
        type=int,
        default=0,
        help="Optional hard cap on expand todo (0=no cap). "
        "If --expand-frac=0, this is the fixed count (legacy).",
    )
    ap.add_argument(
        "--min-expand-per-round",
        type=int,
        default=0,
        help="Optional floor when frac×candidates is small (0=no floor).",
    )
    ap.add_argument(
        "--max-reparse",
        type=int,
        default=0,
        help="0=all with tex; -1=skip reparse; >0=subsample",
    )
    ap.add_argument("--allow-download", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--expand-sim-tau",
        type=float,
        default=0.0,
        help="If >0, expand only citees with topic-emb cosine to --expand-query ≥ tau",
    )
    ap.add_argument(
        "--expand-query",
        action="append",
        default=[],
        help="Domain query prototype for expand sim gate (repeatable)",
    )
    ap.add_argument(
        "--report",
        default="datasets/rr_pool_ranker/graph_rebuild_report.json",
    )
    args = ap.parse_args()
    enrich_citation_graph(
        gexf_in=args.gexf_in,
        gexf_out=args.gexf_out,
        papers_dir=args.papers_dir,
        zip_dir=args.zip_dir or None,
        metadata_path=args.metadata_path,
        expand_rounds=args.expand_rounds,
        expand_frac=args.expand_frac,
        max_expand_per_round=args.max_expand_per_round,
        min_expand_per_round=args.min_expand_per_round,
        max_reparse=args.max_reparse,
        allow_download=args.allow_download,
        seed=args.seed,
        report_path=args.report,
        expand_sim_tau=args.expand_sim_tau,
        expand_queries=args.expand_query or None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
