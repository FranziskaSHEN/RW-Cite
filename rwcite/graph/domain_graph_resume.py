"""Resumable native domain graph build (seed + expand).

Used by ``rwcite.cli.graph_pipeline`` for all domains.
Checkpoint: ``work_dir/checkpoint.json`` + ``work_dir/partial.gexf``.

Download policy: never re-fetch when ``final_cleaned.tex`` exists *and* bib is
present; if bib is missing, re-materialize from the best local zip root.
``allow_download`` covers seed papers that still lack a local zip, and expand
citees.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Sequence

import networkx as nx


def zip_tar_count(path: Path) -> int:
    if not path.is_dir():
        return 0
    n = 0
    for _ in path.glob("*.tar.gz"):
        n += 1
        if n >= 50_000:
            break
    return n


def pick_zip_dirs(
    *,
    papers_parent: Path,
    asset_root: Path | None,
    explicit: Path | None = None,
    extra: Sequence[Path] = (),
) -> list[Path]:
    """Order zip roots by coverage (most tarballs first)."""
    if explicit is not None and explicit.is_dir():
        return [explicit]
    cands: list[tuple[int, Path]] = []
    for name in ("research_papers_zip", "research_papers_zip_rr"):
        p = papers_parent / name
        n = zip_tar_count(p)
        if n:
            cands.append((n, p))
    if asset_root is not None:
        # Optional shared corpus: same domain folder under asset_root
        domain_dir = papers_parent.name
        alt = asset_root / domain_dir / "research_papers_zip"
        n = zip_tar_count(alt)
        if n:
            cands.append((n, alt))
    for p in extra:
        n = zip_tar_count(p)
        if n:
            cands.append((n, p))
    cands.sort(key=lambda x: -x[0])
    out: list[Path] = []
    seen: set[Path] = set()
    for _, p in cands:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            out.append(p)
    return out


def find_zip(mod: Any, zip_dirs: Sequence[Path], pid: str) -> Path | None:
    for zd in zip_dirs:
        z = mod._zip_path(zd, pid)
        if z is not None:
            return z
    return None


def _empty_ckpt(meta: dict[str, Any]) -> dict[str, Any]:
    return {
        "version": 1,
        "stage": "seed",
        "seed_done": [],
        "expand_round_done": 0,
        "expand_done": [],
        "stats_snapshots": [],
        "meta": meta,
        "updated_at": None,
    }


def load_ckpt(path: Path, meta: dict[str, Any]) -> dict[str, Any]:
    if not path.is_file():
        return _empty_ckpt(meta)
    ckpt = json.loads(path.read_text(encoding="utf-8"))
    old_meta = ckpt.get("meta") or {}
    merged = {**old_meta, **meta}
    ckpt["meta"] = merged
    ckpt.setdefault("seed_done", [])
    ckpt.setdefault("expand_done", [])
    ckpt.setdefault("expand_round_done", 0)
    ckpt.setdefault("stats_snapshots", [])
    ckpt.setdefault("stage", "seed")
    return ckpt


def save_ckpt(path: Path, ckpt: dict[str, Any]) -> None:
    ckpt["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(ckpt, indent=2), encoding="utf-8")
    tmp.replace(path)


def save_gexf(g: nx.DiGraph, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.gexf")
    nx.write_gexf(g, str(tmp))
    tmp.replace(path)


def has_local_paper(mod: Any, pid: str, papers_dir: Path, zip_dirs: Sequence[Path]) -> bool:
    pdir = mod._paper_dir(papers_dir, pid)
    if (pdir / "final_cleaned.tex").is_file():
        return True
    return find_zip(mod, zip_dirs, pid) is not None


def ensure_materialized(
    mod: Any,
    pid: str,
    papers_dir: Path,
    zip_dirs: Sequence[Path],
    *,
    allow_download: bool,
    download_root: Path,
    refresh_bib_from_zip: bool = True,
) -> tuple[bool, str, Path | None]:
    pdir = mod._paper_dir(papers_dir, pid)
    z = find_zip(mod, zip_dirs, pid)
    has_tex = (pdir / "final_cleaned.tex").is_file()
    has_bib = bool(list(pdir.glob("*.bbl")) or list(pdir.glob("*.bib")))
    if has_tex and (has_bib or not refresh_bib_from_zip or z is None):
        return True, "skip_tex", z
    if z is not None:
        ok = mod.materialize_paper_from_zip(pid, z, papers_dir)
        action = "materialize_refresh" if has_tex else "materialize"
        return ok, (action if ok else "fail"), z
    if has_tex:
        return True, "skip_tex_no_zip", None
    if not allow_download:
        return False, "missing_no_download", None
    tar = download_root / f"{pid.replace('/', '')}.tar.gz"
    if tar.is_file() and tar.stat().st_size > 0:
        ok = mod.materialize_paper_from_zip(pid, tar, papers_dir)
        return ok, ("materialize_cached_tar" if ok else "fail"), tar
    print(f"  downloading missing {pid} ...", flush=True)
    if not mod.download_arxiv_src(pid, tar):
        return False, "download_fail", None
    ok = mod.materialize_paper_from_zip(pid, tar, papers_dir)
    return ok, ("download" if ok else "fail"), tar


def run_native_resume_build(
    mod: Any,
    *,
    retrieval_ids: Sequence[str],
    papers_dir: Path,
    zip_dirs: Sequence[Path],
    metadata_path: Path,
    work_dir: Path,
    out_gexf: Path,
    queries: Sequence[str],
    expand_rounds: int = 2,
    expand_frac: float = 0.2,
    max_expand_per_round: int = 0,
    min_expand_per_round: int = 0,
    expand_sim_tau: float = 0.0,
    expand_embedder: str = "models/base/bge-large-en-v1.5",
    allow_download: bool = False,
    ckpt_every: int = 50,
    reset: bool = False,
    seed_only: bool = False,
    run_tag: str = "",
) -> dict[str, Any]:
    """Seed + expand with checkpoint. Returns final report dict."""
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = work_dir / "checkpoint.json"
    partial_gexf = work_dir / "partial.gexf"
    report_path = work_dir / "graph_build_report.json"
    download_root = work_dir / "research_papers_zip_expand"
    seed_copy = out_gexf.with_name("test_graph_seed.gexf")

    ids = [str(x).strip() for x in retrieval_ids]
    meta = {
        "run_tag": run_tag or work_dir.name,
        "n_retrieval": len(ids),
        "papers_dir": str(papers_dir),
        "zip_dirs": [str(z) for z in zip_dirs],
        "expand_rounds": expand_rounds,
        "expand_frac": expand_frac,
        "max_expand_per_round": max_expand_per_round,
        "expand_sim_tau": expand_sim_tau,
        "allow_download": allow_download,
        "out_gexf": str(out_gexf),
    }

    if reset:
        for p in (ckpt_path, partial_gexf, report_path):
            if p.is_file():
                p.unlink()
        print(f"[reset] cleared progress under {work_dir}", flush=True)

    ckpt = load_ckpt(ckpt_path, meta)
    ckpt["meta"] = meta

    print(f"work_dir={work_dir}", flush=True)
    print(f"papers_dir={papers_dir}", flush=True)
    print(f"zip_dirs={list(zip_dirs)}", flush=True)
    print(
        f"retrieval={len(ids)} stage={ckpt.get('stage')} "
        f"seed_done={len(ckpt['seed_done'])} "
        f"expand_round_done={ckpt['expand_round_done']}",
        flush=True,
    )

    if ckpt.get("stage") == "done" and out_gexf.is_file() and not reset:
        print(f"[skip] stage=done; reuse {out_gexf}", flush=True)
        if report_path.is_file():
            return json.loads(report_path.read_text(encoding="utf-8"))
        return {"stage": "done", "gexf_out": str(out_gexf), "skipped": True}

    if partial_gexf.is_file():
        g = mod.seed_graph_from_gexf(partial_gexf)
        print(f"resumed graph: {mod.graph_stats(g)}", flush=True)
    else:
        g = nx.DiGraph()
        print("new empty graph", flush=True)

    title_index, known_ids = mod.load_title_index(metadata_path)
    papers_cache: dict[str, dict] = {}
    seed_done = set(ckpt["seed_done"])

    if ckpt.get("stage") in ("seed", None):
        todo_seed = [pid for pid in ids if pid not in seed_done]
        mod.ensure_nodes(g, ids, metadata_path, papers_cache)
        print(f"[seed] todo={len(todo_seed)} already_done={len(seed_done)}", flush=True)
        new_edges = 0
        n_tex = n_skip = n_miss = 0
        for i, pid in enumerate(todo_seed):
            ok, action, z_used = ensure_materialized(
                mod,
                pid,
                papers_dir,
                zip_dirs,
                allow_download=allow_download,
                download_root=download_root,
            )
            if action.startswith("skip_tex"):
                n_skip += 1
            if not ok:
                n_miss += 1
                seed_done.add(pid)
                continue
            zip_for = z_used.parent if z_used is not None else (
                zip_dirs[0] if zip_dirs else None
            )
            edges, st = mod.extract_edges_for_paper(
                pid, papers_dir, zip_for, title_index, known_ids
            )
            if st.get("has_tex"):
                n_tex += 1
            if edges:
                mod.ensure_nodes(g, [t for _, t, _ in edges], metadata_path, papers_cache)
                ne, _ = mod.add_edges(g, edges)
                new_edges += ne
            seed_done.add(pid)
            if (i + 1) % ckpt_every == 0 or i + 1 == len(todo_seed):
                ckpt["seed_done"] = sorted(seed_done)
                ckpt["stage"] = "seed"
                snap = mod.graph_stats(g)
                ckpt["stats_snapshots"].append({"stage": "seed", "i": i + 1, "stats": snap})
                save_gexf(g, partial_gexf)
                save_ckpt(ckpt_path, ckpt)
                print(
                    f"  [seed ckpt] {i+1}/{len(todo_seed)} +edges={new_edges} "
                    f"tex={n_tex} skip_tex={n_skip} miss={n_miss} "
                    f"nodes={snap['nodes']} edges={snap['edges']}",
                    flush=True,
                )

        iso = list(nx.isolates(g))
        g.remove_nodes_from(iso)
        ckpt["seed_done"] = sorted(seed_done)
        ckpt["stage"] = "expand" if not seed_only else "done_seed_only"
        ckpt["stats_snapshots"].append(
            {
                "stage": "seed_final",
                "stats": mod.graph_stats(g),
                "removed_isolates": len(iso),
            }
        )
        save_gexf(g, partial_gexf)
        save_gexf(g, seed_copy)
        save_ckpt(ckpt_path, ckpt)
        print(f"[seed] done {mod.graph_stats(g)} removed_isolates={len(iso)}", flush=True)

    if seed_only or ckpt.get("stage") == "done_seed_only":
        save_gexf(g, out_gexf)
        report = {
            "mode": "domain_graph_resume",
            "stage": "seed_only",
            "meta": meta,
            "stats": mod.graph_stats(g),
            "gexf_out": str(out_gexf),
        }
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"[done seed-only] → {out_gexf}", flush=True)
        return report

    expand_done = set(ckpt.get("expand_done") or [])
    start_round = int(ckpt.get("expand_round_done") or 0)
    q_list = [str(q).strip() for q in queries if str(q).strip()]

    for rnd in range(start_round, expand_rounds):
        candidates = [
            n
            for n in g.nodes
            if g.in_degree(n) >= 1 and g.out_degree(n) == 0 and n not in expand_done
        ]
        candidates.sort(key=lambda n: g.in_degree(n), reverse=True)
        n_raw = len(candidates)
        n_sim_drop = 0
        if expand_sim_tau > 0 and q_list:
            sim_map = mod._expand_sim_scores(
                q_list, candidates, embedder=expand_embedder
            )
            before = len(candidates)
            candidates = [
                n
                for n in candidates
                if sim_map.get(mod._norm_id(n), -1.0) >= expand_sim_tau
            ]
            n_sim_drop = before - len(candidates)
            candidates.sort(
                key=lambda n: (sim_map.get(mod._norm_id(n), -1.0), g.in_degree(n)),
                reverse=True,
            )

        budget = mod.expand_budget(
            len(candidates),
            expand_frac=expand_frac,
            max_expand_per_round=max_expand_per_round,
            min_expand_per_round=min_expand_per_round,
        )
        todo: list[str] = []
        for n in candidates:
            if len(todo) >= budget:
                break
            if has_local_paper(mod, n, papers_dir, zip_dirs) or allow_download:
                todo.append(n)

        print(
            f"[expand {rnd+1}/{expand_rounds}] todo={len(todo)} "
            f"cand={len(candidates)} raw={n_raw} sim_drop={n_sim_drop} "
            f"budget={budget} already_expanded={len(expand_done)}",
            flush=True,
        )

        round_new = 0
        actions: dict[str, int] = {}
        for j, pid in enumerate(todo):
            ok, action, z_used = ensure_materialized(
                mod,
                pid,
                papers_dir,
                zip_dirs,
                allow_download=allow_download,
                download_root=download_root,
            )
            actions[action] = actions.get(action, 0) + 1
            if ok:
                zip_for = z_used.parent if z_used is not None else (
                    zip_dirs[0] if zip_dirs else None
                )
                edges, _st = mod.extract_edges_for_paper(
                    pid, papers_dir, zip_for, title_index, known_ids
                )
                if edges:
                    mod.ensure_nodes(
                        g, [pid] + [t for _, t, _ in edges], metadata_path, papers_cache
                    )
                    ne, _ = mod.add_edges(g, edges)
                    round_new += ne
            expand_done.add(pid)
            if (j + 1) % ckpt_every == 0 or j + 1 == len(todo):
                ckpt["expand_done"] = sorted(expand_done)
                ckpt["expand_round_done"] = rnd
                ckpt["stage"] = "expand"
                snap = mod.graph_stats(g)
                save_gexf(g, partial_gexf)
                save_ckpt(ckpt_path, ckpt)
                print(
                    f"  [expand ckpt] {j+1}/{len(todo)} +edges={round_new} "
                    f"actions={actions} nodes={snap['nodes']}",
                    flush=True,
                )

        ckpt["expand_round_done"] = rnd + 1
        ckpt["expand_done"] = sorted(expand_done)
        ckpt["stats_snapshots"].append(
            {
                "stage": f"expand_{rnd+1}",
                "todo": len(todo),
                "new_edges": round_new,
                "actions": actions,
                "stats": mod.graph_stats(g),
            }
        )
        save_gexf(g, partial_gexf)
        save_ckpt(ckpt_path, ckpt)
        print(f"[expand {rnd+1}] done {mod.graph_stats(g)}", flush=True)

    iso = list(nx.isolates(g))
    g.remove_nodes_from(iso)
    final_stats = mod.graph_stats(g)
    ckpt["stage"] = "done"
    ckpt["stats_snapshots"].append(
        {"stage": "final", "stats": final_stats, "removed_isolates": len(iso)}
    )
    save_gexf(g, partial_gexf)
    save_gexf(g, out_gexf)
    save_ckpt(ckpt_path, ckpt)

    report = {
        "mode": "domain_graph_resume",
        "meta": meta,
        "queries": list(q_list),
        "stats": final_stats,
        "removed_isolates": len(iso),
        "gexf_out": str(out_gexf),
        "partial_gexf": str(partial_gexf),
        "checkpoint": str(ckpt_path),
        "design_goals": {
            "sentence_rate_ge_0_95": final_stats["edges_with_sentence_rate"] >= 0.95,
            "has_elig10": final_stats["sources_outdeg_ge10"] > 0,
        },
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"stats": final_stats, "gexf_out": str(out_gexf)}, indent=2), flush=True)
    print(f"[done] → {out_gexf}", flush=True)
    return report
