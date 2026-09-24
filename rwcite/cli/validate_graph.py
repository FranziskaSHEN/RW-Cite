#!/usr/bin/env python3
"""Validate domain graph build outputs (retrieval / download / GEXF)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import networkx as nx

ROOT = Path(__file__).resolve().parents[2]


def check(report: dict, name: str, cond: bool, detail: str = "") -> None:
    report["checks"].append({"name": name, "pass": bool(cond), "detail": detail})
    if not cond:
        report["ok"] = False
        report["failures"].append(f"{name}: {detail}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--domain-dir",
        default="superconducting_quantum_computing_retrieval",
    )
    ap.add_argument(
        "--retrieval-nodes",
        default="superconducting_quantum_computing_retrieval/retrieval_nodes.json",
    )
    ap.add_argument("--log", default="logs/domain_graph_pipeline.log")
    ap.add_argument(
        "--enrich-report",
        default="datasets/superconducting_quantum_computing_graph_enrich_report.json",
    )
    ap.add_argument("--out", default="logs/domain_graph_validation.json")
    ap.add_argument("--expect-retrievals", type=int, default=5000)
    args = ap.parse_args()

    dom = ROOT / args.domain_dir
    log_path = ROOT / args.log
    log = log_path.read_text(encoding="utf-8", errors="replace") if log_path.is_file() else ""
    report: dict = {"ok": True, "checks": [], "stats": {}, "failures": []}

    check(report, "pipeline_complete_log", "Pipeline complete" in log, "need success footer")
    check(
        report,
        "no_traceback",
        "Traceback (most recent call last)" not in log,
        "traceback in log" if "Traceback" in log else "ok",
    )

    nodes_path = ROOT / args.retrieval_nodes
    check(report, "retrieval_nodes_exist", nodes_path.is_file())
    ids: list[str] = []
    n_ret = 0
    if nodes_path.is_file():
        ids = list(json.load(open(nodes_path, encoding="utf-8")).keys())
        n_ret = len(ids)
        lo = int(args.expect_retrievals * 0.9)
        check(
            report,
            "retrieval_count_near_target",
            lo <= n_ret <= args.expect_retrievals,
            f"n={n_ret} expect~{args.expect_retrievals}",
        )
        report["stats"]["retrieval_n"] = n_ret

    papers = dom / "research_papers"
    zips = dom / "research_papers_zip"
    tex_n = len(list(papers.glob("*/final_cleaned.tex"))) if papers.is_dir() else 0
    zip_n = len(list(zips.glob("*.tar.gz"))) if zips.is_dir() else 0
    dir_n = sum(1 for p in papers.iterdir() if p.is_dir()) if papers.is_dir() else 0
    report["stats"].update(
        {"zip_n": zip_n, "paper_dirs": dir_n, "final_cleaned": tex_n}
    )
    check(
        report,
        "download_nontrivial",
        zip_n >= 100 or tex_n >= 100,
        f"zips={zip_n} tex={tex_n}",
    )
    if n_ret:
        check(
            report,
            "clean_rate_ge_0_5",
            tex_n / n_ret >= 0.5,
            f"tex/retr={tex_n}/{n_ret}",
        )

    seed = dom / "description" / "test_graph_seed.gexf"
    gexf = dom / "description" / "test_graph.gexf"
    check(report, "seed_or_final_gexf", seed.is_file() or gexf.is_file())
    check(report, "final_gexf_exists", gexf.is_file())

    er_path = ROOT / args.enrich_report
    er = None
    if er_path.is_file():
        er = json.loads(er_path.read_text(encoding="utf-8"))
    report["stats"]["enrich_report"] = str(er_path) if er else None

    if gexf.is_file():
        g = nx.read_gexf(gexf, node_type=None, relabel=False, version="1.2draft")
        n, e = g.number_of_nodes(), g.number_of_edges()
        sent = sum(
            1
            for _, _, a in g.edges(data=True)
            if str(a.get("sentence") or "").strip()
        )
        elig = sum(1 for x in g.nodes if g.out_degree(x) >= 10)
        out1 = sum(1 for x in g.nodes if g.out_degree(x) >= 1)
        report["stats"]["gexf"] = {
            "nodes": n,
            "edges": e,
            "sentence_rate": sent / max(e, 1),
            "sources_outdeg_ge1": out1,
            "sources_outdeg_ge10": elig,
        }
        check(
            report,
            "gexf_nodes_cover_retrieval",
            n >= n_ret * 0.8 if n_ret else n > 0,
            f"nodes={n} retr={n_ret}",
        )
        check(report, "gexf_has_edges", e > 0, f"edges={e}")
        check(
            report,
            "sentence_rate_ge_0_9",
            (sent / max(e, 1)) >= 0.9,
            f"rate={sent / max(e, 1):.3f}",
        )
        check(report, "elig10_sources_ge_50", elig >= 50, f"elig10={elig}")

        keys = (
            "qubit",
            "josephson",
            "quantum",
            "superconduct",
            "transmon",
            "fluxonium",
            "error correction",
        )
        sample_ids = ids[: min(200, len(ids))]
        hit = 0
        for pid in sample_ids:
            if pid not in g:
                continue
            t = (g.nodes[pid].get("title") or "").lower()
            if any(k in t for k in keys):
                hit += 1
        if sample_ids:
            check(
                report,
                "topic_title_hit_rate_ge_0_2",
                hit / len(sample_ids) >= 0.2,
                f"{hit}/{len(sample_ids)}",
            )

        if seed.is_file():
            gs = nx.read_gexf(seed, node_type=None, relabel=False, version="1.2draft")
            report["stats"]["seed"] = {
                "nodes": gs.number_of_nodes(),
                "edges": gs.number_of_edges(),
            }
            check(
                report,
                "enrich_edges_ge_seed",
                e >= gs.number_of_edges(),
                f"final_e={e} seed_e={gs.number_of_edges()}",
            )

    if er:
        report["stats"]["enrich"] = {
            "old": er.get("old"),
            "new": er.get("new"),
            "steps": er.get("steps"),
            "design_goals": er.get("design_goals"),
        }

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print("VALIDATION_OK" if report["ok"] else "VALIDATION_FAIL")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
