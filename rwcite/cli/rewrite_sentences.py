#!/usr/bin/env python3
"""Offline rewrite of GEXF edge ``sentence`` via cite_sentence_clean.

Does not change topology. Writes a new GEXF + JSON report.
Default out: sibling ``*.sentences_clean.gexf`` (production path untouched).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import networkx as nx

ROOT = Path(__file__).resolve().parents[2]

from rwcite.latex.cite_sentence_clean import clean_cite_sentence, garbage_flags  # noqa: E402


def _stats(g: nx.DiGraph, key: str = "sentence") -> dict:
    c = Counter()
    n = 0
    for _u, _v, d in g.edges(data=True):
        s = str((d or {}).get(key) or "").strip()
        if not s:
            c["empty"] += 1
            continue
        n += 1
        for f in garbage_flags(s):
            c[f] += 1
    out = {"n_nonempty": n, "n_empty": int(c.get("empty", 0))}
    for k, v in c.items():
        if k == "empty":
            continue
        out[k] = v
        out[f"rate_{k}"] = round(v / max(1, n), 4)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--gexf",
        default="",
    )
    ap.add_argument(
        "--out",
        default="",
        help="Default: <stem>.sentences_clean.gexf next to input",
    )
    ap.add_argument(
        "--report",
        default="",
        help="Default: <stem>.sentences_clean_report.json",
    )
    ap.add_argument("--max-len", type=int, default=600)
    ap.add_argument("--max-cite-keys", type=int, default=6)
    ap.add_argument(
        "--backup",
        action="store_true",
        help="Copy input to <stem>.pre_sentence_clean.gexf.bak before overwrite if --out==input",
    )
    ap.add_argument(
        "--also-install",
        action="store_true",
        help="After write, copy cleaned GEXF over --gexf (with .bak). Use carefully.",
    )
    args = ap.parse_args()
    if not str(args.gexf).strip():
        raise SystemExit("--gexf required")

    gexf = Path(args.gexf)
    if not gexf.is_absolute():
        gexf = ROOT / gexf
    if not gexf.is_file():
        raise SystemExit(f"missing {gexf}")

    out = Path(args.out) if args.out else gexf.with_name(gexf.stem + ".sentences_clean.gexf")
    if not out.is_absolute():
        out = ROOT / out
    report_path = (
        Path(args.report)
        if args.report
        else out.with_name(out.stem.replace(".sentences_clean", "") + ".sentences_clean_report.json")
        if "sentences_clean" in out.name
        else out.with_suffix(".sentences_clean_report.json")
    )
    if not report_path.is_absolute():
        report_path = ROOT / report_path
    # Prefer sibling report name
    if not args.report:
        report_path = gexf.with_name(gexf.stem + ".sentences_clean_report.json")

    print(f"Loading {gexf} ...", flush=True)
    g = nx.read_gexf(str(gexf))
    before = _stats(g)
    print("before:", json.dumps(before, indent=2), flush=True)

    reason_c = Counter()
    n_ok = n_weak = n_empty = n_changed = 0
    examples: list[dict] = []

    for u, v, d in g.edges(data=True):
        raw = str((d or {}).get("sentence") or (d or {}).get("label") or "")
        raw_s = raw.strip()
        if not raw_s:
            n_empty += 1
            continue
        res = clean_cite_sentence(
            raw_s, max_len=args.max_len, max_cite_keys=args.max_cite_keys
        )
        for r in res.reasons:
            reason_c[r] += 1
        new = (res.text or "").strip()
        if res.ok:
            n_ok += 1
            g.edges[u, v]["sentence"] = new
            g.edges[u, v]["sentence_quality"] = "ok"
        elif new:
            n_weak += 1
            g.edges[u, v]["sentence"] = new
            g.edges[u, v]["sentence_quality"] = "weak"
        else:
            # Last resort: still try a permissive clean without hard reject empty
            n_weak += 1
            g.edges[u, v]["sentence_quality"] = "bad"
            reason_c["kept_raw_bad"] += 1
            # Prefer any cleaned fragment over raw LaTeX debris when available
            soft = clean_cite_sentence(raw_s, max_len=args.max_len, max_cite_keys=args.max_cite_keys)
            # already failed; leave raw
        if new and new != raw_s:
            n_changed += 1
            if len(examples) < 12 and (
                "label" in garbage_flags(raw_s)
                or "lead_junk" in garbage_flags(raw_s)
                or raw_s.startswith("%")
            ):
                examples.append(
                    {
                        "src": str(u),
                        "tgt": str(v),
                        "before": raw_s[:220],
                        "after": new[:220],
                        "ok": res.ok,
                        "reasons": res.reasons,
                    }
                )

    after = _stats(g)
    print("after:", json.dumps(after, indent=2), flush=True)

    if args.backup and out.resolve() == gexf.resolve():
        bak = gexf.with_name(gexf.stem + ".pre_sentence_clean.gexf.bak")
        print(f"backup → {bak}", flush=True)
        shutil.copy2(gexf, bak)

    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing {out} ...", flush=True)
    nx.write_gexf(g, str(out))

    report = {
        "kind": "gexf_sentence_clean",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input": str(gexf),
        "output": str(out),
        "n_edges": g.number_of_edges(),
        "n_ok": n_ok,
        "n_weak_or_bad": n_weak,
        "n_empty": n_empty,
        "n_changed": n_changed,
        "reason_counts": dict(reason_c.most_common()),
        "before": before,
        "after": after,
        "examples": examples,
        "params": {"max_len": args.max_len, "max_cite_keys": args.max_cite_keys},
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote report {report_path}", flush=True)

    if args.also_install:
        bak = gexf.with_name(gexf.stem + ".pre_sentence_clean.gexf.bak")
        if not bak.exists():
            shutil.copy2(gexf, bak)
            print(f"backup → {bak}", flush=True)
        shutil.copy2(out, gexf)
        print(f"installed cleaned → {gexf}", flush=True)

    print(
        json.dumps(
            {
                "n_ok": n_ok,
                "n_weak_or_bad": n_weak,
                "n_changed": n_changed,
                "rate_label_before": before.get("rate_label"),
                "rate_label_after": after.get("rate_label"),
                "rate_lead_junk_before": before.get("rate_lead_junk"),
                "rate_lead_junk_after": after.get("rate_lead_junk"),
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
