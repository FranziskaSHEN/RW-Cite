"""Paired Hits@10 compare: GAT (or any) candidate vs Release 2.0 nofuture baseline.

Usage:
  python -m rwcite.gat.compare
  python -m rwcite.gat.compare --baseline PATH --candidate PATH
  python -m rwcite.gat.compare --candidate PATH --bar 2.735
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from rwcite.gat import BASELINE_FULL381_HITS_AT_10
from rwcite.gat import protocol as P


def _load_summary(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "summary" in data and isinstance(data["summary"], dict):
        return data["summary"]
    if isinstance(data, dict):
        return data
    raise ValueError(f"unrecognized eval JSON: {path}")


def extract_hits_at_10(summary: dict[str, Any]) -> float | None:
    for key in (
        "mean_hits_at_10",
        "hits_at_10",
        "mean_hits@10",
        "hits@10",
    ):
        if key in summary and summary[key] is not None:
            return float(summary[key])
    return None


def extract_metric(summary: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        if key in summary and summary[key] is not None:
            return float(summary[key])
    return None


def compare(
    baseline_path: Path | None,
    candidate_path: Path | None,
    bar: float = BASELINE_FULL381_HITS_AT_10,
) -> dict[str, Any]:
    """Return compare dict.

    promote_ok: candidate @10 > charter **bar** (default 2.735), not vs baseline file.
    beats_baseline_file: candidate @10 > baseline file @10 when both present.
    """
    out: dict[str, Any] = {
        "bar_hits_at_10": bar,
        "baseline_path": str(baseline_path) if baseline_path else None,
        "candidate_path": str(candidate_path) if candidate_path else None,
        "baseline_hits_at_10": None,
        "candidate_hits_at_10": None,
        "delta_hits_at_10": None,
        "delta_vs_baseline_file": None,
        "beats_baseline_file": None,
        "promote_ok": False,
        "decision": "incomplete",
        "note": (
            "promote_ok = candidate full381 hits@10 > charter bar "
            f"(default {BASELINE_FULL381_HITS_AT_10}), under nofuture + test_admit_v1. "
            "beats_baseline_file = candidate > baseline file @10 (e.g. vs E4). "
            "Do not cut over production CE until product review."
        ),
    }

    if baseline_path and baseline_path.is_file():
        bsum = _load_summary(baseline_path)
        out["baseline_hits_at_10"] = extract_hits_at_10(bsum)
        out["baseline_universe_recall"] = extract_metric(
            bsum, "mean_universe_recall", "universe_recall"
        )
    elif baseline_path:
        out["baseline_missing"] = True

    if candidate_path and candidate_path.is_file():
        csum = _load_summary(candidate_path)
        out["candidate_hits_at_10"] = extract_hits_at_10(csum)
        out["candidate_universe_recall"] = extract_metric(
            csum, "mean_universe_recall", "universe_recall"
        )
    elif candidate_path:
        out["candidate_missing"] = True

    cand = out["candidate_hits_at_10"]
    base = out["baseline_hits_at_10"]
    if cand is not None and base is not None:
        out["delta_vs_baseline_file"] = float(cand) - float(base)
        out["beats_baseline_file"] = bool(float(cand) > float(base))

    if cand is None:
        out["decision"] = "waiting_for_candidate"
        return out

    out["delta_hits_at_10"] = float(cand) - float(bar)
    if float(cand) > float(bar):
        out["promote_ok"] = True
        out["decision"] = "promote_gat_primary_design"
    else:
        out["promote_ok"] = False
        out["decision"] = "keep_rr_c2s_production_plan_b_coverage"

    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Paired GAT vs 2.0 nofuture full381 compare")
    ap.add_argument(
        "--baseline",
        type=Path,
        default=None,
        help=f"default: {{RWCITE_ROOT}}/{P.BASELINE_FULL381_MERGED}",
    )
    ap.add_argument(
        "--candidate",
        type=Path,
        default=None,
        help=f"default: {{RWCITE_ROOT}}/{P.GAT_MVP_MERGED}",
    )
    ap.add_argument(
        "--bar",
        type=float,
        default=BASELINE_FULL381_HITS_AT_10,
        help=f"promote if candidate hits@10 > bar (default {BASELINE_FULL381_HITS_AT_10})",
    )
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args(argv)

    root = P.root_path()
    baseline = args.baseline or (root / P.BASELINE_FULL381_MERGED)
    candidate = args.candidate or (root / P.GAT_MVP_MERGED)

    result = compare(baseline, candidate, bar=args.bar)
    text = json.dumps(result, indent=2, ensure_ascii=False)
    print(text)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n", encoding="utf-8")

    if result["decision"] == "waiting_for_candidate":
        print(
            "\nNo candidate eval yet — train/eval GAT MVP then re-run compare.",
            file=sys.stderr,
        )
        return 0
    if result.get("candidate_missing"):
        return 0
    return 0 if result["promote_ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
