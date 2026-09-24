#!/usr/bin/env python3
"""Offline L2 BertScore fill-in for RR v6 cite eval JSON.

Sources (first hit wins):
  1) <out>.bertscore_pairs.json sidecar from eval
  2) details[].cites[{sentence, gold_sentence}]
  3) details[].bertscore_pairs

Updates summary.mean_cite_bertscore_f1 / n_bertscore_pairs in place.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_pairs(eval_path: Path) -> list[tuple[str, str]]:
    side = eval_path.with_name(eval_path.stem + ".bertscore_pairs.json")
    if side.is_file():
        blob = json.loads(side.read_text(encoding="utf-8"))
        pairs = blob.get("pairs") or []
        return [(str(a), str(b)) for a, b in pairs if str(a).strip() and str(b).strip()]

    data = json.loads(eval_path.read_text(encoding="utf-8"))
    details = data.get("details") or []
    out: list[tuple[str, str]] = []
    for d in details:
        if d.get("bertscore_pairs"):
            for a, b in d["bertscore_pairs"]:
                if str(a).strip() and str(b).strip():
                    out.append((str(a), str(b)))
            continue
        for c in d.get("cites") or []:
            a = (c.get("sentence") or "").strip()
            b = (c.get("gold_sentence") or "").strip()
            if a and b:
                out.append((a, b))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--eval-json",
        type=Path,
        required=True,
        help="eval_cite_full370.json (or n80)",
    )
    args = ap.parse_args()
    path: Path = args.eval_json
    if not path.is_file():
        raise SystemExit(f"missing {path}")

    pairs = _load_pairs(path)
    if not pairs:
        raise SystemExit(
            f"no BertScore pairs found for {path}; re-run eval after upgrade "
            "(needs cites or *.bertscore_pairs.json)"
        )

    from rwcite.runtime.services.bertscore_service import get_bertscore_service

    rows = get_bertscore_service().score_many([a for a, _ in pairs], [b for _, b in pairs])
    mean_f1 = sum(r["f1"] for r in rows) / max(1, len(rows))

    data = json.loads(path.read_text(encoding="utf-8"))
    summary = data.setdefault("summary", {})
    summary["n_bertscore_pairs"] = len(pairs)
    summary["mean_cite_bertscore_f1"] = float(mean_f1)
    summary.pop("bertscore_error", None)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "eval_json": str(path),
                "n_bertscore_pairs": len(pairs),
                "mean_cite_bertscore_f1": mean_f1,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
