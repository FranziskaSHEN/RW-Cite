#!/usr/bin/env python3
"""Export G_train / G_nofuture: strip test∪frontier out-edges from a domain GEXF."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--domain", default="ewm")
    ap.add_argument("--gexf-in", default="")
    ap.add_argument("--gexf-out", default="")
    ap.add_argument("--split-dir", default="")
    args = ap.parse_args()

    sys.path.insert(0, str(ROOT))
    import yaml
    from rwcite.ranker.rr_graph_mask import export_nofuture_gexf

    block = yaml.safe_load((ROOT / "configs/domains.yaml").read_text())["domains"][
        args.domain
    ]
    gexf_in = Path(args.gexf_in) if args.gexf_in else ROOT / block["gexf"]
    if not gexf_in.is_absolute():
        gexf_in = ROOT / gexf_in
    split_dir = (
        Path(args.split_dir)
        if args.split_dir
        else ROOT / Path(block["gexf"]).parts[0] / "data" / "splits"
    )
    if not split_dir.is_absolute():
        split_dir = ROOT / split_dir
    if args.gexf_out:
        gexf_out = Path(args.gexf_out)
        if not gexf_out.is_absolute():
            gexf_out = ROOT / gexf_out
    else:
        # test_graph_rr.o2.gexf → test_graph_rr.o2.nofuture.gexf
        gexf_out = gexf_in.with_name(gexf_in.stem + ".nofuture.gexf")

    meta = export_nofuture_gexf(in_gexf=gexf_in, out_gexf=gexf_out, split_dir=split_dir)
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
