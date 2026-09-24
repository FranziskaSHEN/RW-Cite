"""CLI: verify the frozen EWM split-masked data contract.

Usage: python -m rwcite.gat.contract_check
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import networkx as nx
import yaml

from rwcite.gat import NOFUTURE_GEXF_SHA256, SPLIT_ID
from rwcite.gat import protocol as P
from rwcite.ranker.reference_recommend import normalize_arxiv_id


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_ids(path: Path) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [normalize_arxiv_id(str(x)) for x in data if str(x).strip()]
    if isinstance(data, dict):
        return [normalize_arxiv_id(str(x)) for x in data.keys()]
    raise SystemExit(f"unexpected JSON shape: {path}")


def _fail(msg: str, errors: list[str]) -> None:
    errors.append(msg)
    print(f"FAIL  {msg}", file=sys.stderr)


def _ok(msg: str) -> None:
    print(f"OK    {msg}")


def main(argv: list[str] | None = None) -> int:
    del argv  # unused
    root = P.root_path()
    errors: list[str] = []

    # domains.yaml points at nofuture
    domains_path = root / "configs" / "domains.yaml"
    block = yaml.safe_load(domains_path.read_text(encoding="utf-8"))["domains"]["ewm"]
    gexf_cfg = str(block.get("gexf") or "")
    if gexf_cfg.replace("\\", "/") != P.NOFUTURE_GEXF:
        _fail(
            f"domains.yaml ewm.gexf={gexf_cfg!r} expected {P.NOFUTURE_GEXF!r}",
            errors,
        )
    else:
        _ok(f"domains.yaml ewm.gexf → {P.NOFUTURE_GEXF}")

    full_o2 = root / P.FULL_O2_GEXF
    nofuture = root / P.NOFUTURE_GEXF
    for label, path, expected in (
        ("full O2", full_o2, P.FULL_O2_SHA256),
        ("nofuture", nofuture, NOFUTURE_GEXF_SHA256),
    ):
        if not path.is_file():
            _fail(f"missing {label} gexf: {path}", errors)
            continue
        got = _sha256(path)
        if got != expected:
            _fail(f"{label} sha256 mismatch: got {got} expected {expected}", errors)
        else:
            _ok(f"{label} sha256 {got[:24]}…")

    split_dir = root / P.SPLIT_DIR
    meta_path = split_dir / "split_meta.json"
    if not meta_path.is_file():
        _fail(f"missing {meta_path}", errors)
        return 1
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("split_id") != SPLIT_ID:
        _fail(f"split_id={meta.get('split_id')!r} expected {SPLIT_ID!r}", errors)
    else:
        _ok(f"split_id={SPLIT_ID}")

    for key, expected in (
        ("n_train_sources", P.N_TRAIN_SOURCES),
        ("n_test_sources", P.N_TEST_SOURCES),
        ("n_frontier_sources", P.N_FRONTIER_SOURCES),
    ):
        got = int(meta.get(key) or -1)
        if got != expected:
            _fail(f"{key}={got} expected {expected}", errors)
        else:
            _ok(f"{key}={got}")

    admit_gexf = str(meta.get("gexf") or "").replace("\\", "/")
    if not admit_gexf.endswith("test_graph_rr.o2.gexf"):
        _fail(f"split admit gexf must be full O2, got {admit_gexf!r}", errors)
    else:
        _ok(f"admit gexf fingerprint path={admit_gexf}")
    if meta.get("gexf_sha256") != P.FULL_O2_SHA256:
        _fail(
            f"split gexf_sha256={meta.get('gexf_sha256')} expected full O2 {P.FULL_O2_SHA256}",
            errors,
        )
    else:
        _ok("split meta gexf_sha256 = full O2")

    train_src = _load_ids(split_dir / "train_sources.json")
    test_src = _load_ids(split_dir / "test_sources.json")
    frontier = _load_ids(split_dir / "frontier_sources.json")
    if len(train_src) != P.N_TRAIN_SOURCES:
        _fail(f"train_sources.json len={len(train_src)}", errors)
    if len(test_src) != P.N_TEST_SOURCES:
        _fail(f"test_sources.json len={len(test_src)}", errors)
    if len(frontier) != P.N_FRONTIER_SOURCES:
        _fail(f"frontier_sources.json len={len(frontier)}", errors)
    else:
        _ok("source list lengths match meta")

    for name, rel in (("train", P.TRAIN_JSONL), ("test", P.TEST_JSONL)):
        path = root / rel
        if not path.is_file():
            _fail(f"missing {name} jsonl: {path}", errors)
            continue
        n_lines = 0
        sources: set[str] = set()
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                n_lines += 1
                row = json.loads(line)
                sid = row.get("source_id") or row.get("query_id") or row.get("id")
                if sid:
                    sources.add(normalize_arxiv_id(str(sid)))
        expect = P.N_TRAIN_SOURCES if name == "train" else P.N_TEST_SOURCES
        # train.jsonl may hold multiple rows per source (pool variants); require unique sources.
        if len(sources) != expect:
            _fail(
                f"{name}.jsonl unique_sources={len(sources)} lines={n_lines} expected {expect}",
                errors,
            )
        else:
            _ok(f"{name}.jsonl unique_sources={len(sources)} (lines={n_lines})")

    # test∪frontier outdeg must be 0 on nofuture graph
    if nofuture.is_file():
        g = nx.read_gexf(str(nofuture), node_type=None, relabel=False, version="1.2draft")
        future = set(test_src) | set(frontier)
        bad: list[tuple[str, int]] = []
        for pid in future:
            if pid not in g:
                # id may be stored with different keying; try raw
                continue
            od = int(g.out_degree(pid))
            if od != 0:
                bad.append((pid, od))
        # also resolve nodes by arxiv-like attrs if needed
        if not bad:
            # count how many future ids are in graph
            in_g = sum(1 for pid in future if pid in g)
            _ok(
                f"nofuture: sampled future sources in graph={in_g}/{len(future)}; "
                f"outdeg>0 count={len(bad)}"
            )
        if bad:
            sample = ", ".join(f"{a}:{b}" for a, b in bad[:5])
            _fail(f"nofuture future outdeg>0 for {len(bad)} sources (e.g. {sample})", errors)
        elif not any(pid in g for pid in future):
            # try matching via node id attribute
            id_to_node = {}
            for n, attrs in g.nodes(data=True):
                for key in ("id", "label", "arxiv_id"):
                    v = attrs.get(key)
                    if v:
                        id_to_node[normalize_arxiv_id(str(v))] = n
                id_to_node[normalize_arxiv_id(str(n))] = n
            bad2: list[tuple[str, int]] = []
            for pid in future:
                node = id_to_node.get(pid)
                if node is None:
                    continue
                od = int(g.out_degree(node))
                if od != 0:
                    bad2.append((pid, od))
            if bad2:
                sample = ", ".join(f"{a}:{b}" for a, b in bad2[:5])
                _fail(
                    f"nofuture future outdeg>0 for {len(bad2)} sources (e.g. {sample})",
                    errors,
                )
            else:
                _ok("nofuture: test∪frontier outdeg all 0 (via id map)")

    if errors:
        print(f"\ncontract FAILED ({len(errors)} issue(s))", file=sys.stderr)
        return 1
    print("\ncontract OK — ewm 3.0 spine ready (admit full O2; train nofuture; frozen jsonl)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
