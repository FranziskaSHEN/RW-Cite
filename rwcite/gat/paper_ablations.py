"""Paper ablations for score fusion, graph channels, and window controls.

The harness writes only experiment outputs and never overwrites the canonical
score-and-rank fusion recipe.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rwcite.gat.domain_layout import effective_nofuture_gexf, resolve_gat_domain
from rwcite.gat.evaluate import evaluate
from rwcite.gat.fuse_ce import (
    _fuse_scores,
    _gat_scores_from_eval,
    _hits_at_k,
    _zscore,
    assert_ce_cache_matches,
    cache_ce_scores,
)


PHASE1_CELLS = (
    "C0",
    "C1",
    "C2",
    "C3",
    "C4a",
    "C4b",
    "C5a",
    "C5b",
    "C6a",
    "C6b",
    "C7",
)
GATE_CELLS = ("C0", "C1", "C2", "C3", "C5a")
PHASE2_CELLS = ("C9w200", "C9w700", "C10", "C11e3", "C11e5")


def _cell_fuse_spec(cell: str) -> dict[str, Any]:
    """Return fuse knobs for a phase1 (or post-phase2 L0_rrf) cell."""
    c = cell.strip()
    if c == "C0":
        return {"kind": "fuse", "fuse_mode": "l0_rrf", "alpha": 0.4, "rrf_k": 20, "rrf_w": 0.4}
    if c == "C1":
        return {"kind": "fuse", "fuse_mode": "l0", "alpha": 0.4, "rrf_k": 20, "rrf_w": 0.4}
    if c == "C2":
        return {"kind": "fuse", "fuse_mode": "l0", "alpha": 0.0, "rrf_k": 20, "rrf_w": 0.4}
    if c == "C3":
        return {"kind": "fuse", "fuse_mode": "l0", "alpha": 1.0, "rrf_k": 20, "rrf_w": 0.4}
    if c == "C4a":
        return {"kind": "fuse", "fuse_mode": "l0_rrf", "alpha": 0.3, "rrf_k": 20, "rrf_w": 0.4}
    if c == "C4b":
        return {"kind": "fuse", "fuse_mode": "l0_rrf", "alpha": 0.5, "rrf_k": 20, "rrf_w": 0.4}
    if c == "C5a":
        return {"kind": "fuse", "fuse_mode": "l0_rrf", "alpha": 0.4, "rrf_k": 20, "rrf_w": 0.2}
    if c == "C5b":
        return {"kind": "fuse", "fuse_mode": "l0_rrf", "alpha": 0.4, "rrf_k": 20, "rrf_w": 0.6}
    if c == "C6a":
        return {"kind": "fuse", "fuse_mode": "l0_rrf", "alpha": 0.4, "rrf_k": 10, "rrf_w": 0.4}
    if c == "C6b":
        return {"kind": "fuse", "fuse_mode": "l0_rrf", "alpha": 0.4, "rrf_k": 40, "rrf_w": 0.4}
    if c == "C7":
        return {"kind": "oracle_max"}
    # phase2 final fuse is always default L0_rrf on that cell's assets
    if c in PHASE2_CELLS:
        return {"kind": "fuse", "fuse_mode": "l0_rrf", "alpha": 0.4, "rrf_k": 20, "rrf_w": 0.4}
    raise KeyError(f"unknown ablation cell {cell!r}")


def _n_test(dpaths) -> int:
    meta = dpaths.split_dir / "split_meta.json"
    if meta.is_file():
        return int(json.loads(meta.read_text(encoding="utf-8")).get("n_test_sources") or 0)
    return 0


def _sample_indices(n: int, max_samples: int, seed: int) -> np.ndarray:
    if max_samples <= 0 or max_samples >= n:
        return np.arange(n, dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    return np.sort(rng.choice(n, size=int(max_samples), replace=False).astype(np.int64))


def _oracle_max_scores(sg: np.ndarray, sc: np.ndarray) -> np.ndarray:
    bad = ~np.isfinite(sc) | (sc <= -1e8)
    sc_clean = np.where(bad, np.nan, sc.astype(np.float64))
    zg = _zscore(sg.astype(np.float64))
    zc = _zscore(sc_clean)
    zc = np.where(bad | ~np.isfinite(zc), zg, zc)
    return np.maximum(zg, zc).astype(np.float32)


def ensure_e4_scores(
    *,
    root: Path,
    domain: str,
    out_dir: Path,
    windows_path: Path,
    ckpt_path: Path,
    eval_dir: Path,
    n_test: int,
    ablation: str = "E4",
    force: bool = False,
    tag_suffix: str = "",
) -> Path:
    suf = f"_{tag_suffix}" if tag_suffix else ""
    tag = f"{domain}_gat_mvp_test{n_test}_{ablation.lower()}{suf}_scores"
    out = eval_dir / f"{tag}_merged.json"
    if out.is_file() and not force:
        # sanity: enough queries
        j = json.loads(out.read_text(encoding="utf-8"))
        n = len(j.get("per_query") or [])
        if n >= max(1, int(0.9 * n_test)):
            print(f"reuse e4 scores {out} n={n}", flush=True)
            return out
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"dump-scores ablation={ablation} ckpt={ckpt_path} win={windows_path} -> {out.name}", flush=True)
    result = evaluate(
        out_dir=out_dir,
        windows_path=windows_path,
        ckpt_path=ckpt_path,
        ablation=ablation,
        device=device,
        max_queries=0,
        dump_scores=True,
    )
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"wrote {out} n={len(result.get('per_query') or [])}", flush=True)
    return out


def ensure_ce_cache(
    *,
    root: Path,
    windows_path: Path,
    ce_npz: Path,
    ce_path: Path,
    gexf: Path,
    force: bool = False,
) -> Path:
    if force or not ce_npz.is_file():
        cache_ce_scores(
            root=root,
            windows_path=windows_path,
            out_npz=ce_npz,
            ce_path=ce_path,
            gexf=gexf,
            mode="ce_sent",
            short=True,
            max_sents=3,
        )
    else:
        assert_ce_cache_matches(ce_npz, ce_path=ce_path)
        print(f"reuse ce cache {ce_npz}", flush=True)
    return ce_npz


def run_cell_fuse(
    *,
    root: Path,
    domain: str,
    cell: str,
    max_samples: int,
    seed: int,
    windows_path: Path | None = None,
    ce_npz: Path | None = None,
    gat_eval: Path | None = None,
    asset_dir: Path | None = None,
    force_e4: bool = False,
    force_ce: bool = False,
) -> dict[str, Any]:
    dpaths = resolve_gat_domain(domain, root)
    n_test = _n_test(dpaths)
    if n_test <= 0:
        raise SystemExit(f"missing split_meta n_test for {domain}")
    mvp = root / dpaths.gat_mvp_dir
    eval_dir = root / dpaths.eval_dir
    eval_dir.mkdir(parents=True, exist_ok=True)
    out_dir = Path(asset_dir) if asset_dir else mvp
    if not out_dir.is_absolute():
        out_dir = root / out_dir

    win = Path(windows_path) if windows_path else out_dir / f"windows_test{n_test}.npz"
    if not win.is_file():
        win = mvp / f"windows_test{n_test}.npz"
    if not win.is_file():
        raise SystemExit(f"missing windows {win}")

    ce_path = dpaths.ce_stage1
    gexf = effective_nofuture_gexf(dpaths, root)
    ce = Path(ce_npz) if ce_npz else out_dir / f"ce_scores_test{n_test}_sent_s1.npz"
    if not ce.is_file() and out_dir != mvp:
        # allow CE rebuild into side dir
        pass
    elif not ce.is_file():
        ce = mvp / f"ce_scores_test{n_test}_sent_s1.npz"
    ensure_ce_cache(
        root=root,
        windows_path=win,
        ce_npz=ce if ce.parent == out_dir or force_ce or not ce.is_file() else ce,
        ce_path=ce_path,
        gexf=gexf,
        force=force_ce or (not ce.is_file()),
    )
    if not ce.is_file():
        # written to out_dir by ensure
        ce = out_dir / f"ce_scores_test{n_test}_sent_s1.npz"

    ckpt = out_dir / "ckpt_e4.pt"
    if not ckpt.is_file():
        ckpt = mvp / "ckpt_e4.pt"
    # C11 may use e3/e5 ckpt names via env ABL31_CKPT
    import os

    ckpt_override = os.environ.get("ABL31_CKPT", "").strip()
    abl = os.environ.get("ABL31_ABLATION", "E4").strip() or "E4"
    if ckpt_override:
        ckpt = Path(ckpt_override)
        if not ckpt.is_absolute():
            ckpt = root / ckpt
    if gat_eval is None:
        # Side-dir / non-default windows must not clobber production E4 score JSON.
        tag_suffix = ""
        if asset_dir is not None:
            tag_suffix = Path(asset_dir).name.replace("gat_abl31_", "abl31_")
        elif windows_path is not None:
            tag_suffix = f"win{windows_path.stem}"
        gat_eval = ensure_e4_scores(
            root=root,
            domain=domain,
            out_dir=out_dir if (out_dir / "id_map.json").is_file() else mvp,
            windows_path=win,
            ckpt_path=ckpt,
            eval_dir=eval_dir,
            n_test=n_test,
            ablation=abl,
            force=force_e4,
            tag_suffix=tag_suffix,
        )
    else:
        gat_eval = Path(gat_eval)

    spec = _cell_fuse_spec(cell)
    windows = np.load(win, allow_pickle=True)
    ce_blob = np.load(ce, allow_pickle=True)
    ce_scores = ce_blob["ce_scores"]
    ce_by = {str(q): i for i, q in enumerate(ce_blob["query_ids"].tolist())}
    gat_by = _gat_scores_from_eval(gat_eval)
    ids_path = win.parent / "id_map.json"
    if not ids_path.is_file():
        ids_path = mvp / "id_map.json"
    ids = json.loads(ids_path.read_text(encoding="utf-8"))["ids"]
    if not gat_by:
        raise SystemExit(f"no scores in {gat_eval}")

    n = int(windows["query_ids"].shape[0])
    idxs = _sample_indices(n, max_samples, seed)
    sample_tag = "full" if idxs.size >= n else f"n{idxs.size}"
    tag = f"abl31_{domain}_{cell}_{sample_tag}"

    hits10 = hits30 = 0.0
    per: list[dict[str, Any]] = []
    for i in idxs.tolist():
        qid = str(windows["query_ids"][i])
        nc = int(windows["n_cands"][i])
        gold = windows["gold_mask"][i, :nc].astype(np.float32)
        gat_row = gat_by.get(qid)
        ci = ce_by.get(qid)
        if gat_row is None or ci is None:
            per.append({"source_id": qid, "hits_at_10": 0, "skip": True})
            continue
        sg = np.asarray(gat_row["scores"][:nc], dtype=np.float32)
        sc = np.asarray(ce_scores[ci, :nc], dtype=np.float32)
        m = min(sg.size, sc.size, nc)
        sg, sc, gold = sg[:m], sc[:m], gold[:m]
        if spec["kind"] == "oracle_max":
            blend = _oracle_max_scores(sg, sc)
        else:
            blend = _fuse_scores(
                sg=sg,
                sc=sc,
                alpha=float(spec["alpha"]),
                fuse_mode=str(spec["fuse_mode"]),
                rrf_k=int(spec["rrf_k"]),
                rrf_w=float(spec["rrf_w"]),
            )
        h10 = _hits_at_k(blend, gold, 10)
        h30 = _hits_at_k(blend, gold, 30)
        hits10 += h10
        hits30 += h30
        order = blend.argsort()[::-1][:30]
        cand_globals = windows["cand_idx"][i, :m]
        top_ids = [ids[int(cand_globals[j])] for j in order if int(cand_globals[j]) >= 0]
        per.append(
            {
                "source_id": qid,
                "hits_at_10": h10,
                "hits_at_30": h30,
                "n_gold_in_window": int(gold.sum()),
                "top_ids": top_ids,
            }
        )

    nn = max(int(idxs.size), 1)
    summary = {
        "n": int(idxs.size),
        "n_windows": n,
        "seed": int(seed),
        "mean_hits_at_10": float(hits10 / nn),
        "mean_hits_at_30": float(hits30 / nn),
        "cell": cell,
        "domain": domain,
        "protocol": "ablation_3_1",
        "spec": spec,
        "windows": str(win),
        "ce_cache": str(ce),
        "gat_eval": str(gat_eval),
        "ckpt": str(ckpt),
    }
    out_path = eval_dir / f"{tag}_merged.json"
    payload = {"summary": summary, "per_query": per}
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        f"abl31 cell={cell} domain={domain} tag={tag} "
        f"@10={summary['mean_hits_at_10']:.4f} @30={summary['mean_hits_at_30']:.4f} "
        f"-> {out_path}",
        flush=True,
    )
    return {"summary": summary, "path": str(out_path)}


def main() -> None:
    ap = argparse.ArgumentParser(description="Ablation 3.1 fuse / oracle cell")
    ap.add_argument("--domain", default="ewm", choices=("ewm", "sqc", "gw"))
    ap.add_argument("--cell", required=True)
    ap.add_argument("--max-samples", type=int, default=80)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--windows", default=None)
    ap.add_argument("--ce-cache", default=None)
    ap.add_argument("--gat-eval", default=None)
    ap.add_argument("--asset-dir", default=None, help="side gat_abl31_* dir")
    ap.add_argument("--force-e4", action="store_true")
    ap.add_argument("--force-ce", action="store_true")
    args = ap.parse_args()
    root = Path(".").resolve()
    run_cell_fuse(
        root=root,
        domain=args.domain,
        cell=args.cell,
        max_samples=args.max_samples,
        seed=args.seed,
        windows_path=Path(args.windows) if args.windows else None,
        ce_npz=Path(args.ce_cache) if args.ce_cache else None,
        gat_eval=Path(args.gat_eval) if args.gat_eval else None,
        asset_dir=Path(args.asset_dir) if args.asset_dir else None,
        force_e4=args.force_e4,
        force_ce=args.force_ce,
    )


if __name__ == "__main__":
    main()
