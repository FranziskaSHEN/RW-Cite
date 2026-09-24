"""Paper score-and-rank fusion recipe.

For every configured domain the reported score is
  s = rrf_w · z(RRF_k(CE, GAT)) + (1 − rrf_w) · score_fusion_α
with α=0.4, k=20, w=0.4.

Historical ``l0`` and ``l0_rrf`` identifiers are retained for frozen-artifact
compatibility. Paths resolve through ``rwcite.gat.domain_layout``.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from rwcite.gat import BASELINE_FULL381_HITS_AT_10
from rwcite.gat import protocol as P
from rwcite.gat.compare import compare
from rwcite.gat.domain_layout import effective_nofuture_gexf, resolve_gat_domain
from rwcite.gat.fuse_ce import assert_ce_cache_matches, blend_sweep, cache_ce_scores


DEFAULT_ALPHA = 0.4
DEFAULT_RRF_K = 20
DEFAULT_RRF_W = 0.4
PRIOR_ZBLEND_L0_HITS_AT_10 = 3.924
PRIOR_HN_L0_HITS_AT_10 = 3.79002624671916


def ensure_l0_default(
    *,
    root: Path,
    domain: str = "ewm",
    alpha: float = DEFAULT_ALPHA,
    rrf_k: int = DEFAULT_RRF_K,
    rrf_w: float = DEFAULT_RRF_W,
    force_cache: bool = False,
    force_blend: bool = False,
) -> dict[str, Any]:
    dpaths = resolve_gat_domain(domain, root)
    out_dir = root / dpaths.gat_mvp_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    eval_dir = root / dpaths.eval_dir
    eval_dir.mkdir(parents=True, exist_ok=True)

    meta = dpaths.split_dir / "split_meta.json"
    n_test = 381
    if meta.is_file():
        n_test = int(json.loads(meta.read_text(encoding="utf-8")).get("n_test_sources") or n_test)
    canonical_tag = f"{dpaths.domain}_gat_l0_default_test{n_test}"
    ce_cache_name = f"ce_scores_test{n_test}_sent_s1.npz"
    windows_path = out_dir / f"windows_test{n_test}.npz"
    # Legacy ewm names only when still on the old 381 split
    if not windows_path.is_file() and dpaths.domain == "ewm" and n_test == 381:
        legacy = out_dir / "windows_test381.npz"
        if legacy.is_file():
            windows_path = legacy
            ce_cache_name = "ce_scores_test381_sent_s1.npz"
            canonical_tag = "ewm_gat_l0_default_test381"

    ce_npz = out_dir / ce_cache_name
    gat_eval = eval_dir / f"{dpaths.domain}_gat_mvp_test{n_test}_e4_scores_merged.json"
    if not gat_eval.is_file() and dpaths.domain == "ewm" and n_test == 381:
        alt = eval_dir / "ewm_gat_mvp_test381_e4_scores_merged.json"
        if alt.is_file():
            gat_eval = alt

    gexf = effective_nofuture_gexf(dpaths, root)
    ce_path = dpaths.ce_stage1

    if force_cache or not ce_npz.is_file():
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

    if not gat_eval.is_file():
        from rwcite.gat.evaluate import evaluate
        import torch

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = out_dir / "ckpt_e4.pt"
        if not ckpt.is_file():
            raise SystemExit(f"missing {ckpt}")
        print(f"dump-scores from {ckpt}", flush=True)
        result = evaluate(
            out_dir=out_dir,
            windows_path=windows_path,
            ckpt_path=ckpt,
            ablation="E4",
            device=device,
            fair=True,
            dump_scores=True,
        )
        gat_eval.write_text(json.dumps(result, indent=2), encoding="utf-8")

    rrf_eval = eval_dir / f"{dpaths.domain}_gat_l0_rrf_test{n_test}_merged.json"
    # Do not promote old-split (381) RRF onto a new n_test canonical path.
    if not rrf_eval.is_file() and dpaths.domain == "ewm" and n_test == 381:
        alt = eval_dir / "ewm_gat_l0_rrf_test381_merged.json"
        if alt.is_file():
            rrf_eval = alt

    dest = eval_dir / f"{canonical_tag}_merged.json"
    reused_rrf = False
    if (
        not force_blend
        and rrf_eval.is_file()
        and float(alpha) == DEFAULT_ALPHA
        and int(rrf_k) == DEFAULT_RRF_K
        and float(rrf_w) == DEFAULT_RRF_W
    ):
        summary = json.loads(rrf_eval.read_text(encoding="utf-8"))["summary"]
        if int(summary.get("n") or 0) == int(n_test):
            shutil.copy2(rrf_eval, dest)
            reused_rrf = True
            mean10 = float(summary["mean_hits_at_10"])
            mean30 = float(summary["mean_hits_at_30"])
            print(f"promoted {rrf_eval.name} → {dest.name} (default≡L0_rrf)", flush=True)
        else:
            print(
                f"skip promote {rrf_eval.name}: n={summary.get('n')} != split n_test={n_test}",
                flush=True,
            )
    if not reused_rrf:
        blend_prefix = f"{dpaths.domain}_gat_l0_default_test{n_test}"
        report = blend_sweep(
            root=root,
            windows_path=windows_path,
            ce_npz=ce_npz,
            gat_eval_json=gat_eval,
            alphas=[float(alpha)],
            out_prefix=blend_prefix,
            fuse_mode="l0_rrf",
            rrf_k=int(rrf_k),
            rrf_w=float(rrf_w),
            bar_hits_at_10=float(P.L0_DEFAULT_HITS_AT_10),
        )
        a_key = str(float(alpha))
        if a_key not in report["alphas"] and str(alpha) in report["alphas"]:
            a_key = str(alpha)
        row = report["alphas"][a_key]
        src = Path(row["path"])
        if force_blend or not dest.is_file() or src.resolve() != dest.resolve():
            shutil.copy2(src, dest)
        mean10 = float(row["summary"]["mean_hits_at_10"])
        mean30 = float(row["summary"]["mean_hits_at_30"])
        ir_summary = {
            k: row["summary"].get(k)
            for k in (
                "mean_ndcg_at_10",
                "mean_mrr_at_10",
                "mean_recall_at_30",
                "mean_recall_at_100",
                "mean_recall_at_400",
                "recall_note",
            )
            if k in row["summary"]
        }
    else:
        ir_summary = {
            k: summary.get(k)
            for k in (
                "mean_ndcg_at_10",
                "mean_mrr_at_10",
                "mean_recall_at_30",
                "mean_recall_at_100",
                "mean_recall_at_400",
                "recall_note",
            )
            if k in summary
        }

    bar_path = root / P.BASELINE_FULL381_MERGED
    cmp_ce = None
    if dpaths.domain == "ewm" and bar_path.is_file():
        cmp_ce = compare(bar_path, dest, bar=BASELINE_FULL381_HITS_AT_10)

    recipe = {
        "name": "l0_default",
        "domain": dpaths.domain,
        "protocol": "ce_gat_l0_rrf_hybrid",
        "fuse_mode": "l0_rrf",
        "alpha_gat": float(alpha),
        "rrf_k": int(rrf_k),
        "rrf_w": float(rrf_w),
        "formula": "s = rrf_w * z(RRF_k(CE,E4)) + (1-rrf_w) * L0_alpha",
        "ce_mode": "ce_sent_short_maxlen256",
        "ce_path": str(ce_path),
        "ce_checkpoint": "stage1_nofuture",
        "prior_zblend_l0_hits_at_10": PRIOR_ZBLEND_L0_HITS_AT_10,
        "prior_hn_l0_hits_at_10": PRIOR_HN_L0_HITS_AT_10,
        "ce_cache": str(ce_npz),
        "gat_ckpt": str(out_dir / "ckpt_e4.pt"),
        "gat_eval": str(gat_eval),
        "windows": str(windows_path),
        "canonical_eval": str(dest),
        "mean_hits_at_10": mean10,
        "mean_hits_at_30": mean30,
        **ir_summary,
        "promoted_from_l0_rrf_eval": reused_rrf,
        "compare_vs_ce_bar": cmp_ce,
        "env": {
            "RR_CE_SENT_SHORT": "1",
            "RR_CE_SENT_MAX_SENTS": "3",
            "RR_CE_SENT_MAX_LEN": "256",
            "RR_CE_SENT_CITELINK_BLEND": "0",
        },
        "note": (
            "Production default = L0_rrf. stage1 CE only; single-GPU train. "
            f"domain={dpaths.domain}"
        ),
    }
    recipe_path = out_dir / "l0_default_recipe.json"
    recipe_path.write_text(json.dumps(recipe, indent=2), encoding="utf-8")
    rrf_recipe = dict(recipe)
    rrf_recipe["name"] = "l0_rrf"
    rrf_recipe["alias_of"] = "l0_default"
    (out_dir / "l0_rrf_recipe.json").write_text(
        json.dumps(rrf_recipe, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "domain": dpaths.domain,
                "fuse_mode": "l0_rrf",
                "mean_hits_at_10": mean10,
                "canonical_eval": str(dest),
            },
            indent=2,
        ),
        flush=True,
    )
    print(f"wrote {recipe_path}", flush=True)
    return recipe


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Ensure operational L0 default (= L0_rrf hybrid)"
    )
    ap.add_argument(
        "--domain",
        default="ewm",
        choices=("ewm", "sqc", "gw", "fno", "wsi", "cosmo", "qopt", "sce", "hep", "exo", "radio", "radseg", "driving"),
    )
    ap.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    ap.add_argument("--rrf-k", type=int, default=DEFAULT_RRF_K)
    ap.add_argument("--rrf-w", type=float, default=DEFAULT_RRF_W)
    ap.add_argument("--force-cache", action="store_true")
    ap.add_argument("--force-blend", action="store_true")
    args = ap.parse_args(argv)
    ensure_l0_default(
        root=P.root_path(),
        domain=args.domain,
        alpha=args.alpha,
        rrf_k=args.rrf_k,
        rrf_w=args.rrf_w,
        force_cache=args.force_cache,
        force_blend=args.force_blend,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
