#!/usr/bin/env bash
# Pack a paper-aligned domain graph, frozen split, ranking artifacts, and
# evaluation outputs for Hugging Face / offline reproduction.
#
# See docs/DATASETS_HF_UPLOAD.md.
#
# Usage:
#   bash scripts/pack_domain_release.sh --domain ewm --tag 20260924
#   bash scripts/pack_domain_release.sh --all-paper-domains --tag 20260924
#   OUT_DIR=/data/releases CHECKSUMS=1 bash scripts/pack_domain_release.sh --domain gw --no-tar
#
# Outputs under OUT_DIR (default: releases/huggingface/domains):
#   RW-Cite-domain-<dom>-r<tag>/
#   RW-Cite-domain-<dom>-r<tag>.tar.gz
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${OUT_DIR:-$ROOT/releases/huggingface/domains}"
DOMAIN=""
TAG=""
MAKE_TAR="${MAKE_TAR:-1}"
CHECKSUMS="${CHECKSUMS:-0}"
ALL_PAPER_DOMAINS=0
PAPER_DOMAINS=(ewm gw driving cosmo radio wsi exo sqc fno sce)

usage() {
  sed -n '2,15p' "$0"
  exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --domain) DOMAIN="$2"; shift 2 ;;
    --tag) TAG="$2"; shift 2 ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    --all-paper-domains) ALL_PAPER_DOMAINS=1; shift ;;
    --all-3.1) ALL_PAPER_DOMAINS=1; shift ;; # legacy CLI alias
    --no-tar) MAKE_TAR=0; shift ;;
    --checksums) CHECKSUMS=1; shift ;;
    -h|--help) usage 0 ;;
    *)
      echo "Unknown arg: $1" >&2
      usage 1
      ;;
  esac
done

cd "$ROOT"
export RWCITE_ROOT="$ROOT"

PY="${RWCITE_PYTHON:-python3}"
if [[ -x "${ROOT}/.venv/bin/python" && -z "${RWCITE_PYTHON:-}" && "${PY}" == "python3" ]]; then
  PY="${ROOT}/.venv/bin/python"
fi

if [[ -z "$TAG" ]]; then
  TAG="$(date -u +%Y%m%d)"
fi
TAG="${TAG//-/}"
TAG="$(printf '%s' "$TAG" | tr -cd '0-9A-Za-z')"
if [[ -z "$TAG" ]]; then
  echo "ERROR: empty tag after normalize" >&2
  exit 1
fi

pack_one_domain() {
DOMAIN="$1"

mapfile -t _meta < <("$PY" - <<PY
import json
import yaml
from pathlib import Path
from rwcite.graph.domain_paths import (
    domain_data_paths,
    domain_layout,
    normalize_working_dir,
)

root = Path(".")
dom = "$DOMAIN"
dy = yaml.safe_load((root / "configs/domains.yaml").read_text(encoding="utf-8")) or {}
block = (dy.get("domains") or {}).get(dom)
if not block:
    raise SystemExit(f"unknown domain: {dom}")
cfg_rel = block["domain_config"]
ycfg = yaml.safe_load((root / cfg_rel).read_text(encoding="utf-8")) or {}
wd = normalize_working_dir(ycfg["data_downloading"]["download_directory"])
layout = domain_layout(wd, base_tag="scibert_scivocab_uncased")
data = domain_data_paths(wd)
gexf = block.get("gexf") or (wd + "description/test_graph_rr.gexf")
ret = block.get("retrieval_nodes") or layout["retrieval_nodes"]
n_test = 0
sm = Path(data["splits_dir"]) / "split_meta.json"
if sm.is_file():
    n_test = int(json.loads(sm.read_text(encoding="utf-8")).get("n_test_sources") or 0)
print(cfg_rel)
print(wd.rstrip("/"))
print(gexf)
print(ret)
print(layout["eval_dir"].rstrip("/"))
print(data["splits_dir"].rstrip("/"))
print(data["jsonl_dir"].rstrip("/"))
print((dy.get("domains") or {}).get(dom, {}).get("label") or dom)
print(str(n_test))
PY
)

CFG_REL="${_meta[0]}"
WORKING="${_meta[1]}"
GEXF="${_meta[2]}"
RET_NODES="${_meta[3]}"
EVAL_DIR="${_meta[4]}"
SPLITS_DIR="${_meta[5]}"
JSONL_DIR="${_meta[6]}"
DOMAIN_LABEL="${_meta[7]}"
N_TEST="${_meta[8]}"

RANK_VARIANT="live"
PIN_SPLITS=""
PIN_JSONL=""
GEXF_BASENAME=""

CE_STAGE1="${WORKING}/ranker/ce_sent/stage1_nofuture"
STRUCT_NF_DIR="${WORKING}/ranker/struct_nofuture"
STRUCT_NF_MODEL="${STRUCT_NF_DIR}/model.npz"
GAT_MVP="${WORKING}/ranker/gat_mvp"

if [[ "$DOMAIN" == "ewm" ]]; then
  RANK_VARIANT="frozen_test_manifest"
  N_TEST=381
  PIN_SPLITS="${WORKING}/data/_admit_backup_20260912_163729/splits"
  PIN_JSONL="${WORKING}/data/_admit_backup_20260912_163729/reference_recommend"
fi

GEXF_BASENAME="$(basename "$GEXF")"
if [[ ! -f "$GEXF" ]]; then
  alt="${WORKING}/description/test_graph_rr.o2.nofuture.gexf"
  [[ -f "$alt" ]] && GEXF="$alt" && GEXF_BASENAME="$(basename "$GEXF")"
fi
if [[ ! -f "$GEXF" ]]; then
  alt="${WORKING}/description/test_graph_rr.gexf"
  [[ -f "$alt" ]] && GEXF="$alt" && GEXF_BASENAME="$(basename "$GEXF")"
fi

missing=()
[[ -f "$CFG_REL" ]] || missing+=("$CFG_REL")
[[ -f "$GEXF" ]] || missing+=("$GEXF (domain GEXF)")
[[ -f "${GAT_MVP}/ckpt_e4.pt" ]] || missing+=("${GAT_MVP}/ckpt_e4.pt (GAT E4)")
[[ -f "$STRUCT_NF_MODEL" ]] || missing+=("$STRUCT_NF_MODEL (struct_nofuture; Wave S)")
[[ -d "$CE_STAGE1" && -n "$(find "$CE_STAGE1" -type f 2>/dev/null | head -1)" ]] \
  || missing+=("$CE_STAGE1 (stage1 CE; Wave CE)")
if [[ -n "$PIN_SPLITS" && ! -d "$PIN_SPLITS" ]]; then
  missing+=("$PIN_SPLITS (pinned split)")
fi
if [[ "$N_TEST" != "0" ]]; then
  [[ -f "${GAT_MVP}/windows_test${N_TEST}.npz" ]] || missing+=("${GAT_MVP}/windows_test${N_TEST}.npz")
  [[ -f "${GAT_MVP}/ce_scores_test${N_TEST}_sent_s1.npz" ]] \
    || missing+=("${GAT_MVP}/ce_scores_test${N_TEST}_sent_s1.npz")
  [[ -f "${EVAL_DIR}/${DOMAIN}_gat_mvp_test${N_TEST}_e4_scores_merged.json" ]] \
    || missing+=("${EVAL_DIR}/${DOMAIN}_gat_mvp_test${N_TEST}_e4_scores_merged.json")
fi
for req in id_map.json neighbors_k32.npz node_scibert_whitened.npy \
  adj_in.npz adj_out.npz adj_cocite.npz fold_masks.json; do
  [[ -f "${GAT_MVP}/$req" ]] || missing+=("${GAT_MVP}/$req")
done
for fold in 0 1 2 3; do
  for kind in in out cocite; do
    req="adj_${kind}_fold${fold}.npz"
    [[ -f "${GAT_MVP}/$req" ]] || missing+=("${GAT_MVP}/$req")
  done
done
if [[ "$RANK_VARIANT" == "live" ]]; then
  [[ -f "${GAT_MVP}/l0_default_recipe.json" ]] || missing+=("${GAT_MVP}/l0_default_recipe.json (L0)")
fi
if ((${#missing[@]})); then
  echo "ERROR: required paper-aligned artifacts are missing:" >&2
  printf '  - %s\n' "${missing[@]}" >&2
  exit 1
fi

STAGE_NAME="RW-Cite-domain-${DOMAIN}-r${TAG}"
STAGE="${OUT_DIR}/${STAGE_NAME}"
ARCHIVE="${OUT_DIR}/${STAGE_NAME}.tar.gz"

echo "Packing domain release"
echo "  domain=$DOMAIN ($DOMAIN_LABEL) tag=r$TAG variant=$RANK_VARIANT"
echo "  stage=$STAGE"

rm -rf "$STAGE"
mkdir -p "$STAGE/configs" "$STAGE/${WORKING}" "$STAGE/docs"

copy_one() {
  local src="$1" dst="$2"
  if [[ ! -e "$src" ]]; then
    return 1
  fi
  mkdir -p "$(dirname "$dst")"
  if [[ -d "$src" ]]; then
    mkdir -p "$dst"
    if cp -al "$src/." "$dst/" 2>/dev/null; then
      echo "  link-dir $src"
      return 0
    fi
    cp -a "$src/." "$dst/"
    echo "  copy-dir $src"
    return 0
  fi
  if ln "$src" "$dst" 2>/dev/null; then
    echo "  link $src"
  else
    cp -a "$src" "$dst"
    echo "  copy $src"
  fi
}

copy_opt() {
  local src="$1" dst="$2"
  if [[ -e "$src" ]]; then
    copy_one "$src" "$dst" || true
  fi
}

copy_one "$CFG_REL" "$STAGE/$CFG_REL"
copy_one "configs/env_domain_rr.sh" "$STAGE/configs/env_domain_rr.sh"

"$PY" - <<PY
import yaml
from pathlib import Path

root = Path(".")
dom = "$DOMAIN"
raw = yaml.safe_load((root / "configs/domains.yaml").read_text(encoding="utf-8")) or {}
domains = raw.get("domains") or {}
if dom not in domains:
    raise SystemExit(f"missing domain {dom}")
out = {
    "domains": {dom: domains[dom]},
    "shared": raw.get("shared") or {},
}
dst = Path("$STAGE/configs/domains.yaml")
dst.parent.mkdir(parents=True, exist_ok=True)
dst.write_text(yaml.safe_dump(out, sort_keys=False, allow_unicode=True), encoding="utf-8")
print(f"  wrote {dst.relative_to('$STAGE')}")
PY

copy_opt "docs/DATASETS_HF_UPLOAD.md" "$STAGE/DATASETS_HF_UPLOAD.md"

copy_one "$GEXF" "$STAGE/${WORKING}/description/${GEXF_BASENAME}"
if [[ "$GEXF_BASENAME" != "test_graph_rr.gexf" ]]; then
  # Convenience alias for tools that still look for the short name
  copy_opt "$GEXF" "$STAGE/${WORKING}/description/test_graph_rr.gexf"
fi
copy_opt "$RET_NODES" "$STAGE/${WORKING}/retrieval_nodes.json"
copy_opt "${WORKING}/failed_downloads.json" "$STAGE/${WORKING}/failed_downloads.json"
_meta_ret="${WORKING}/retrieval_nodes_retrieve_meta.json"
if [[ "$RET_NODES" == *.json ]]; then
  _cand="${RET_NODES%.json}_retrieve_meta.json"
  [[ -f "$_cand" ]] && _meta_ret="$_cand"
fi
copy_opt "$_meta_ret" "$STAGE/${WORKING}/retrieval_nodes_retrieve_meta.json"

_splits_src="${PIN_SPLITS:-$SPLITS_DIR}"
_jsonl_src="${PIN_JSONL:-$JSONL_DIR}"
copy_opt "$_splits_src" "$STAGE/${WORKING}/data/splits"
copy_opt "$_jsonl_src" "$STAGE/${WORKING}/data/reference_recommend"
copy_opt "${WORKING}/data/retrieval_diag.json" "$STAGE/${WORKING}/data/retrieval_diag.json"

CE_PACK_BASE="stage1_nofuture"
copy_one "$CE_STAGE1" "$STAGE/${WORKING}/ranker/ce_sent/stage1_nofuture"
mkdir -p "$STAGE/${WORKING}/ranker/struct_nofuture"
copy_one "$STRUCT_NF_MODEL" "$STAGE/${WORKING}/ranker/struct_nofuture/model.npz"
copy_opt "${STRUCT_NF_DIR}/train_report.json" "$STAGE/${WORKING}/ranker/struct_nofuture/train_report.json"
copy_opt "${STRUCT_NF_DIR}/meta.json" "$STAGE/${WORKING}/ranker/struct_nofuture/meta.json"

mkdir -p "$STAGE/${GAT_MVP}"
copy_one "${GAT_MVP}/ckpt_e4.pt" "$STAGE/${GAT_MVP}/ckpt_e4.pt"
if [[ "$RANK_VARIANT" == "live" ]]; then
  copy_one "${GAT_MVP}/l0_default_recipe.json" "$STAGE/${GAT_MVP}/l0_default_recipe.json"
  copy_opt "${GAT_MVP}/l0_rrf_recipe.json" "$STAGE/${GAT_MVP}/l0_rrf_recipe.json"
fi
if [[ "$N_TEST" != "0" ]]; then
  copy_one "${GAT_MVP}/windows_test${N_TEST}.npz" "$STAGE/${GAT_MVP}/windows_test${N_TEST}.npz"
  copy_one "${GAT_MVP}/ce_scores_test${N_TEST}_sent_s1.npz" \
    "$STAGE/${GAT_MVP}/ce_scores_test${N_TEST}_sent_s1.npz"
fi
# GAT forward (evaluate / L0) reads these beside the checkpoint.
copy_one "${GAT_MVP}/id_map.json" "$STAGE/${GAT_MVP}/id_map.json"
copy_one "${GAT_MVP}/neighbors_k32.npz" "$STAGE/${GAT_MVP}/neighbors_k32.npz"
copy_one "${GAT_MVP}/node_scibert_whitened.npy" "$STAGE/${GAT_MVP}/node_scibert_whitened.npy"
copy_opt "${GAT_MVP}/node_scibert.npy" "$STAGE/${GAT_MVP}/node_scibert.npy"
copy_opt "${GAT_MVP}/whiten.npz" "$STAGE/${GAT_MVP}/whiten.npz"
copy_one "${GAT_MVP}/fold_masks.json" "$STAGE/${GAT_MVP}/fold_masks.json"
for kind in in out cocite; do
  copy_one "${GAT_MVP}/adj_${kind}.npz" "$STAGE/${GAT_MVP}/adj_${kind}.npz"
  for fold in 0 1 2 3; do
    copy_one "${GAT_MVP}/adj_${kind}_fold${fold}.npz" \
      "$STAGE/${GAT_MVP}/adj_${kind}_fold${fold}.npz"
  done
done

# Eval reports
if [[ -d "$EVAL_DIR" ]]; then
  mkdir -p "$STAGE/${WORKING}/ranker/eval"
  shopt -s nullglob
  for f in "$EVAL_DIR"/*.json; do
    base="$(basename "$f")"
    case "$base" in
      *shard*|*score_dump*|*_scores_merged.json|*_e4_scores*) continue ;;
    esac
    case "$base" in
      "${DOMAIN}_gat_mvp_test${N_TEST}_e4_merged.json") ;;
      "${DOMAIN}_gat_l0_default_test${N_TEST}"*_merged.json) ;;
      "${DOMAIN}_ce_sent_s1_nofuture_test${N_TEST}"*_merged.json) ;;
      "${DOMAIN}_gat_ce_sent_s1_blend_test${N_TEST}_a0p0_merged.json") ;;
      "${DOMAIN}_gat_l0_rrf_test${N_TEST}"*_merged.json) ;;
      *) continue ;;
    esac
    # CE report written before the packed cache cannot be reproduced from this pack.
    if [[ "$base" == ${DOMAIN}_ce_sent_s1_nofuture_test${N_TEST}*merged.json \
       || "$base" == ${DOMAIN}_gat_ce_sent_s1_blend_test${N_TEST}_a0p0_merged.json ]]; then
      _ce_cache="${GAT_MVP}/ce_scores_test${N_TEST}_sent_s1.npz"
      if [[ -f "$_ce_cache" && "$f" -ot "$_ce_cache" ]]; then
        echo "WARN: skip stale CE eval ${base} (older than CE cache)" >&2
        continue
      fi
    fi
    copy_one "$f" "$STAGE/${WORKING}/ranker/eval/$base"
  done
  shopt -u nullglob
fi
if [[ "$N_TEST" != "0" ]]; then
  copy_one "${EVAL_DIR}/${DOMAIN}_gat_mvp_test${N_TEST}_e4_scores_merged.json" \
    "$STAGE/${WORKING}/ranker/eval/${DOMAIN}_gat_mvp_test${N_TEST}_e4_scores_merged.json"
fi

# Live l0_default_recipe.json follows the latest split. Staged recipe must match the packed eval.
if [[ "$RANK_VARIANT" != "live" ]]; then
  _l0_src="$STAGE/${WORKING}/ranker/eval/${DOMAIN}_gat_l0_default_test${N_TEST}_merged.json"
  [[ -f "$_l0_src" ]] || _l0_src="${EVAL_DIR}/${DOMAIN}_gat_l0_default_test${N_TEST}_merged.json"
  "$PY" - "$_l0_src" "$STAGE/${GAT_MVP}/l0_default_recipe.json" "$DOMAIN" "$N_TEST" "$RANK_VARIANT" "$WORKING" <<'PY'
import json, sys
from pathlib import Path
src, dst = Path(sys.argv[1]), Path(sys.argv[2])
domain, n_test, variant, working = sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6]
summary = {}
if src.is_file():
    summary = (json.loads(src.read_text(encoding="utf-8")).get("summary") or {})
rel_eval = f"{working}/ranker/eval/{domain}_gat_l0_default_test{n_test}_merged.json"
recipe = {
    "name": "l0_default",
    "domain": domain,
    "protocol": "ce_gat_l0_rrf_hybrid",
    "fuse_mode": "l0_rrf",
    "alpha_gat": 0.4,
    "rrf_k": 20,
    "rrf_w": 0.4,
    "ranker_variant": variant,
    "n_test": int(n_test),
    "windows": f"{working}/ranker/gat_mvp/windows_test{n_test}.npz",
    "ce_cache": f"{working}/ranker/gat_mvp/ce_scores_test{n_test}_sent_s1.npz",
    "gat_ckpt": f"{working}/ranker/gat_mvp/ckpt_e4.pt",
    "canonical_eval": rel_eval,
    "mean_hits_at_10": summary.get("mean_hits_at_10"),
    "mean_hits_at_30": summary.get("mean_hits_at_30"),
}
dst.parent.mkdir(parents=True, exist_ok=True)
dst.write_text(json.dumps(recipe, indent=2) + "\n", encoding="utf-8")
print(f"  wrote pinned L0 recipe variant={variant} @10={recipe['mean_hits_at_10']}")
PY
fi

gexf_base="$GEXF_BASENAME"

cat >"$STAGE/README.md" <<EOF
---
pretty_name: RW-Cite ${DOMAIN_LABEL}
task_categories:
  - text-retrieval
language:
  - en
license: other
---

# RW-Cite domain pack (\`${DOMAIN}\`, tag \`r${TAG}\`)

**${DOMAIN_LABEL}** — paper-aligned benchmark graph, frozen split, learned
structural Top-400 window, SciBERT cross-encoder, graph-attention model, and
score-and-rank fusion artifacts.

This archive is **domain artifacts only**. Code: install RW-Cite from GitHub / \`pack_github_release.sh\`.

## Contents

| Path | Role |
|------|------|
| \`configs/config_${DOMAIN}.yaml\` | Retriever / download config |
| \`configs/domains.yaml\` | Single-domain registry (gexf → o2.nofuture) |
| \`${WORKING}/description/${gexf_base}\` | Train / score citation graph |
| \`${WORKING}/data/splits/\` | Admit F q[0.10,0.99] on full O2 |
| \`${WORKING}/ranker/ce_sent/stage1_nofuture/\` | SciBERT CE stage1 (no HN) |
| \`${WORKING}/ranker/struct_nofuture/model.npz\` | Struct shortlist (nofuture) |
| \`${WORKING}/ranker/gat_mvp/ckpt_e4.pt\` | GAT production ckpt |
| \`${WORKING}/ranker/gat_mvp/id_map.json\` | Node id order for the GAT graph |
| \`${WORKING}/ranker/gat_mvp/node_scibert_whitened.npy\` | Whitened node text features |
| \`${WORKING}/ranker/gat_mvp/neighbors_k32.npz\` | L4 neighbor index |
| \`${WORKING}/ranker/gat_mvp/adj_*.npz\` | Full graph and 4-fold adjacency |
| \`${WORKING}/ranker/gat_mvp/fold_masks.json\` | Train-source fold assignment |
| \`${WORKING}/ranker/gat_mvp/windows_test${N_TEST}.npz\` | Frozen test windows |
| \`${WORKING}/ranker/gat_mvp/ce_scores_test${N_TEST}_sent_s1.npz\` | CE cache for L0 |
| \`${WORKING}/ranker/gat_mvp/l0_default_recipe.json\` | L0_rrf recipe |
| \`${WORKING}/ranker/eval/\` | CE / GAT / L0 reports, plus test E4 scores |

## Not included

- \`research_papers/\`, \`ranker/pool/\`, \`struct*/feats/\`, \`ranker/citelink/\`
- Train windows and train \`*_scores_merged.json\` dumps
- C₂s-HN CE and \`ranker/struct/\`
- Base models under \`models/base/\`

## Install

\`\`\`bash
rsync -a configs/ /path/to/RW-Cite/configs/
rsync -a ${WORKING}/ /path/to/RW-Cite/${WORKING}/
cd /path/to/RW-Cite
source configs/env_domain_rr.sh ${DOMAIN}
# L0 default: DOMAIN=${DOMAIN} bash scripts/run_gat_l0_default.sh
\`\`\`
EOF

"$PY" - "$STAGE" "$DOMAIN" "$TAG" "$CHECKSUMS" "$DOMAIN_LABEL" "$WORKING" "$gexf_base" "$N_TEST" "$CE_PACK_BASE" "$RANK_VARIANT" <<'PY'
import hashlib, json, sys
from datetime import datetime, timezone
from pathlib import Path

stage = Path(sys.argv[1])
domain = sys.argv[2]
tag = sys.argv[3]
do_hash = sys.argv[4] == "1"
label = sys.argv[5]
working = sys.argv[6]
gexf_base = sys.argv[7]
n_test = int(sys.argv[8] or 0)
ce_base = sys.argv[9]
rank_variant = sys.argv[10] if len(sys.argv) > 10 else "live"

files = []
for path in sorted(stage.rglob("*")):
    if not path.is_file() or path.name in ("MANIFEST.json", "README.md"):
        continue
    rel = path.relative_to(stage).as_posix()
    st = path.stat()
    entry = {
        "path": rel,
        "bytes": st.st_size,
        "mtime_utc": datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
    }
    if do_hash:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024 * 8), b""):
                h.update(chunk)
        entry["sha256"] = h.hexdigest()
    files.append(entry)

split_meta = {}
sm = stage / working / "data" / "splits" / "split_meta.json"
if sm.is_file():
    try:
        split_meta = json.loads(sm.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        pass

deploy = {
    "stack": "L0_rrf",
    "CE": "stage1_nofuture",
    "GAT": "ckpt_e4.pt",
    "struct": "struct_nofuture",
    "gexf": "o2.nofuture",
    "L0_ALPHA": 0.4,
    "L0_RRF_K": 20,
    "L0_RRF_W": 0.4,
    "N_TEST": n_test or None,
    "ranker_variant": rank_variant,
}
notes = [
    "Paper-aligned domain pack with split-masked graph, structural shortlist, cross-encoder, GAT, and fixed score-and-rank fusion artifacts.",
    "Includes id_map, whitened node features, neighbors, full and fold adjacency, test windows, CE cache, and test E4 scores so L0 can be re-fused.",
    "Excludes research_papers, ranker/pool, feats, citelink, C2s-HN, train windows, and train score dumps.",
]
struct_model = f"{working.rstrip('/')}/ranker/struct_nofuture/model.npz"
ce_path = f"{working.rstrip('/')}/ranker/ce_sent/{ce_base}" if ce_base else None
if not (stage / struct_model).is_file():
    struct_model = None
if ce_path and not (stage / ce_path).is_dir():
    ce_path = None

manifest = {
    "name": "RW-Cite-domain",
    "domain": domain,
    "label": label,
    "schema_version": 1,
    "tag": f"r{tag}",
    "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "working_dir": working.rstrip("/") + "/",
    "gexf": f"{working.rstrip('/')}/description/{gexf_base}",
    "struct_model": struct_model,
    "ce_path": ce_path,
    "ranker_variant": rank_variant,
    "split_meta": split_meta or None,
    "deploy": deploy,
    "base_ce": "allenai/scibert_scivocab_uncased",
    "embedder": "BAAI/bge-large-en-v1.5",
    "files": files,
    "notes": notes,
}
(stage / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
print(f"Wrote MANIFEST.json ({len(files)} files)")
PY

mkdir -p "$OUT_DIR"
if [[ "$MAKE_TAR" == "1" ]]; then
  echo "Creating $ARCHIVE ..."
  tar -C "$OUT_DIR" -czf "$ARCHIVE" "$STAGE_NAME"
  ls -lh "$ARCHIVE"
fi

echo "Done."
echo "  staged: $STAGE"
[[ "$MAKE_TAR" == "1" ]] && echo "  archive: $ARCHIVE"
echo "Upload guide: docs/DATASETS_HF_UPLOAD.md"
} # pack_one_domain

if [[ "$ALL_PAPER_DOMAINS" == "1" ]]; then
  if [[ -n "$DOMAIN" ]]; then
    echo "ERROR: use either --all-paper-domains or --domain, not both" >&2
    exit 2
  fi
  for d in "${PAPER_DOMAINS[@]}"; do
    pack_one_domain "$d"
  done
elif [[ -n "$DOMAIN" ]]; then
  pack_one_domain "$DOMAIN"
else
  echo "ERROR: --domain <tag> or --all-paper-domains required" >&2
  exit 2
fi
