#!/usr/bin/env bash
# Fully eliminate frontier/test out-edge leakage on ewm (§4 of GRAPH_FRONTIER_TEST_IMPACT):
#   1) export G_nofuture (strip test∪frontier out-edges)
#   2) retrain struct → citelink → CE on G_nofuture (side artifact dirs)
#   3) full-test eval with RR_STRUCT_MASK_QUERY_OUT + FUTURE_OUT on G_nofuture
#
# Usage:
#   GPUS=0,1,2,3 CE_GPUS=0 bash scripts/run_ewm_nofuture_eliminate.sh
#   SKIP_TRAIN=1 bash scripts/run_ewm_nofuture_eliminate.sh   # eval only if artifacts exist
set -euo pipefail
ROOT="${RWCITE_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$ROOT"
export RWCITE_ROOT="$ROOT"

PY="${RWCITE_PYTHON:-${PY:-python3}}"
if [[ -x "${ROOT}/.venv/bin/python" && -z "${RWCITE_PYTHON:-}" && "${PY}" == "python3" ]]; then
  PY="${ROOT}/.venv/bin/python"
fi
module load cuda-12.8 2>/dev/null || true

DOMAIN=ewm
GPUS="${GPUS:-0,1,2,3}"
CE_GPUS="${CE_GPUS:-0}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
SKIP_EVAL="${SKIP_EVAL:-0}"
MAX_SAMPLES="${MAX_SAMPLES:-381}"
N="${N:-30}"

WD=embodied_world_model_retrieval
SPLIT_DIR="$WD/data/splits"
JSONL_DIR="$WD/data/reference_recommend"
GEXF_IN="$WD/description/test_graph_rr.o2.gexf"
GEXF_NF="$WD/description/test_graph_rr.o2.nofuture.gexf"

STRUCT_MODEL="$WD/ranker/struct_nofuture/model.npz"
STRUCT_FEATS="$WD/ranker/struct_nofuture/feats"
CITELINK_OUT="$WD/ranker/citelink_nofuture/model.npz"
CITELINK_CACHE="$WD/ranker/pool/citelink_pairs_nofuture.npz"
BASE_PAIRS="$WD/ranker/pool/ce_base_pairs_nofuture.jsonl"
HN_PAIRS="$WD/ranker/pool/ce_hn_pairs_nofuture.jsonl"
OUT_STAGE1="$WD/ranker/ce_sent/stage1_nofuture"
OUT="$WD/ranker/ce_sent/c2s-hn-scibert_scivocab_uncased_nofuture"
EVAL_DIR="$WD/ranker/eval"
GATE_OUT="$WD/ranker/eval/citelink_gate_nofuture.json"
REPORT="$WD/ranker/eval/c2s_report_nofuture_scibert_scivocab_uncased.json"
LOG_DIR="logs/rr_ltr_ewm_c2s_nofuture"
TAG_FULL="ewm_v5_scibert_c2s_nofuture_test381"

echo "== [1/3] export G_nofuture =="
"$PY" -m rwcite.cli.export_nofuture_gexf \
  --domain ewm \
  --gexf-in "$GEXF_IN" \
  --gexf-out "$GEXF_NF" \
  --split-dir "$SPLIT_DIR"

export RR_GEXF_OVERRIDE="$ROOT/$GEXF_NF"
# Masks are eval-time (§4.1/4.3); training uses physical G_nofuture only (§4.2).
unset RR_STRUCT_MASK_QUERY_OUT RR_STRUCT_MASK_FUTURE_OUT RR_SPLIT_DIR || true

if [[ "$SKIP_TRAIN" != "1" ]]; then
  echo "== [2/3] retrain struct→citelink→CE on G_nofuture (side dirs) =="
  FORCE=1 DO_PREP=0 SKIP_REWRITE=1 \
    DOMAIN="$DOMAIN" GPUS="$GPUS" CE_GPUS="$CE_GPUS" \
    GEXF="$GEXF_NF" \
    RR_GEXF_OVERRIDE="$ROOT/$GEXF_NF" \
    SPLIT_DIR="$SPLIT_DIR" JSONL_DIR="$JSONL_DIR" \
    STRUCT_MODEL="$STRUCT_MODEL" STRUCT_FEATS="$STRUCT_FEATS" \
    CITELINK_OUT="$CITELINK_OUT" CITELINK_CACHE="$CITELINK_CACHE" \
    BASE_PAIRS="$BASE_PAIRS" HN_PAIRS="$HN_PAIRS" \
    OUT_STAGE1="$OUT_STAGE1" OUT="$OUT" \
    EVAL_DIR="$EVAL_DIR" GATE_OUT="$GATE_OUT" REPORT="$REPORT" \
    LOG_DIR="$LOG_DIR" \
    MAX_SAMPLES=80 \
    bash scripts/run_domain_rr_c2s.sh --domain ewm
else
  echo "== [2/3] SKIP_TRAIN=1 =="
fi

if [[ "$SKIP_EVAL" == "1" ]]; then
  echo "== [3/3] SKIP_EVAL=1 =="
  exit 0
fi

echo "== [3/3] full-test eval on G_nofuture + masks =="
source configs/env_domain_rr.sh ewm
export RR_GEXF_OVERRIDE="$ROOT/$GEXF_NF"
export RR_SPLIT_DIR="$ROOT/$SPLIT_DIR"
export RR_STRUCT_MASK_QUERY_OUT=1
export RR_STRUCT_MASK_FUTURE_OUT=1
export RR_RANKER_PATH="$(readlink -f "$STRUCT_MODEL")"
export RR_STRUCT_COARSE=linear
export RR_RANKER_FULL_EMB=0
export RR_RANKER_WITH_EMB=0
export RR_CE_SENT_CITELINK_BLEND=0

DOMAIN=ewm \
  TEST_JSONL="$JSONL_DIR/test.jsonl" \
  RANKER_CE_SENT="$OUT" \
  OUT_DIR="$EVAL_DIR" \
  LOG_DIR="$LOG_DIR" \
  TAG="$TAG_FULL" \
  MAX_SAMPLES="$MAX_SAMPLES" \
  N="$N" \
  SEED=42 \
  GPUS="$GPUS" \
  RR_GEXF_OVERRIDE="$ROOT/$GEXF_NF" \
  RR_SPLIT_DIR="$ROOT/$SPLIT_DIR" \
  RR_STRUCT_MASK_QUERY_OUT=1 \
  RR_STRUCT_MASK_FUTURE_OUT=1 \
  bash scripts/run_eval_rr_pool_ranker_4gpu.sh

echo "== done =="
echo "  gexf:   $GEXF_NF"
echo "  struct: $STRUCT_MODEL"
echo "  ce:     $OUT"
echo "  eval:   $EVAL_DIR/${TAG_FULL}_merged.json"
echo "  report: $REPORT"
