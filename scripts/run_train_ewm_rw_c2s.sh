#!/usr/bin/env bash
# Full ranker retraining on the domain-local ``ewm_rw`` graph.
#
# Pipeline (same recipe as run_domain_rr_c2s.sh):
#   0c) domain struct logistic (WITH_EMB=1 dump, infer RR_STRUCT_COARSE=linear)
#   1)  citelink + gate
#   2)  C2s CE stage1 (2/1) + HN stage2 (2/0)
#   3)  n80 pure CE
#
# Graph: embodied_world_model_retrieval_rw/description/test_graph_rr.o2.gexf
# Split: data/splits_p3 (frontier margin=1, test412 comparable)
# Jsonl: data/reference_recommend_o2
#
# Usage:
#   scripts/run_train_ewm_rw_c2s.sh
#   FORCE=1 GPUS=0,1,2,3 CE_GPUS=0 nohup bash scripts/run_train_ewm_rw_c2s.sh \
#     > logs/ewm_rw_o2_c2s.log 2>&1 &
set -euo pipefail
ROOT="${RWCITE_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$ROOT"
export RWCITE_ROOT="$ROOT"

LANE="embodied_world_model_retrieval_rw"
GEXF="$LANE/description/test_graph_rr.o2.gexf"
[[ -f "$GEXF" ]] || { echo "ERROR: missing O2 gexf $GEXF" >&2; exit 1; }
ln -sfn "$ROOT/$GEXF" "$LANE/description/test_graph_rr.gexf"

_ban_as() {
  case "${1:-}" in
    *rr-pool-ranker-v1*)
      echo "ERROR: deprecated shared artifact forbidden on ewm_rw: $1" >&2
      exit 1
      ;;
  esac
}

export DOMAIN=ewm_rw
export DATA_TAG=o2_sclean
export SPLIT_DIR="${SPLIT_DIR:-$LANE/data/splits_p3}"
export JSONL_DIR="${JSONL_DIR:-$LANE/data/reference_recommend_o2}"
export TRAIN_JSONL="${TRAIN_JSONL:-$JSONL_DIR/train.jsonl}"
export TEST_JSONL="${TEST_JSONL:-$JSONL_DIR/test.jsonl}"

export STRUCT_MODEL="${STRUCT_MODEL:-$LANE/ranker/struct/model.npz}"
export STRUCT_FEATS="${STRUCT_FEATS:-$LANE/ranker/struct/feats_o2}"
export CITELINK_OUT="${CITELINK_OUT:-$LANE/ranker/citelink_o2/model.npz}"
export CITELINK_CACHE="${CITELINK_CACHE:-$LANE/ranker/pool/citelink_o2_pairs.npz}"
export BASE_PAIRS="${BASE_PAIRS:-$LANE/ranker/pool/ce_base_pairs_o2.jsonl}"
export HN_PAIRS="${HN_PAIRS:-$LANE/ranker/pool/ce_hn_pairs_o2.jsonl}"
export OUT_STAGE1="${OUT_STAGE1:-$LANE/ranker/ce_sent/stage1_o2/}"
export OUT="${OUT:-$LANE/ranker/ce_sent/c2s-hn-scibert_scivocab_uncased.o2/}"
export EVAL_DIR="${EVAL_DIR:-$LANE/ranker/eval/}"
export REPORT="${REPORT:-$LANE/ranker/eval/c2s_report_o2.json}"
export GATE_OUT="${GATE_OUT:-$LANE/ranker/eval/citelink_gate_o2.json}"
export CITELINK_EVAL_TAG="${CITELINK_EVAL_TAG:-citelink_ewm_rw_o2_n80}"
export TAG_PURE="${TAG_PURE:-ewm_rw_o2_c2s_pure_n80}"
export TAG_BLEND="${TAG_BLEND:-ewm_rw_o2_c2s_blend_n80}"
export LOG_DIR="${LOG_DIR:-logs/rr_ltr_ewm_rw_o2}"

export DO_PREP=0
export GPUS="${GPUS:-0,1,2,3}"
export CE_GPUS="${CE_GPUS:-0}"
export MAX_LENGTH="${MAX_LENGTH:-256}"
export MAX_SAMPLES="${MAX_SAMPLES:-80}"
export SKIP_BLEND="${SKIP_BLEND:-1}"
export REBUILD_PAIRS="${REBUILD_PAIRS:-1}"
export FORCE="${FORCE:-1}"
export SKIP_POOL_RANKER="${SKIP_POOL_RANKER:-0}"

export RR_STRUCT_COARSE="${RR_STRUCT_COARSE:-linear}"
export RR_RANKER_STRUCT_PREFILTER="${RR_RANKER_STRUCT_PREFILTER:-800}"
export RR_RANKER_FULL_EMB="${RR_RANKER_FULL_EMB:-0}"
export RR_RANKER_WITH_EMB=0
export WITH_EMB="${WITH_EMB:-1}"
export LOSS="${LOSS:-logistic}"
unset RR_RANKER_PATH || true
unset RR_RANKER_MLP_PATH || true

_ban_as "$STRUCT_MODEL"
_ban_as "$OUT"
_ban_as "$CITELINK_OUT"

[[ -f "$TRAIN_JSONL" ]] || { echo "ERROR: missing $TRAIN_JSONL" >&2; exit 1; }
[[ -f "$SPLIT_DIR/split_meta.json" ]] || { echo "ERROR: missing $SPLIT_DIR/split_meta.json" >&2; exit 1; }

echo "== ewm_rw O2 full C2s: gexf=$GEXF struct=$STRUCT_MODEL coarse=$RR_STRUCT_COARSE =="
echo "   train=$TRAIN_JSONL test=$TEST_JSONL out=$OUT"
exec bash scripts/run_domain_rr_c2s.sh --domain ewm_rw
