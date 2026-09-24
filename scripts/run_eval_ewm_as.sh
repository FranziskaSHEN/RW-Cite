#!/usr/bin/env bash
# Evaluate the legacy external-graph compatibility lane. Structural
# shortlisting uses the domain-local linear
# model.npz (RR_STRUCT_COARSE=linear). Incompatible shared checkpoints are banned.
#
# Usage:
#   TAG=ewm_as_rw_c2s_lin_pure_n80 scripts/run_eval_ewm_as.sh
set -euo pipefail
ROOT="${RWCITE_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$ROOT"
export RWCITE_ROOT="$ROOT"

AS_ROOT="${RWCITE_EXTERNAL_EWM_ROOT:?Set RWCITE_EXTERNAL_EWM_ROOT for this legacy compatibility lane}"
LANE="embodied_world_model_retrieval_as"

_ban_external_checkpoint() {
  local p="${1:-}"
  case "$p" in
    *rr-pool-ranker-v1*)
      echo "ERROR: shared checkpoint forbidden in the external-graph lane: $p" >&2
      exit 1
      ;;
  esac
}

export DOMAIN=ewm_as
export TEST_JSONL="${TEST_JSONL:-$AS_ROOT/datasets/reference_recommend_ewm_time_elig10_v2/test.jsonl}"
export RANKER_CE_SENT="${RANKER_CE_SENT:-$LANE/ranker/ce_sent/c2s-hn-scibert_scivocab_uncased.lin}"
export OUT_DIR="${OUT_DIR:-$LANE/ranker/eval}"
export LOG_DIR="${LOG_DIR:-logs/rr_ltr_ewm_as}"
export MAX_SAMPLES="${MAX_SAMPLES:-80}"
export SEED="${SEED:-42}"
export N="${N:-50}"
export GPUS="${GPUS:-0,1,2,3}"
export TAG="${TAG:?TAG required}"

export RR_CE_SENT_CITELINK_BLEND=0
export RR_CE_SENT_SHORT=1
export RR_CE_SENT_WINDOW="${RR_CE_SENT_WINDOW:-400}"
export RR_CE_SENT_MAX_SENTS=3
export RR_CE_SENT_MAX_LEN=256
export RR_CAND_N="${RR_CAND_N:-50}"

export RR_STRUCT_COARSE="${RR_STRUCT_COARSE:-linear}"
export RR_RANKER_STRUCT_PREFILTER="${RR_RANKER_STRUCT_PREFILTER:-800}"
export RR_RANKER_FULL_EMB="${RR_RANKER_FULL_EMB:-0}"
export RR_RANKER_WITH_EMB=0
export STRUCT_MODEL="${STRUCT_MODEL:-$ROOT/$LANE/ranker/struct/model.npz}"
export RR_RANKER_PATH="$(cd "$(dirname "$STRUCT_MODEL")" && pwd)/$(basename "$STRUCT_MODEL")"
unset RR_RANKER_MLP_PATH || true

_ban_external_checkpoint "$RR_RANKER_PATH"
_ban_external_checkpoint "$RANKER_CE_SENT"
[[ -f "$RR_RANKER_PATH" ]] || { echo "ERROR: missing $RR_RANKER_PATH" >&2; exit 1; }
[[ -d "$RANKER_CE_SENT" ]] || { echo "ERROR: missing CE dir $RANKER_CE_SENT" >&2; exit 1; }

echo "== ewm_as eval: tag=$TAG samples=$MAX_SAMPLES struct=$RR_RANKER_PATH coarse=$RR_STRUCT_COARSE ce=$RANKER_CE_SENT =="
exec scripts/run_eval_rr_pool_ranker_4gpu.sh
