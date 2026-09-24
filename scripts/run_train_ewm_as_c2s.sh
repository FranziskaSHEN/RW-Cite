#!/usr/bin/env bash
# Retrain C2s CE on an external EWM graph using the RW-Cite standard
# struct coarse: domain logistic ``model.npz`` + RR_STRUCT_COARSE=linear.
# All ranker weights are RW-built under embodied_world_model_retrieval_as/;
# Shared checkpoints are banned; only the external graph and evaluation JSONL
# may come from outside the repository's domain lane.
#
# Prerequisites: domain-local linear struct at $LANE/ranker/struct/model.npz
#   FORCE=1 WITH_EMB=1 LOSS=logistic DOMAIN=ewm_as GPUS=0,1,2,3 \
#     scripts/run_domain_train_pool_ranker.sh
#
# Usage:
#   scripts/run_train_ewm_as_c2s.sh
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
export SPLIT_DIR="$LANE/data/splits"
export JSONL_DIR="$LANE/data/reference_recommend"
export TRAIN_JSONL="$JSONL_DIR/train.jsonl"
# Graph-aligned evaluation JSONL; it is not a model checkpoint.
export TEST_JSONL="$AS_ROOT/datasets/reference_recommend_ewm_time_elig10_v2/test.jsonl"

export CITELINK_OUT="$LANE/ranker/citelink/model.npz"
export CITELINK_CACHE="$LANE/ranker/pool/citelink_cache.npz"
export GATE_OUT="$LANE/ranker/eval/citelink_gate.json"
export CITELINK_EVAL_TAG="citelink_ewm_as_n80"
export SKIP_CITELINK="${SKIP_CITELINK:-1}"

export BASE_PAIRS="$LANE/ranker/pool/ce_base_pairs_lin.jsonl"
export HN_PAIRS="$LANE/ranker/pool/ce_hn_pairs_lin.jsonl"
export OUT_STAGE1="$LANE/ranker/ce_sent/stage1_lin/"
export OUT="$LANE/ranker/ce_sent/c2s-hn-scibert_scivocab_uncased.lin/"
export EVAL_DIR="$LANE/ranker/eval/"
export REPORT="$LANE/ranker/eval/c2s_report_lin_scibert_scivocab_uncased.json"
export TAG_PURE="ewm_as_rw_c2s_lin_pure_n80"
export TAG_BLEND="ewm_as_rw_c2s_lin_blend_n80"
export LOG_DIR="${LOG_DIR:-logs/rr_ltr_ewm_as_c2s_lin}"
export DATA_TAG=as_sclean
export ANCHOR_HITS_AT_10="${ANCHOR_HITS_AT_10:-8.0625}"

export DO_PREP=0
export GPUS="${GPUS:-0,1,2,3}"
export CE_GPUS="${CE_GPUS:-0}"
export MAX_LENGTH="${MAX_LENGTH:-256}"
export MAX_SAMPLES="${MAX_SAMPLES:-80}"
export SKIP_BLEND="${SKIP_BLEND:-1}"
# Struct model trained separately via run_domain_train_pool_ranker.sh
export SKIP_POOL_RANKER=1
export REBUILD_PAIRS="${REBUILD_PAIRS:-1}"
export FORCE="${FORCE:-1}"

# Standard linear coarse — domain-local logistic model.npz, infer emb=0.
export RR_STRUCT_COARSE="${RR_STRUCT_COARSE:-linear}"
export RR_RANKER_STRUCT_PREFILTER="${RR_RANKER_STRUCT_PREFILTER:-800}"
export RR_RANKER_FULL_EMB="${RR_RANKER_FULL_EMB:-0}"
export RR_RANKER_WITH_EMB=0
export STRUCT_MODEL="${STRUCT_MODEL:-$ROOT/$LANE/ranker/struct/model.npz}"
export RR_RANKER_PATH="$(cd "$(dirname "$STRUCT_MODEL")" && pwd)/$(basename "$STRUCT_MODEL")"
unset RR_RANKER_MLP_PATH || true

_ban_external_checkpoint "$RR_RANKER_PATH"
_ban_external_checkpoint "$OUT"
_ban_external_checkpoint "$CITELINK_OUT"
[[ -f "$RR_RANKER_PATH" ]] || {
  echo "ERROR: missing domain struct $RR_RANKER_PATH" >&2
  echo "  Train first: FORCE=1 WITH_EMB=1 LOSS=logistic DOMAIN=ewm_as GPUS=0,1,2,3 \\" >&2
  echo "    scripts/run_domain_train_pool_ranker.sh" >&2
  exit 1
}
[[ -f "$CITELINK_OUT" ]] || { echo "ERROR: missing citelink $CITELINK_OUT" >&2; exit 1; }

echo "== ewm_as RW C2s (linear struct standard, no AS weights) on AS graph =="
echo "   struct=$RR_RANKER_PATH coarse=$RR_STRUCT_COARSE"
echo "   train=$TRAIN_JSONL test=$TEST_JSONL out=$OUT"
echo "   anchor_hits_at_10=$ANCHOR_HITS_AT_10"
exec bash scripts/run_domain_rr_c2s.sh --domain ewm_as
