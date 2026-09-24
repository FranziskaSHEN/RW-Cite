#!/usr/bin/env bash
# Gate eval for the EWM independent-rebuild lane (domain `ewm_rw`).
#
# Every asset resolved here is RW-built: graph, split, jsonl, CE weights, domain
# embeddings and the structural shortlist model. RR_RANKER_PATH is always set
# explicitly because rr_ranker_params() otherwise falls back to the deprecated
# shared model at models/rr-pool-ranker-v1/.
#
# Standard struct: RR_STRUCT_COARSE=linear + domain model.npz (logistic+emb train).
#
#   STRUCT_MODEL=none                        -> heuristic shortlist
#   STRUCT_MODEL=<path>/model.npz            -> domain/linear Ranker
#
# Usage: TAG=... [MAX_SAMPLES=431] [STRUCT_MODEL=...] scripts/run_eval_ewm_rw.sh
set -euo pipefail
ROOT="${RWCITE_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$ROOT"
export RWCITE_ROOT="$ROOT"

LANE="embodied_world_model_retrieval_rw"

export DOMAIN=ewm_rw
export TEST_JSONL="${TEST_JSONL:-$LANE/data/reference_recommend_o2/test.jsonl}"
export RANKER_CE_SENT="${RANKER_CE_SENT:-$LANE/ranker/ce_sent/c2s-hn-scibert_scivocab_uncased.o2}"
export OUT_DIR="${OUT_DIR:-$LANE/ranker/eval}"
export LOG_DIR="${LOG_DIR:-logs/rr_ltr_ewm_rw}"
export MAX_SAMPLES="${MAX_SAMPLES:-431}"
export SEED="${SEED:-42}"
export N="${N:-50}"
export GPUS="${GPUS:-0,1,2,3}"
export TAG="${TAG:?TAG required}"

# Protocol: pure CE (beta=0), sclean SciBERT C2s, ml=256.
export RR_CE_SENT_CITELINK_BLEND=0
export RR_CE_SENT_SHORT=1
export RR_CE_SENT_WINDOW="${RR_CE_SENT_WINDOW:-400}"
export RR_CE_SENT_MAX_SENTS=3
export RR_CE_SENT_MAX_LEN=256
export RR_STRUCT_COARSE="${RR_STRUCT_COARSE:-linear}"
export RR_RANKER_STRUCT_PREFILTER="${RR_RANKER_STRUCT_PREFILTER:-800}"
export RR_RANKER_FULL_EMB=0
export RR_RANKER_WITH_EMB=0
export RR_CAND_N="${RR_CAND_N:-50}"
unset RR_RANKER_MLP_PATH || true

STRUCT_MODEL="${STRUCT_MODEL:-$ROOT/$LANE/ranker/struct/model.npz}"
if [[ "$STRUCT_MODEL" == "none" ]]; then
  # Non-existent path -> load_ranker() returns None -> heuristic_score.
  export RR_RANKER_PATH="$ROOT/models/__no_struct_model__/model.npz"
else
  if [[ ! -f "$STRUCT_MODEL" ]]; then
    echo "ERROR: STRUCT_MODEL not found: $STRUCT_MODEL" >&2
    exit 1
  fi
  case "$STRUCT_MODEL" in
    *rr-pool-ranker-v1*)
      echo "ERROR: deprecated shared struct artifact forbidden on ewm_rw" >&2
      exit 1
      ;;
  esac
  export RR_RANKER_PATH="$(cd "$(dirname "$STRUCT_MODEL")" && pwd)/$(basename "$STRUCT_MODEL")"
fi

echo "== ewm_rw eval: tag=$TAG samples=$MAX_SAMPLES struct=$STRUCT_MODEL coarse=$RR_STRUCT_COARSE =="
exec scripts/run_eval_rr_pool_ranker_4gpu.sh
