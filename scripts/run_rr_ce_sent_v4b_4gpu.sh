#!/usr/bin/env bash
# v4b: short-cand + citelink hard-negs + more queries → train → eval (pure + blend).
set -euo pipefail
ROOT="${RWCITE_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$ROOT"
export RWCITE_ROOT="$ROOT"

PY="${RWCITE_PYTHON:-${PY:-python3}}"
if [[ -x "${ROOT}/.venv/bin/python" && -z "${RWCITE_PYTHON:-}" && "${PY}" == "python3" ]]; then
  PY="${ROOT}/.venv/bin/python"
fi
GPUS="${GPUS:-0,1,2,3}"
IFS=',' read -r -a GPU_ARR <<< "$GPUS"
NUM_GPUS="${#GPU_ARR[@]}"

MAX_QUERIES="${MAX_QUERIES:-2500}"
WINDOW="${WINDOW:-400}"
NEG_PER_LIST="${NEG_PER_LIST:-64}"
MAX_SENTS="${MAX_SENTS:-3}"
SEED="${SEED:-0}"
PAIRS="${PAIRS:-datasets/rr_pool_ranker/ce_sent_v4b_pairs.jsonl}"
OUT="${OUT:-models/rr-pool-ranker-ce-sent-v4b}"
BASE="${BASE:-models/base/scibert_scivocab_uncased}"
# Empty HARD_NEG_CITELINK → omit --hard-neg-citelink (domain-isolated train).
HARD_NEG_CITELINK="${HARD_NEG_CITELINK-models/rr-pool-ranker-citelink-v3c/model.npz}"
CITELINK="${CITELINK:-models/rr-pool-ranker-citelink-v3c/model.npz}"
DOMAIN="${DOMAIN:?DOMAIN required}"
TRAIN_JSONL="${TRAIN_JSONL:-datasets/reference_recommend/train.jsonl}"
LOG_DIR="${LOG_DIR:-logs/rr_ltr}"
REBUILD="${REBUILD:-1}"
DO_TRAIN="${DO_TRAIN:-1}"
DO_EVAL="${DO_EVAL:-1}"
MAX_SAMPLES="${MAX_SAMPLES:-80}"
EPOCHS="${EPOCHS:-3}"
FREEZE_EPOCHS="${FREEZE_EPOCHS:-1}"
MAX_LENGTH="${MAX_LENGTH:-256}"
BUILD_LOG_PREFIX="${BUILD_LOG_PREFIX:-v4b_build}"
TRAIN_LOG="${TRAIN_LOG:-v4b_ce_sent_train.log}"
EVAL_TAG_PURE="${EVAL_TAG_PURE:-ce_sent_v4b_pure_n80}"
EVAL_TAG_BLEND="${EVAL_TAG_BLEND:-ce_sent_v4b_n80}"
TEST_JSONL="${TEST_JSONL:-}"

mkdir -p "$LOG_DIR" "$(dirname "$PAIRS")" "$OUT"

HN_ARGS=()
if [[ -n "${HARD_NEG_CITELINK}" ]]; then
  HN_ARGS=(--hard-neg-citelink "$HARD_NEG_CITELINK")
fi

if [[ "$REBUILD" == "1" ]]; then
  echo "== CE pair build: domain=${DOMAIN} nq=${MAX_QUERIES} short shards=${NUM_GPUS} hn=${HARD_NEG_CITELINK:-none} =="
  pids=()
  for ((i=0; i<NUM_GPUS; i++)); do
    gpu="${GPU_ARR[$i]}"
    log="$LOG_DIR/${BUILD_LOG_PREFIX}_shard${i}of${NUM_GPUS}.log"
    echo "  shard ${i}/${NUM_GPUS} on GPU ${gpu} -> $log"
    (
      export CUDA_VISIBLE_DEVICES="$gpu"
      export RWCITE_EMBEDDER_DEVICE=cuda
            "$PY" -m rwcite.cli.train_ce_sent \
        --domain "$DOMAIN" \
        --train-jsonl "$TRAIN_JSONL" \
        --max-queries "$MAX_QUERIES" \
        --window "$WINDOW" \
        --neg-per-list "$NEG_PER_LIST" \
        --max-sents "$MAX_SENTS" \
        --short-cand \
        "${HN_ARGS[@]}" \
        --seed "$SEED" \
        --pairs "$PAIRS" \
        --shard-id "$i" \
        --num-shards "$NUM_GPUS" \
        --rebuild \
        --build-only
    ) >"$log" 2>&1 &
    pids+=($!)
  done
  fail=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then fail=1; fi
  done
  if (( fail )); then
    echo "ERROR: CE shard build failed; see $LOG_DIR/${BUILD_LOG_PREFIX}_shard*.log" >&2
    exit 1
  fi
    "$PY" -m rwcite.cli.train_ce_sent \
    --pairs "$PAIRS" --num-shards "$NUM_GPUS" --merge-shards --build-only
fi

if [[ "$DO_TRAIN" == "1" ]]; then
  echo "== Train CE-sent domain=${DOMAIN} base=${BASE} -> $OUT [${NUM_GPUS} GPU DDP] =="
  export CUDA_VISIBLE_DEVICES="$GPUS"
    export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
  export MASTER_PORT="${MASTER_PORT:-29518}"
  unset RANK WORLD_SIZE LOCAL_RANK LOCAL_WORLD_SIZE GROUP_RANK ROLE_RANK \
    TORCHELASTIC_RUN_ID TORCHELASTIC_RESTART_COUNT ACCELERATE_MIXED_PRECISION || true
  TRAIN_ARGS=(
    --pairs "$PAIRS"
    --base "$BASE"
    --out "$OUT"
    --epochs "$EPOCHS"
    --freeze-epochs "$FREEZE_EPOCHS"
    --list-size "${LIST_SIZE:-64}"
    --lr "${LR:-2e-5}"
    --max-length "$MAX_LENGTH"
    --seed "$SEED"
    --ddp-epoch-mode "${DDP_EPOCH_MODE:-full}"
    --train-only
  )
  if [[ "${LR_SCALE_DDP:-1}" == "1" ]]; then
    TRAIN_ARGS+=(--lr-scale-ddp)
  else
    TRAIN_ARGS+=(--no-lr-scale-ddp)
  fi
  # </dev/null: never let the trainer consume the parent shell's stdin/script stream.
  if [[ "$NUM_GPUS" -gt 1 ]]; then
    "$PY" -m torch.distributed.run \
      --standalone \
      --nproc_per_node="$NUM_GPUS" \
      -m rwcite.cli.train_ce_sent \
      "${TRAIN_ARGS[@]}" \
      </dev/null 2>&1 | tee "$LOG_DIR/$TRAIN_LOG"
  else
    export CUDA_VISIBLE_DEVICES="${GPU_ARR[0]}"
    "$PY" -m rwcite.cli.train_ce_sent \
      "${TRAIN_ARGS[@]}" \
      </dev/null 2>&1 | tee "$LOG_DIR/$TRAIN_LOG"
  fi
fi

if [[ "$DO_EVAL" == "1" ]]; then
  export RR_CE_SENT_WINDOW="$WINDOW"
  export RR_CE_SENT_MAX_SENTS="$MAX_SENTS"
  export RR_CE_SENT_MAX_LEN="$MAX_LENGTH"
  export RR_CE_SENT_SHORT=1
  EVAL_EXTRA=()
  if [[ -n "$TEST_JSONL" ]]; then
    EVAL_EXTRA+=(TEST_JSONL="$TEST_JSONL")
  fi

  echo "== Eval pure CE n=${MAX_SAMPLES} domain=${DOMAIN} =="
  env DOMAIN="$DOMAIN" RR_CE_SENT_CITELINK_BLEND=0 \
  RANKER_CE_SENT="$OUT" RANKER_CITELINK="$CITELINK" \
  RANKER_SENT= RANKER_FUSION= RANKER_LTR= \
  TAG="$EVAL_TAG_PURE" MAX_SAMPLES="$MAX_SAMPLES" SEED=42 GPUS="$GPUS" \
  ${TEST_JSONL:+TEST_JSONL="$TEST_JSONL"} \
    bash scripts/run_eval_rr_pool_ranker_4gpu.sh

  if [[ -n "${HARD_NEG_CITELINK}" || "${DO_BLEND_EVAL:-0}" == "1" ]]; then
    echo "== Eval CE⊕citelink blend=0.25 domain=${DOMAIN} =="
    env DOMAIN="$DOMAIN" RR_CE_SENT_CITELINK_BLEND=0.25 \
    RANKER_CE_SENT="$OUT" RANKER_CITELINK="$CITELINK" \
    RANKER_SENT= RANKER_FUSION= RANKER_LTR= \
    TAG="$EVAL_TAG_BLEND" MAX_SAMPLES="$MAX_SAMPLES" SEED=42 GPUS="$GPUS" \
    ${TEST_JSONL:+TEST_JSONL="$TEST_JSONL"} \
      bash scripts/run_eval_rr_pool_ranker_4gpu.sh
  fi
fi

echo "== CE-sent done: domain=$DOMAIN model=$OUT =="
