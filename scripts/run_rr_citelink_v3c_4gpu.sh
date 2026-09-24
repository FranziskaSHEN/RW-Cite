#!/usr/bin/env bash
# Build in-window cite-link pairs on 4 GPUs, merge, train MLP, optionally eval.
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

MAX_QUERIES="${MAX_QUERIES:-0}"
WINDOW="${WINDOW:-400}"
NEG_PER_LIST="${NEG_PER_LIST:-64}"
SEED="${SEED:-0}"
CACHE="${CACHE:-datasets/rr_pool_ranker/citelink_v3c_pairs.npz}"
OUT="${OUT:-models/rr-pool-ranker-citelink-v3c/model.npz}"
DOMAIN="${DOMAIN:?DOMAIN required}"
TRAIN_JSONL="${TRAIN_JSONL:-datasets/reference_recommend/train.jsonl}"
LOG_DIR="${LOG_DIR:-logs/rr_ltr}"
REBUILD="${REBUILD:-1}"
DO_TRAIN="${DO_TRAIN:-1}"
DO_EVAL="${DO_EVAL:-1}"
MAX_SAMPLES="${MAX_SAMPLES:-80}"
TEST_JSONL="${TEST_JSONL:-}"
BUILD_LOG_PREFIX="${BUILD_LOG_PREFIX:-v3c_build}"

mkdir -p "$LOG_DIR" "$(dirname "$CACHE")" "$(dirname "$OUT")"

if [[ "$DO_TRAIN" == "1" ]]; then
  REBUILD_FLAG=()
  if [[ "$REBUILD" == "1" ]]; then
    REBUILD_FLAG=(--rebuild)
  fi

  echo "== v3c window-pair build: domain=${DOMAIN} nq=${MAX_QUERIES:-all} window=${WINDOW} shards=${NUM_GPUS} =="
  pids=()
  for ((i=0; i<NUM_GPUS; i++)); do
    gpu="${GPU_ARR[$i]}"
    log="$LOG_DIR/${BUILD_LOG_PREFIX}_shard${i}of${NUM_GPUS}.log"
    echo "  shard ${i}/${NUM_GPUS} on GPU ${gpu} -> $log"
    (
      export CUDA_VISIBLE_DEVICES="$gpu"
      export RWCITE_EMBEDDER_DEVICE=cuda
          "$PY" -m rwcite.cli.train_citelink \
        --domain "$DOMAIN" \
        --train-jsonl "$TRAIN_JSONL" \
        --max-queries "$MAX_QUERIES" \
        --window "$WINDOW" \
        --neg-per-list "$NEG_PER_LIST" \
        --seed "$SEED" \
        --cache "$CACHE" \
        --shard-id "$i" \
        --num-shards "$NUM_GPUS" \
        --build-only \
        "${REBUILD_FLAG[@]}"
    ) >"$log" 2>&1 &
    pids+=($!)
  done

  fail=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      fail=1
    fi
  done
  if (( fail )); then
    echo "ERROR: shard build failed; see $LOG_DIR/${BUILD_LOG_PREFIX}_shard*.log" >&2
    exit 1
  fi

  echo "== Merge + train domain=${DOMAIN} → $OUT =="
  "$PY" -m rwcite.cli.train_citelink \
    --domain "$DOMAIN" \
    --train-jsonl "$TRAIN_JSONL" \
    --cache "$CACHE" \
    --out "$OUT" \
    --window "$WINDOW" \
    --neg-per-list "$NEG_PER_LIST" \
    --max-queries "$MAX_QUERIES" \
    --seed "$SEED" \
    --epochs "${EPOCHS:-8}" \
    --lr "${LR:-1e-3}" \
    --hidden "${HIDDEN:-512}" \
    --device cuda:0 \
    --num-shards "$NUM_GPUS" \
    --merge-shards

  echo "== Done model: $OUT =="
else
  echo "== DO_TRAIN=0 skip pair build + MLP (reuse $OUT) =="
  [[ -f "$OUT" ]] || { echo "missing citelink model $OUT" >&2; exit 1; }
fi

if [[ "$DO_EVAL" == "1" ]]; then
  echo "== Eval n=${MAX_SAMPLES} domain=${DOMAIN} =="
  export RR_CITELINK_WINDOW="$WINDOW"
  export RR_CITELINK_STRUCT_BLEND="${RR_CITELINK_STRUCT_BLEND:-0.25}"
  env DOMAIN="$DOMAIN" \
    RANKER_CITELINK="$OUT" RANKER_LTR= \
    TAG="${TAG:-citelink_v3c_n80}" \
    MAX_SAMPLES="$MAX_SAMPLES" SEED=42 GPUS="$GPUS" \
    OUT_DIR="${OUT_DIR:-datasets/rr_pool_ranker}" \
    ${TEST_JSONL:+TEST_JSONL="$TEST_JSONL"} \
    bash scripts/run_eval_rr_pool_ranker_4gpu.sh
fi
