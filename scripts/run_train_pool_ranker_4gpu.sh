#!/usr/bin/env bash
# Dump structural-shortlist training features in parallel, one shard per GPU.
#
# Usage:
#   DOMAIN=ewm_rw OUT_DIR=... NUM_SHARDS=8 SHARDS=0,1,2,3 GPUS=0,1,2,3 \
#     scripts/run_train_pool_ranker_4gpu.sh
#
# SHARDS may be a subset of 0..NUM_SHARDS-1 so shards can be launched in waves
# while other jobs still hold some GPUs.
set -euo pipefail
ROOT="${RWCITE_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$ROOT"
export RWCITE_ROOT="$ROOT"

PY="${RWCITE_PYTHON:-${PY:-python3}}"
if [[ -x "${ROOT}/.venv/bin/python" && -z "${RWCITE_PYTHON:-}" && "${PY}" == "python3" ]]; then
  PY="${ROOT}/.venv/bin/python"
fi

DOMAIN="${DOMAIN:?DOMAIN required}"
OUT_DIR="${OUT_DIR:?OUT_DIR required}"
NUM_SHARDS="${NUM_SHARDS:-8}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
SHARDS="${SHARDS:-}"
MAX_SOURCES="${MAX_SOURCES:-0}"
TRAIN_JSONL="${TRAIN_JSONL:-}"
WITH_EMB="${WITH_EMB:-1}"
LOG_DIR="${LOG_DIR:-logs/train_pool_ranker_${DOMAIN}}"

IFS=',' read -r -a GPU_ARR <<< "$GPUS"
if [[ -z "$SHARDS" ]]; then
  SHARD_ARR=()
  local_i=0
  for ((local_i=0; local_i<NUM_SHARDS; local_i++)); do SHARD_ARR+=("$local_i"); done
else
  IFS=',' read -r -a SHARD_ARR <<< "$SHARDS"
fi
if (( ${#SHARD_ARR[@]} > ${#GPU_ARR[@]} )); then
  echo "ERROR: ${#SHARD_ARR[@]} shards but only ${#GPU_ARR[@]} GPUs" >&2
  exit 1
fi

mkdir -p "$OUT_DIR" "$LOG_DIR"
echo "== dump features: domain=$DOMAIN shards=${SHARDS:-all}/$NUM_SHARDS gpus=$GPUS out=$OUT_DIR =="

pids=()
for idx in "${!SHARD_ARR[@]}"; do
  shard="${SHARD_ARR[$idx]}"
  gpu="${GPU_ARR[$idx]}"
  log="$LOG_DIR/dump_shard${shard}of${NUM_SHARDS}.log"
  echo "  shard ${shard}/${NUM_SHARDS} on GPU ${gpu} -> $log"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    export RWCITE_EMBEDDER_DEVICE=cuda
    emb_flag=()
    if [[ "$WITH_EMB" == "1" ]]; then
      emb_flag=(--with-emb)
    fi
    "$PY" -m rwcite.cli.train_pool_ranker \
      --stage dump \
      --domain "$DOMAIN" \
      --out-dir "$OUT_DIR" \
      --max-sources "$MAX_SOURCES" \
      --shard-id "$shard" \
      --num-shards "$NUM_SHARDS" \
      "${emb_flag[@]}" \
      ${TRAIN_JSONL:+--train-jsonl "$TRAIN_JSONL"}
  ) >"$log" 2>&1 &
  pids+=($!)
done

fail=0
for pid in "${pids[@]}"; do
  wait "$pid" || fail=1
done
if (( fail )); then
  echo "ERROR: one or more dump shards failed; see $LOG_DIR" >&2
  exit 1
fi
echo "== dump done -> $OUT_DIR =="
