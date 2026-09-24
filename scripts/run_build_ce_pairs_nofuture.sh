#!/usr/bin/env bash
# Build cross-encoder pairs on the split-masked graph for an enabled domain.
# Writes $CE_PAIRS (…/ranker/pool/ce_base_pairs_nofuture.jsonl).
#
#   DOMAIN=ewm bash scripts/run_build_ce_pairs_nofuture.sh
#   DOMAIN=sqc GPUS=2,3 bash scripts/run_build_ce_pairs_nofuture.sh
#
# Uses RR_GEXF_OVERRIDE from lib_domain_env (o2.nofuture). Does NOT train CE.
set -euo pipefail
ROOT="${RWCITE_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$ROOT"
PY="${RWCITE_PYTHON:-python3}"
if [[ -x "${ROOT}/.venv/bin/python" && -z "${RWCITE_PYTHON:-}" && "${PY}" == "python3" ]]; then
  PY="${ROOT}/.venv/bin/python"
fi
export RWCITE_ROOT="$ROOT"
export PYTHONUNBUFFERED=1
# shellcheck source=scripts/lib_domain_env.sh
source "$ROOT/scripts/lib_domain_env.sh"
rwcite_load_domain_env

export RR_STRUCT_MASK_QUERY_OUT="${RR_STRUCT_MASK_QUERY_OUT:-1}"
export RR_STRUCT_MASK_FUTURE_OUT="${RR_STRUCT_MASK_FUTURE_OUT:-1}"
export RR_STRUCT_COARSE="${RR_STRUCT_COARSE:-linear}"

PAIRS="${PAIRS:-$CE_PAIRS}"
TRAIN_JSONL="${TRAIN_JSONL:-$JSONL_DIR/train.jsonl}"
[[ -f "$TRAIN_JSONL" ]] || TRAIN_JSONL="$ROOT/$JSONL_DIR/train.jsonl"
[[ -f "$TRAIN_JSONL" ]] || { echo "missing train jsonl $TRAIN_JSONL" >&2; exit 1; }

# Prefer pack GPUs when set; else single CPU-ish / one visible GPU for shards.
GPUS="${GPUS:-${PACK_GPUS:-${CUDA_VISIBLE_DEVICES:-0}}}"
IFS=',' read -r -a GPU_ARR <<< "$GPUS"
NUM_GPUS="${#GPU_ARR[@]}"

MAX_QUERIES="${MAX_QUERIES:-0}"
WINDOW="${WINDOW:-400}"
NEG_PER_LIST="${NEG_PER_LIST:-64}"
MAX_SENTS="${MAX_SENTS:-3}"
SEED="${SEED:-0}"
LOG_DIR="${LOG_DIR:-logs/ce_pairs_nofuture_${DOMAIN}}"
mkdir -p "$LOG_DIR" "$(dirname "$PAIRS")"

echo "== CE pairs nofuture DOMAIN=$DOMAIN gexf=$RR_GEXF_OVERRIDE shards=$NUM_GPUS max_q=$MAX_QUERIES → $PAIRS =="

pids=()
for ((i = 0; i < NUM_GPUS; i++)); do
  gpu="${GPU_ARR[$i]}"
  log="$LOG_DIR/build_shard${i}of${NUM_GPUS}.log"
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
[[ "$fail" == "0" ]] || { echo "ERROR: CE pair shard build failed; see $LOG_DIR" >&2; exit 1; }

if (( NUM_GPUS > 1 )); then
  echo "== merge pair shards → $PAIRS =="
  "$PY" -m rwcite.cli.train_ce_sent \
    --pairs "$PAIRS" \
    --num-shards "$NUM_GPUS" \
    --merge-shards \
    --build-only
fi

[[ -f "$PAIRS" ]] || { echo "ERROR: pairs missing after build: $PAIRS" >&2; exit 1; }
echo "OK: CE pairs DOMAIN=$DOMAIN → $PAIRS"
