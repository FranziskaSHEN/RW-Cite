#!/usr/bin/env bash
# Evaluate RR pool ranker on 4 GPUs in parallel, then merge summaries.
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

MAX_SAMPLES="${MAX_SAMPLES:-80}"
SEED="${SEED:-42}"
N="${N:-30}"
TEST_JSONL="${TEST_JSONL:-datasets/reference_recommend/test.jsonl}"
DOMAIN="${DOMAIN:?DOMAIN required}"
RANKER_LTR="${RANKER_LTR:-}"
RANKER_CITELINK="${RANKER_CITELINK:-}"
RANKER_FUSION="${RANKER_FUSION:-}"
RANKER_SENT="${RANKER_SENT:-}"
RANKER_CE_SENT="${RANKER_CE_SENT:-}"
RANKER_MLP="${RANKER_MLP:-}"
OUT_DIR="${OUT_DIR:-datasets/rr_pool_ranker}"
LOG_DIR="${LOG_DIR:-logs/rr_ltr}"
TAG="${TAG:-pool_eval}"

mkdir -p "$OUT_DIR" "$LOG_DIR"

EXTRA=(--test-jsonl "$TEST_JSONL" --domain "$DOMAIN")
if [[ -n "$RANKER_CE_SENT" ]]; then
  EXTRA+=(--ranker-ce-sent "$RANKER_CE_SENT")
fi
if [[ -n "$RANKER_SENT" ]]; then
  EXTRA+=(--ranker-sent "$RANKER_SENT")
fi
if [[ -n "$RANKER_CITELINK" ]]; then
  EXTRA+=(--ranker-citelink "$RANKER_CITELINK")
fi
if [[ -n "$RANKER_FUSION" ]]; then
  EXTRA+=(--ranker-fusion "$RANKER_FUSION")
fi
if [[ -n "$RANKER_LTR" ]]; then
  EXTRA+=(--ranker-ltr "$RANKER_LTR")
fi
if [[ -n "$RANKER_MLP" ]]; then
  EXTRA+=(--ranker-mlp "$RANKER_MLP")
fi

echo "== Pool eval: domain=${DOMAIN} samples=${MAX_SAMPLES} n=${N} test=${TEST_JSONL} tag=${TAG} shards=${NUM_GPUS} GPUS=${GPUS} =="
pids=()
for ((i=0; i<NUM_GPUS; i++)); do
  gpu="${GPU_ARR[$i]}"
  out="$OUT_DIR/${TAG}_shard${i}of${NUM_GPUS}.json"
  log="$LOG_DIR/${TAG}_shard${i}of${NUM_GPUS}.log"
  echo "  shard ${i}/${NUM_GPUS} on GPU ${gpu} -> $out"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    export RWCITE_EMBEDDER_DEVICE=cuda
    # v4b blend defaults if CE sent is set
    if [[ -n "$RANKER_CE_SENT" ]]; then
      export RR_RANKER_CE_SENT_PATH="$(cd "$(dirname "$RANKER_CE_SENT")" && pwd)/$(basename "$RANKER_CE_SENT")"
      export RR_CE_SENT_SHORT="${RR_CE_SENT_SHORT:-1}"
      export RR_CE_SENT_CITELINK_BLEND="${RR_CE_SENT_CITELINK_BLEND:-0.25}"
      export RR_CE_SENT_WINDOW="${RR_CE_SENT_WINDOW:-400}"
      export RR_CE_SENT_MAX_SENTS="${RR_CE_SENT_MAX_SENTS:-3}"
      export RR_CE_SENT_MAX_LEN="${RR_CE_SENT_MAX_LEN:-256}"
      # Respect caller (env_domain_rr / ablation): default on only if unset
      export RR_RANKER_FULL_EMB="${RR_RANKER_FULL_EMB:-1}"
      export RR_RANKER_WITH_EMB="${RR_RANKER_WITH_EMB:-1}"
      if [[ -n "$RANKER_CITELINK" ]]; then
        export RR_RANKER_CITELINK_PATH="$(cd "$(dirname "$RANKER_CITELINK")" && pwd)/$(basename "$RANKER_CITELINK")"
      fi
    fi
    "$PY" -m rwcite.cli.eval_pool_ranker \
      --max-samples "$MAX_SAMPLES" \
      --seed "$SEED" \
      --n "$N" \
      --shard-id "$i" \
      --num-shards "$NUM_GPUS" \
      --tag "$TAG" \
      --out-dir "$OUT_DIR" \
      --out "$out" \
      "${EXTRA[@]}"
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
  echo "ERROR: one or more eval shards failed; see $LOG_DIR/eval_shard*.log" >&2
  exit 1
fi

MERGED="$OUT_DIR/${TAG}_merged.json"
"$PY" -m rwcite.cli.eval_pool_ranker \
  --merge-shards \
  --num-shards "$NUM_GPUS" \
  --out "$MERGED" \
  --tag "$TAG" \
  --out-dir "$OUT_DIR"

echo "== Merged → $MERGED =="
cat "$MERGED" | "$PY" -c 'import json,sys; d=json.load(sys.stdin); print(json.dumps(d.get("summary",d), indent=2))'
