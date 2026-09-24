#!/usr/bin/env bash
# Supplementary, retrospectively supervised HLM retriever reproduction.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PY="${PY:-python3}"
PHASE="${PHASE:-dev}"
SEEDS="${SEEDS:-42 43 44}"
DATA_DIR="${SHARED_DATA_DIR:-outputs/ewm_top10/temporal_contract/data}"
OUT_ROOT="${OUT_ROOT:-outputs/ewm_top10/hlm_retrospective_${PHASE}}"
GTE="${GTE:-$ROOT/models/base/gte-base}"
GRAPH="${RW_GRAPH:?Set RW_GRAPH to the benchmark graph}"

mkdir -p "$OUT_ROOT/checkpoints" "$OUT_ROOT/predictions" "$OUT_ROOT/data"
EMPTY="$OUT_ROOT/data/empty.jsonl"
: > "$EMPTY"
if [[ "$PHASE" == "dev" ]]; then
  TRAIN="$DATA_DIR/train.jsonl"
  EVAL="$DATA_DIR/dev.jsonl"
  OTHER_HELDOUT="$DATA_DIR/test.jsonl"
elif [[ "$PHASE" == "test" ]]; then
  TRAIN="$DATA_DIR/pretest_train.jsonl"
  EVAL="$DATA_DIR/test.jsonl"
  OTHER_HELDOUT="$EMPTY"
else
  echo "ERROR: PHASE must be dev or test" >&2
  exit 2
fi

LABELS="$OUT_ROOT/data/hlm_train.jsonl"
if [[ ! -s "$LABELS" ]]; then
  "$PY" -m experiments.ewm_top10.prepare_hlm_labels \
    --train "$TRAIN" --dev "$OTHER_HELDOUT" --test "$EVAL" \
    --corpus "$DATA_DIR/corpus.jsonl" --graph "$GRAPH" \
    --allow-retrospective-labels --out "$LABELS"
fi

for seed in $SEEDS; do
  if [[ ! -s "$OUT_ROOT/checkpoints/hlm_stage1_seed${seed}/config.json" ]]; then
    "$PY" -m experiments.ewm_top10.train_hlm_retriever \
      --stage 1 --train "$LABELS" --corpus "$DATA_DIR/corpus.jsonl" \
      --base "$GTE" --stage1 "$OUT_ROOT/checkpoints/hlm_stage1" --seed "$seed"
  fi
  if [[ ! -s "$OUT_ROOT/checkpoints/hlm_stage2_seed${seed}/config.json" ]]; then
    "$PY" -m experiments.ewm_top10.train_hlm_retriever \
      --stage 2 --train "$LABELS" --corpus "$DATA_DIR/corpus.jsonl" \
      --stage1 "$OUT_ROOT/checkpoints/hlm_stage1" \
      --out "$OUT_ROOT/checkpoints/hlm" --seed "$seed" --grad-accum 96
  fi
  if [[ ! -s "$OUT_ROOT/predictions/hlm_retriever_seed${seed}_top1000.jsonl" ]]; then
    "$PY" -m experiments.ewm_top10.rank_dense \
      --method hlm_retriever_retrospective \
      --model "$OUT_ROOT/checkpoints/hlm_stage2_seed${seed}" \
      --test "$EVAL" --corpus "$DATA_DIR/corpus.jsonl" \
      --max-length 512 --top-k 1000 \
      --out "$OUT_ROOT/predictions/hlm_retriever_seed${seed}_top1000.jsonl"
  fi
done

echo "Supplementary HLM retriever reproduction complete."
echo "Its post-query co-citation supervision is recorded in ${LABELS%.jsonl}.meta.json."
