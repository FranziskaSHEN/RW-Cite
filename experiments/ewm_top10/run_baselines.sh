#!/usr/bin/env bash
# Multi-GPU run for SciBERT-NTX and HLM on a declared evaluation partition.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PY="${PY:-$ROOT/.venv/bin/python}"
TAG="${TAG:-citation_baselines}"
EVAL_PROTOCOL="${EVAL_PROTOCOL:-development}"
OUT_ROOT="${OUT_ROOT:-$ROOT/outputs/ewm_top10/$TAG}"
DATA_DIR="$OUT_ROOT/data"
CHECKPOINT_DIR="$OUT_ROOT/checkpoints"
PREDICTION_DIR="$OUT_ROOT/predictions"
WINDOW_DIR="$OUT_ROOT/windows"
METRIC_DIR="$OUT_ROOT/metrics"
LOG_DIR="$OUT_ROOT/worker_logs"

DATA_ROOT="${BENCHMARK_DATA_ROOT:?Set BENCHMARK_DATA_ROOT to the unpacked EWM benchmark assets}"
EWM_GRAPH="${EWM_GRAPH:?Set EWM_GRAPH to the exact EWM graph file}"
MODEL_ROOT="${MODEL_ROOT:-$ROOT/models/base}"
SCIBERT="${SCIBERT:-$MODEL_ROOT/scibert_scivocab_uncased}"
GTE="${GTE:-$MODEL_ROOT/gte-base}"
BGE="${BGE:-$MODEL_ROOT/bge-large-en-v1.5}"
GPUS=(4 5 6 7)
SEEDS=(42 43 44)
DENSE_BATCH="${DENSE_BATCH:-32}"

SOURCE_ROOT="${EWM_SOURCE_ROOT:-$DATA_ROOT/embodied_world_model_retrieval}"
SOURCE_TRAIN="${SOURCE_TRAIN:-$SOURCE_ROOT/data/reference_recommend/train.jsonl}"
SOURCE_TEST="${SOURCE_TEST:-$SOURCE_ROOT/data/reference_recommend/test.jsonl}"

for path in "$PY" "$SOURCE_TRAIN" "$SOURCE_TEST" "$EWM_GRAPH" \
  "$SCIBERT/config.json" "$GTE/config.json" "$BGE/config.json"; do
  [[ -e "$path" ]] || { echo "ERROR: required input not found: $path" >&2; exit 2; }
done

mkdir -p "$DATA_DIR" "$CHECKPOINT_DIR" "$PREDICTION_DIR" \
  "$WINDOW_DIR" "$METRIC_DIR" "$LOG_DIR" "$OUT_ROOT/bootstrap"
for log in \
  scibert_ntx_seed42_gpu4.log scibert_ntx_seed43_gpu5.log \
  scibert_ntx_seed44_gpu6.log hlm_seed42_gpu7.log \
  hlm_seed43_gpu4.log hlm_seed44_gpu5.log; do
  touch "$LOG_DIR/$log"
done

[[ "$EVAL_PROTOCOL" == "development" || "$EVAL_PROTOCOL" == "retrospective381" ]] || {
  echo "ERROR: EVAL_PROTOCOL must be development or retrospective381" >&2
  exit 2
}

echo "== SciBERT-NTX and HLM baseline run =="
echo "root=$ROOT"
echo "output=$OUT_ROOT"
echo "evaluation_protocol=$EVAL_PROTOCOL"
echo "physical_gpus=${GPUS[*]}"
echo "No RW-Cite checkpoint will be loaded or evaluated."

if [[ ! -s "$DATA_DIR/manifest.json" || "${FORCE_PREP:-0}" == "1" ]]; then
  "$PY" -m experiments.ewm_top10.prepare_data \
    --train-jsonl "$SOURCE_TRAIN" \
    --test-jsonl "$SOURCE_TEST" \
    --graph "$EWM_GRAPH" \
    --out "$DATA_DIR"
fi
for required in train.jsonl dev.jsonl pretest_train.jsonl test.jsonl corpus.jsonl manifest.json; do
  [[ -s "$DATA_DIR/$required" ]] || { echo "ERROR: missing $DATA_DIR/$required" >&2; exit 2; }
done

if [[ "$EVAL_PROTOCOL" == "development" ]]; then
  TRAIN_FILE="$DATA_DIR/train.jsonl"
  EVAL_FILE="$DATA_DIR/dev.jsonl"
  OTHER_HELDOUT="$DATA_DIR/test.jsonl"
  EXPECTED_COUNT_KEY="n_dev"
else
  # The former official test is intentionally reclassified as an open,
  # retrospective benchmark. Training still stops strictly before its first
  # month; a later, genuinely untouched partition must provide final testing.
  TRAIN_FILE="$DATA_DIR/pretest_train.jsonl"
  EVAL_FILE="$DATA_DIR/test.jsonl"
  OTHER_HELDOUT="$DATA_DIR/test.jsonl"
  EXPECTED_COUNT_KEY="n_test"
fi
CORPUS_FILE="$DATA_DIR/corpus.jsonl"
HLM_LABELS="$DATA_DIR/hlm_train_retrospective_${EVAL_PROTOCOL}.jsonl"

"$PY" - "$DATA_DIR/manifest.json" "$EVAL_FILE" "$EXPECTED_COUNT_KEY" <<'PY'
import json
import sys

manifest = json.load(open(sys.argv[1], encoding="utf-8"))
rows = [json.loads(line) for line in open(sys.argv[2], encoding="utf-8") if line.strip()]
ids = {row["query_id"] for row in rows}
expected = manifest.get(sys.argv[3])
if len(rows) != len(ids) or len(ids) != expected:
    raise SystemExit(
        f"evaluation contract mismatch: rows={len(rows)} unique={len(ids)} expected={expected}"
    )
print(f"evaluation contract ready: protocol={sys.argv[3]} queries={len(ids)}")
PY

# HLM-Cite's published retriever uses future co-citation evidence relative to
# individual training queries. It is retained as a paper-faithful supplementary
# reproduction, while dev/test query IDs and their edges are excluded.
if [[ ! -s "$HLM_LABELS" ]]; then
  "$PY" -m experiments.ewm_top10.prepare_hlm_labels \
    --train "$TRAIN_FILE" \
    --dev "$OTHER_HELDOUT" \
    --test "$EVAL_FILE" \
    --corpus "$CORPUS_FILE" \
    --graph "$EWM_GRAPH" \
    --allow-retrospective-labels \
    --out "$HLM_LABELS"
fi

# A neutral retriever supplies the identical, temporally legal Top-400 window
# used only by the controlled dense-reranking experiment.
if [[ ! -s "$PREDICTION_DIR/bge_neutral_top400.jsonl" ]]; then
  echo "== Building neutral Top-400 on physical GPU 7 =="
  CUDA_VISIBLE_DEVICES=7 "$PY" -m experiments.ewm_top10.rank_dense \
    --method bge_neutral \
    --model "$BGE" \
    --test "$EVAL_FILE" \
    --corpus "$CORPUS_FILE" \
    --batch-size "$DENSE_BATCH" \
    --top-k 400 \
    --out "$PREDICTION_DIR/bge_neutral_top400.jsonl"
fi
if [[ ! -s "$WINDOW_DIR/neutral_top400.jsonl" ]]; then
  "$PY" -m experiments.ewm_top10.prepare_fixed_windows \
    --retrieval "$PREDICTION_DIR/bge_neutral_top400.jsonl" \
    --test "$EVAL_FILE" --corpus "$CORPUS_FILE" --graph "$EWM_GRAPH" \
    --window 400 --out "$WINDOW_DIR/neutral_top400.jsonl"
fi
run_ntx() {
  local gpu="$1" seed="$2"
  local model="$CHECKPOINT_DIR/scibert_ntx_seed${seed}"
  echo "[SciBERT-NTX seed=$seed] physical GPU $gpu"
  if [[ ! -s "$model/config.json" ]]; then
    CUDA_VISIBLE_DEVICES="$gpu" "$PY" -m experiments.ewm_top10.train_scibert_ntx \
      --train "$TRAIN_FILE" --corpus "$CORPUS_FILE" --base "$SCIBERT" \
      --out "$CHECKPOINT_DIR/scibert_ntx" --seed "$seed"
  fi
  if [[ ! -s "$PREDICTION_DIR/scibert_ntx_seed${seed}_top1000.jsonl" ]]; then
    CUDA_VISIBLE_DEVICES="$gpu" "$PY" -m experiments.ewm_top10.rank_dense \
      --method scibert_ntx --model "$model" --test "$EVAL_FILE" \
      --corpus "$CORPUS_FILE" --batch-size "$DENSE_BATCH" --top-k 1000 \
      --out "$PREDICTION_DIR/scibert_ntx_seed${seed}_top1000.jsonl"
  fi
  if [[ ! -s "$PREDICTION_DIR/scibert_ntx_seed${seed}_fixed400.jsonl" ]]; then
    CUDA_VISIBLE_DEVICES="$gpu" "$PY" -m experiments.ewm_top10.rerank_dense_fixed \
      --windows "$WINDOW_DIR/neutral_top400.jsonl" --test "$EVAL_FILE" \
      --corpus "$CORPUS_FILE" --model "$model" \
      --method scibert_ntx_fixed400 \
      --out "$PREDICTION_DIR/scibert_ntx_seed${seed}_fixed400.jsonl"
  fi
}

run_hlm_retriever() {
  local gpu="$1" seed="$2"
  local stage1="$CHECKPOINT_DIR/hlm_stage1_seed${seed}"
  local model="$CHECKPOINT_DIR/hlm_stage2_seed${seed}"
  echo "[HLM retriever seed=$seed] physical GPU $gpu"
  if [[ ! -s "$stage1/config.json" ]]; then
    CUDA_VISIBLE_DEVICES="$gpu" "$PY" -m experiments.ewm_top10.train_hlm_retriever \
      --stage 1 --train "$HLM_LABELS" --corpus "$CORPUS_FILE" --base "$GTE" \
      --stage1 "$CHECKPOINT_DIR/hlm_stage1" --seed "$seed"
  fi
  if [[ ! -s "$model/config.json" ]]; then
    CUDA_VISIBLE_DEVICES="$gpu" "$PY" -m experiments.ewm_top10.train_hlm_retriever \
      --stage 2 --train "$HLM_LABELS" --corpus "$CORPUS_FILE" \
      --stage1 "$CHECKPOINT_DIR/hlm_stage1" --out "$CHECKPOINT_DIR/hlm" \
      --seed "$seed" --grad-accum 96
  fi
  if [[ ! -s "$PREDICTION_DIR/hlm_retriever_seed${seed}_top1000.jsonl" ]]; then
    CUDA_VISIBLE_DEVICES="$gpu" "$PY" -m experiments.ewm_top10.rank_dense \
      --method hlm_retriever_retrospective --model "$model" \
      --test "$EVAL_FILE" --corpus "$CORPUS_FILE" --batch-size "$DENSE_BATCH" \
      --max-length 512 --top-k 1000 \
      --out "$PREDICTION_DIR/hlm_retriever_seed${seed}_top1000.jsonl"
  fi
  if [[ ! -s "$PREDICTION_DIR/hlm_retriever_seed${seed}_fixed400.jsonl" ]]; then
    CUDA_VISIBLE_DEVICES="$gpu" "$PY" -m experiments.ewm_top10.rerank_dense_fixed \
      --windows "$WINDOW_DIR/neutral_top400.jsonl" --test "$EVAL_FILE" \
      --corpus "$CORPUS_FILE" --model "$model" --max-length 512 \
      --method hlm_retriever_retrospective_fixed400 \
      --out "$PREDICTION_DIR/hlm_retriever_seed${seed}_fixed400.jsonl"
  fi
}

# The first wave occupies all four GPUs without placing two jobs on one device.
(run_ntx 4 42) >"$LOG_DIR/scibert_ntx_seed42_gpu4.log" 2>&1 & p4=$!
(run_ntx 5 43) >"$LOG_DIR/scibert_ntx_seed43_gpu5.log" 2>&1 & p5=$!
(run_ntx 6 44) >"$LOG_DIR/scibert_ntx_seed44_gpu6.log" 2>&1 & p6=$!
(run_hlm_retriever 7 42) >"$LOG_DIR/hlm_seed42_gpu7.log" 2>&1 & p7=$!

failed=0
for pid in "$p4" "$p5" "$p6" "$p7"; do
  wait "$pid" || failed=1
done
if [[ "$failed" == "1" ]]; then
  echo "ERROR: one or more baseline workers failed; inspect $LOG_DIR" >&2
  exit 1
fi

# The second wave uses the first two freed devices for the remaining HLM seeds.
(run_hlm_retriever 4 43) >"$LOG_DIR/hlm_seed43_gpu4.log" 2>&1 & p4=$!
(run_hlm_retriever 5 44) >"$LOG_DIR/hlm_seed44_gpu5.log" 2>&1 & p5=$!
failed=0
for pid in "$p4" "$p5"; do
  wait "$pid" || failed=1
done
if [[ "$failed" == "1" ]]; then
  echo "ERROR: one or more HLM workers failed; inspect $LOG_DIR" >&2
  exit 1
fi

# Freeze the candidate window produced by the predeclared first-stage HLM
# retriever. The downstream Analyzer--Decider must consume this file and will
# reject any window whose recorded method is not HLM retrieval.
HLM_LLM_RETRIEVER_SEED="${HLM_LLM_RETRIEVER_SEED:-42}"
[[ "$HLM_LLM_RETRIEVER_SEED" =~ ^(42|43|44)$ ]] || {
  echo "ERROR: HLM_LLM_RETRIEVER_SEED must be one of 42, 43, or 44" >&2
  exit 2
}
HLM_LLM_WINDOW="$WINDOW_DIR/hlm_retriever_seed${HLM_LLM_RETRIEVER_SEED}_top30.jsonl"
if [[ ! -s "$HLM_LLM_WINDOW" ]]; then
  "$PY" -m experiments.ewm_top10.prepare_fixed_windows \
    --retrieval "$PREDICTION_DIR/hlm_retriever_seed${HLM_LLM_RETRIEVER_SEED}_top1000.jsonl" \
    --test "$EVAL_FILE" --corpus "$CORPUS_FILE" --graph "$EWM_GRAPH" \
    --window 30 \
    --expected-method-prefix hlm_retriever \
    --output-method "hlm_retriever_seed${HLM_LLM_RETRIEVER_SEED}_temporal_top30" \
    --out "$HLM_LLM_WINDOW"
fi

all_predictions=()
for seed in "${SEEDS[@]}"; do
  all_predictions+=(
    "$PREDICTION_DIR/scibert_ntx_seed${seed}_top1000.jsonl"
    "$PREDICTION_DIR/scibert_ntx_seed${seed}_fixed400.jsonl"
    "$PREDICTION_DIR/hlm_retriever_seed${seed}_top1000.jsonl"
    "$PREDICTION_DIR/hlm_retriever_seed${seed}_fixed400.jsonl"
  )
done
"$PY" -m experiments.ewm_top10.evaluate \
  --test "$EVAL_FILE" --corpus "$CORPUS_FILE" \
  --predictions "${all_predictions[@]}" --out-dir "$METRIC_DIR"

for family in scibert_ntx hlm_retriever; do
  for suffix in top1000 fixed400; do
    summaries=()
    for seed in "${SEEDS[@]}"; do
      summaries+=("$METRIC_DIR/${family}_seed${seed}_${suffix}_summary.json")
    done
    "$PY" -m experiments.ewm_top10.aggregate_seeds \
      --summaries "${summaries[@]}" \
      --out "$METRIC_DIR/${family}_${suffix}_three_seed.json"
  done
done

for seed in "${SEEDS[@]}"; do
  for suffix in top1000 fixed400; do
    "$PY" -m experiments.ewm_top10.bootstrap \
      --baseline "$METRIC_DIR/scibert_ntx_seed${seed}_${suffix}_per_query.jsonl" \
      --system "$METRIC_DIR/hlm_retriever_seed${seed}_${suffix}_per_query.jsonl" \
      --metric hits_at_10 --samples 10000 --seed "$seed" \
      --out "$OUT_ROOT/bootstrap/hlm_minus_ntx_seed${seed}_${suffix}.json"
  done
done

# Optional complete HLM-style comparison. The trained HLM retriever supplies
# the candidate Top-30; the Analyzer--Decider performs the final selection.
if [[ "${RUN_HLM_LLM:-0}" == "1" ]]; then
  : "${HLM_ENDPOINT:?Set HLM_ENDPOINT when RUN_HLM_LLM=1}"
  : "${HLM_API_KEY:?Set HLM_API_KEY when RUN_HLM_LLM=1}"
  : "${HLM_MODEL:?Set HLM_MODEL when RUN_HLM_LLM=1}"
  : "${HLM_ONE_SHOT:?Set HLM_ONE_SHOT to a reviewed one-shot example file}"
  HLM_LLM_OUTPUT="$PREDICTION_DIR/hlm_pipeline_top30.jsonl"
  "$PY" -m experiments.ewm_top10.rerank_hlm \
    --retrieval "$HLM_LLM_WINDOW" \
    --test "$EVAL_FILE" --corpus "$CORPUS_FILE" \
    --model "$HLM_MODEL" --endpoint "$HLM_ENDPOINT" \
    --api-key-env HLM_API_KEY --one-shot "$HLM_ONE_SHOT" \
    --require-frozen-evidence \
    --require-retrieval-method-prefix hlm_retriever \
    --expected-window-size 30 --replicate 1 \
    --cache-dir "$OUT_ROOT/cache/hlm" \
    --out "$HLM_LLM_OUTPUT"
  "$PY" -m experiments.ewm_top10.evaluate \
    --test "$EVAL_FILE" --corpus "$CORPUS_FILE" \
    --predictions "$HLM_LLM_OUTPUT" \
    --out-dir "$METRIC_DIR"
else
  echo "HLM LLM calls skipped (RUN_HLM_LLM=0)."
  echo "Prepared HLM retriever input: $HLM_LLM_WINDOW"
fi

echo "== Complete =="
echo "Three-seed summaries: $METRIC_DIR/*_three_seed.json"
echo "Per-seed paired tests: $OUT_ROOT/bootstrap"
echo "HLM temporal caveat: ${HLM_LABELS%.jsonl}.meta.json"
