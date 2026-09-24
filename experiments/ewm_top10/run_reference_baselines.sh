#!/usr/bin/env bash
# Complete the deterministic reference baselines on the frozen development split.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PY="${PY:-$ROOT/.venv/bin/python}"
BASE_TAG="${BASE_TAG:-scibert_ntx_hlm_dev_v1}"
EVAL_PROTOCOL="${EVAL_PROTOCOL:-development}"
BASE_ROOT="${BASE_ROOT:-$ROOT/outputs/ewm_top10/$BASE_TAG}"
MODEL_ROOT="${MODEL_ROOT:-$ROOT/models/base}"
SCIBERT="${SCIBERT:-$MODEL_ROOT/scibert_scivocab_uncased}"
BGE="${BGE:-$MODEL_ROOT/bge-large-en-v1.5}"
DENSE_BATCH="${DENSE_BATCH:-32}"
GPU_BGE="${GPU_BGE:-4}"
GPU_SCIBERT="${GPU_SCIBERT:-5}"

DATA_DIR="$BASE_ROOT/data"
PREDICTION_DIR="$BASE_ROOT/predictions"
WINDOW_DIR="$BASE_ROOT/windows"
METRIC_DIR="$BASE_ROOT/metrics_reference"
LOG_DIR="$BASE_ROOT/worker_logs_reference"
BOOTSTRAP_DIR="$BASE_ROOT/bootstrap_reference"
[[ "$EVAL_PROTOCOL" == "development" || "$EVAL_PROTOCOL" == "retrospective381" ]] || {
  echo "ERROR: EVAL_PROTOCOL must be development or retrospective381" >&2
  exit 2
}
if [[ "$EVAL_PROTOCOL" == "development" ]]; then
  EVAL="$DATA_DIR/dev.jsonl"
  EXPECTED_COUNT_KEY="n_dev"
else
  EVAL="$DATA_DIR/test.jsonl"
  EXPECTED_COUNT_KEY="n_test"
fi
CORPUS="$DATA_DIR/corpus.jsonl"
MANIFEST="$DATA_DIR/manifest.json"
WINDOW400="$WINDOW_DIR/neutral_top400.jsonl"

for path in "$PY" "$EVAL" "$CORPUS" "$MANIFEST" "$WINDOW400" \
  "$SCIBERT/config.json" "$BGE/config.json"; do
  [[ -s "$path" ]] || { echo "ERROR: required input is missing or empty: $path" >&2; exit 2; }
done

# Consume an already prepared comparison contract and never regenerate its
# partitions silently. Retrospective mode explicitly treats the former test
# partition as open development data; it is not a confirmatory test run.
"$PY" - "$MANIFEST" "$EVAL" "$EXPECTED_COUNT_KEY" "$EVAL_PROTOCOL" <<'PY'
import json
import sys

manifest = json.load(open(sys.argv[1], encoding="utf-8"))
rows = [
    json.loads(line)
    for line in open(sys.argv[2], encoding="utf-8")
    if line.strip()
]
queries = {row["query_id"] for row in rows}
expected = manifest.get(sys.argv[3])
if len(rows) != len(queries) or len(queries) != expected:
    raise SystemExit(
        f"evaluation contract mismatch: rows={len(rows)} unique={len(queries)} expected={expected}"
    )
if manifest.get("split_protocol") != "strict_month_boundary_train_dev":
    raise SystemExit(f"unexpected split protocol: {manifest.get('split_protocol')}")
test_cutoff = manifest.get("test_cutoff_exclusive_for_training")
if not test_cutoff:
    raise SystemExit("manifest predates the strict official-test cutoff; regenerate the comparison contract")
if sys.argv[4] == "development":
    if any(not row.get("published_at") or row["published_at"] >= test_cutoff for row in rows):
        raise SystemExit(f"development query is not strictly earlier than test cutoff {test_cutoff}")
else:
    if any(not row.get("published_at") or row["published_at"] < test_cutoff for row in rows):
        raise SystemExit(f"retrospective query predates declared cutoff {test_cutoff}")
print(
    "evaluation contract ready:",
    f"protocol={sys.argv[4]}",
    f"queries={len(queries)}",
    f"dev_from={manifest.get('dev_cutoff_inclusive')}",
    f"test_from={manifest.get('test_cutoff_exclusive_for_training')}",
)
PY

mkdir -p "$PREDICTION_DIR" "$METRIC_DIR" "$LOG_DIR" "$BOOTSTRAP_DIR"
touch "$LOG_DIR/bm25.log" "$LOG_DIR/bge_pretrained_gpu${GPU_BGE}.log" \
  "$LOG_DIR/scibert_pretrained_gpu${GPU_SCIBERT}.log"

run_bm25() {
  if [[ ! -s "$PREDICTION_DIR/bm25_top1000.jsonl" ]]; then
    "$PY" -m experiments.ewm_top10.rank_bm25 \
      --test "$EVAL" --corpus "$CORPUS" --top-k 1000 \
      --out "$PREDICTION_DIR/bm25_top1000.jsonl"
  fi
}

run_bge() {
  if [[ ! -s "$PREDICTION_DIR/bge_pretrained_top1000.jsonl" ]]; then
    CUDA_VISIBLE_DEVICES="$GPU_BGE" "$PY" -m experiments.ewm_top10.rank_dense \
      --method bge_pretrained --model "$BGE" --test "$EVAL" \
      --corpus "$CORPUS" --batch-size "$DENSE_BATCH" --top-k 1000 \
      --out "$PREDICTION_DIR/bge_pretrained_top1000.jsonl"
  fi
}

run_scibert() {
  if [[ ! -s "$PREDICTION_DIR/scibert_pretrained_top1000.jsonl" ]]; then
    CUDA_VISIBLE_DEVICES="$GPU_SCIBERT" "$PY" -m experiments.ewm_top10.rank_dense \
      --method scibert_pretrained --model "$SCIBERT" --test "$EVAL" \
      --corpus "$CORPUS" --batch-size "$DENSE_BATCH" --top-k 1000 \
      --out "$PREDICTION_DIR/scibert_pretrained_top1000.jsonl"
  fi
  if [[ ! -s "$PREDICTION_DIR/scibert_pretrained_fixed400.jsonl" ]]; then
    CUDA_VISIBLE_DEVICES="$GPU_SCIBERT" "$PY" -m experiments.ewm_top10.rerank_dense_fixed \
      --windows "$WINDOW400" --test "$EVAL" --corpus "$CORPUS" \
      --model "$SCIBERT" --method scibert_pretrained_fixed400 \
      --out "$PREDICTION_DIR/scibert_pretrained_fixed400.jsonl"
  fi
}

echo "== Deterministic reference baselines =="
echo "evaluation=$EVAL protocol=$EVAL_PROTOCOL"
echo "BM25 runs on CPU; BGE uses physical GPU $GPU_BGE; SciBERT uses physical GPU $GPU_SCIBERT."
echo "Existing SciBERT-NTX and HLM checkpoints are not trained or modified."

(run_bm25) >"$LOG_DIR/bm25.log" 2>&1 & p_bm25=$!
(run_bge) >"$LOG_DIR/bge_pretrained_gpu${GPU_BGE}.log" 2>&1 & p_bge=$!
(run_scibert) >"$LOG_DIR/scibert_pretrained_gpu${GPU_SCIBERT}.log" 2>&1 & p_scibert=$!

failed=0
for pid in "$p_bm25" "$p_bge" "$p_scibert"; do
  wait "$pid" || failed=1
done
if [[ "$failed" == "1" ]]; then
  echo "ERROR: one or more reference-baseline workers failed; inspect $LOG_DIR" >&2
  exit 1
fi

# The neutral Top-400 file is evaluated in its original BGE order as the
# control for the fixed-window dense-reranking experiment.
"$PY" -m experiments.ewm_top10.evaluate \
  --test "$EVAL" --corpus "$CORPUS" \
  --predictions \
    "$PREDICTION_DIR/bm25_top1000.jsonl" \
    "$PREDICTION_DIR/bge_pretrained_top1000.jsonl" \
    "$PREDICTION_DIR/scibert_pretrained_top1000.jsonl" \
    "$PREDICTION_DIR/scibert_pretrained_fixed400.jsonl" \
    "$WINDOW400" \
  --out-dir "$METRIC_DIR"

# Quantify the gain from NT-Xent training over the otherwise matched frozen
# SciBERT encoder. Each trained seed is paired with the same deterministic base.
BASE_PER_QUERY="$METRIC_DIR/scibert_pretrained_top1000_per_query.jsonl"
if [[ -s "$BASE_ROOT/metrics/scibert_ntx_seed42_top1000_per_query.jsonl" ]]; then
  for seed in 42 43 44; do
    "$PY" -m experiments.ewm_top10.bootstrap \
      --baseline "$BASE_PER_QUERY" \
      --system "$BASE_ROOT/metrics/scibert_ntx_seed${seed}_top1000_per_query.jsonl" \
      --metric hits_at_10 --samples 10000 --seed "$seed" \
      --out "$BOOTSTRAP_DIR/ntx_minus_pretrained_scibert_seed${seed}.json"
  done
else
  echo "WARN: existing SciBERT-NTX per-query metrics not found; paired tests skipped." >&2
fi

echo "== Complete =="
echo "Reference summaries: $METRIC_DIR/*_summary.json"
echo "Combined evaluator output: $METRIC_DIR/summary.json"
echo "NT-Xent paired tests: $BOOTSTRAP_DIR"
