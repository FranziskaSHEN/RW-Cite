#!/usr/bin/env bash
# Complete HLM-style retriever plus Analyzer--Decider evaluation.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PY="${PY:-$ROOT/.venv/bin/python}"
BASE_TAG="${BASE_TAG:-scibert_ntx_hlm_dev_v1}"
BASE_ROOT="${BASE_ROOT:-$ROOT/outputs/ewm_top10/$BASE_TAG}"
EVAL_PROTOCOL="${EVAL_PROTOCOL:-development}"
MODE="${MODE:-smoke}"
SMOKE_QUERIES="${SMOKE_QUERIES:-3}"
HLM_MODEL="${HLM_MODEL:?Set the frozen API model identifier}"
HLM_RUN_LABEL="${HLM_RUN_LABEL:-$HLM_MODEL}"
HLM_ENDPOINT="${HLM_ENDPOINT:?Set the OpenAI-compatible API endpoint}"
HLM_ONE_SHOT="${HLM_ONE_SHOT:?Set the reviewed training-only one-shot file}"
HLM_API_KEY="${HLM_API_KEY:?Export HLM_API_KEY before launching}"
HLM_INPUT_PRICE_PER_M="${HLM_INPUT_PRICE_PER_M:-0}"
HLM_OUTPUT_PRICE_PER_M="${HLM_OUTPUT_PRICE_PER_M:-0}"
HLM_COST_STATUS="${HLM_COST_STATUS:-priced}"
HLM_DISABLE_THINKING="${HLM_DISABLE_THINKING:-0}"

DATA_DIR="$BASE_ROOT/data"
HLM_RETRIEVER_SEED="${HLM_RETRIEVER_SEED:-42}"
WINDOW="$BASE_ROOT/windows/hlm_retriever_seed${HLM_RETRIEVER_SEED}_top30.jsonl"
SAFE_RUN_LABEL="$(printf '%s' "$HLM_RUN_LABEL" | tr -cs 'A-Za-z0-9._-' '_')"
RUN_ROOT="$BASE_ROOT/hlm_pipeline/$SAFE_RUN_LABEL"
PREDICTION_DIR="$RUN_ROOT/predictions"
METRIC_DIR="$RUN_ROOT/metrics"
CACHE_DIR="$RUN_ROOT/cache"
REPORT_DIR="$RUN_ROOT/reports"
[[ "$EVAL_PROTOCOL" == "development" || "$EVAL_PROTOCOL" == "retrospective381" ]] || {
  echo "ERROR: EVAL_PROTOCOL must be development or retrospective381" >&2
  exit 2
}
if [[ "$EVAL_PROTOCOL" == "development" ]]; then
  EVAL="$DATA_DIR/dev.jsonl"
else
  EVAL="$DATA_DIR/test.jsonl"
fi
CORPUS="$DATA_DIR/corpus.jsonl"

[[ "$HLM_RETRIEVER_SEED" =~ ^(42|43|44)$ ]] || {
  echo "ERROR: HLM_RETRIEVER_SEED must be one of 42, 43, or 44" >&2
  exit 2
}
for path in "$PY" "$EVAL" "$CORPUS" "$HLM_ONE_SHOT"; do
  [[ -s "$path" ]] || { echo "ERROR: required input is missing or empty: $path" >&2; exit 2; }
done
if [[ ! -s "$WINDOW" ]]; then
  retrieval="$BASE_ROOT/predictions/hlm_retriever_seed${HLM_RETRIEVER_SEED}_top1000.jsonl"
  manifest="$DATA_DIR/manifest.json"
  [[ -s "$retrieval" ]] || {
    echo "ERROR: missing trained HLM retriever predictions: $retrieval" >&2
    exit 2
  }
  [[ -s "$manifest" ]] || { echo "ERROR: missing data manifest: $manifest" >&2; exit 2; }
  graph="${EWM_GRAPH:-}"
  if [[ -z "$graph" ]]; then
    graph="$($PY - "$manifest" <<'PY'
import json
import sys
print(json.load(open(sys.argv[1], encoding="utf-8")).get("graph") or "")
PY
)"
  fi
  [[ -s "$graph" ]] || {
    echo "ERROR: benchmark graph is unavailable; set EWM_GRAPH explicitly" >&2
    exit 2
  }
  "$PY" -m experiments.ewm_top10.prepare_fixed_windows \
    --retrieval "$retrieval" \
    --test "$EVAL" --corpus "$CORPUS" --graph "$graph" \
    --window 30 \
    --expected-method-prefix hlm_retriever \
    --output-method "hlm_retriever_seed${HLM_RETRIEVER_SEED}_temporal_top30" \
    --out "$WINDOW"
fi
[[ "$MODE" == "smoke" || "$MODE" == "full" ]] || {
  echo "ERROR: MODE must be smoke or full" >&2
  exit 2
}
[[ "$HLM_COST_STATUS" == "priced" || "$HLM_COST_STATUS" == "not_provided" || "$HLM_COST_STATUS" == "no_charge" ]] || {
  echo "ERROR: HLM_COST_STATUS must be priced, not_provided, or no_charge" >&2
  exit 2
}
[[ "$HLM_DISABLE_THINKING" == "0" || "$HLM_DISABLE_THINKING" == "1" ]] || {
  echo "ERROR: HLM_DISABLE_THINKING must be 0 or 1" >&2
  exit 2
}
if [[ "$MODE" == "full" && "$HLM_COST_STATUS" == "priced" && ( "$HLM_INPUT_PRICE_PER_M" == "0" || "$HLM_OUTPUT_PRICE_PER_M" == "0" ) ]]; then
  echo "ERROR: full mode requires non-zero HLM_INPUT_PRICE_PER_M and HLM_OUTPUT_PRICE_PER_M" >&2
  exit 2
fi

mkdir -p "$PREDICTION_DIR" "$METRIC_DIR" "$CACHE_DIR" "$REPORT_DIR"
export HLM_API_KEY

echo "== Complete HLM-style retriever and Analyzer--Decider =="
echo "mode=$MODE model=$HLM_MODEL endpoint=$HLM_ENDPOINT"
echo "run_label=$SAFE_RUN_LABEL output=$RUN_ROOT"
echo "evaluation_protocol=$EVAL_PROTOCOL queries=$EVAL"
echo "cost_status=$HLM_COST_STATUS"
echo "disable_thinking=$HLM_DISABLE_THINKING"
echo "candidate_source=HLM-style GTE retriever Top-30"
echo "retriever_seed=$HLM_RETRIEVER_SEED input=$WINDOW"
echo "No RW-Cite checkpoint or score is used."

common_args=(
  --retrieval "$WINDOW"
  --test "$EVAL"
  --corpus "$CORPUS"
  --model "$HLM_MODEL"
  --endpoint "$HLM_ENDPOINT"
  --api-key-env HLM_API_KEY
  --one-shot "$HLM_ONE_SHOT"
  --require-frozen-evidence
  --require-retrieval-method-prefix hlm_retriever
  --expected-window-size 30
  --cache-dir "$CACHE_DIR"
  --max-retries 3
)
if [[ "$HLM_DISABLE_THINKING" == "1" ]]; then
  common_args+=(--disable-thinking)
fi

if [[ "$MODE" == "smoke" ]]; then
  output="$PREDICTION_DIR/hlm_pipeline_smoke${SMOKE_QUERIES}.jsonl"
  "$PY" -m experiments.ewm_top10.rerank_hlm \
    "${common_args[@]}" --replicate 1 --max-queries "$SMOKE_QUERIES" \
    --out "$output"
  "$PY" -m experiments.ewm_top10.summarize_hlm_judge \
    --predictions "$output" \
    --input-price-per-million "$HLM_INPUT_PRICE_PER_M" \
    --output-price-per-million "$HLM_OUTPUT_PRICE_PER_M" \
    --cost-status "$HLM_COST_STATUS" \
    --out "$REPORT_DIR/smoke_summary.json"
  echo "== Smoke test complete; inspect $REPORT_DIR/smoke_summary.json =="
  exit 0
fi

output="$PREDICTION_DIR/hlm_pipeline_top30.jsonl"
"$PY" -m experiments.ewm_top10.rerank_hlm \
  "${common_args[@]}" --replicate 1 --out "$output"

"$PY" -m experiments.ewm_top10.evaluate \
  --test "$EVAL" --corpus "$CORPUS" \
  --predictions "$output" --out-dir "$METRIC_DIR"

"$PY" -m experiments.ewm_top10.summarize_hlm_judge \
  --predictions "$output" \
  --input-price-per-million "$HLM_INPUT_PRICE_PER_M" \
  --output-price-per-million "$HLM_OUTPUT_PRICE_PER_M" \
  --cost-status "$HLM_COST_STATUS" \
  --out "$REPORT_DIR/full_summary.json"

echo "== Complete =="
echo "Ranking metrics: $METRIC_DIR/hlm_pipeline_top30_summary.json"
echo "Reliability and cost: $REPORT_DIR/full_summary.json"
