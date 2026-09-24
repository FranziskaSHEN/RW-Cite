#!/usr/bin/env bash
# Batch GLM topic generation for metadata papers missing arXiv Topics.
#
#   source ~/.rwcite_env   # export RWCITE_LLM_API_KEY=...
#   BATCH_SIZE=5000 TOPIC_WORKERS=16 nohup bash scripts/run_topic_supplement_batches.sh \
#     > logs/topic_supplement_batches.nohup 2>&1 &
set -euo pipefail

ROOT="${RWCITE_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$ROOT"
export RWCITE_ROOT="$ROOT"
# shellcheck source=rwcite_python.sh
source "$(dirname "$0")/rwcite_python.sh"

BATCH_SIZE="${BATCH_SIZE:-5000}"
TOPIC_BATCH_SIZE="${TOPIC_BATCH_SIZE:-128}"
TOPIC_WORKERS="${TOPIC_WORKERS:-16}"
TOPIC_SLEEP="${TOPIC_SLEEP:-0}"
MAX_BATCHES="${MAX_BATCHES:-0}"  # 0 = unlimited
# Optional: SINCE_DATE=2026-07-29 — default is corpus_state.last_metadata_from
LOG_DIR="${LOG_DIR:-logs}"
LOG_FILE="${LOG_FILE:-$LOG_DIR/topic_supplement_batches.log}"

mkdir -p "$LOG_DIR"

if [[ -z "${RWCITE_LLM_API_KEY:-}" && -n "${LITBENCH_LLM_API_KEY:-}" ]]; then
  export RWCITE_LLM_API_KEY="$LITBENCH_LLM_API_KEY"
fi
if [[ -z "${RWCITE_LLM_API_KEY:-}" && -z "${OPENAI_API_KEY:-}" ]]; then
  echo "ERROR: set RWCITE_LLM_API_KEY (or OPENAI_API_KEY) before running" >&2
  exit 1
fi

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

count_supplement() {
  if [[ -f datasets/arxiv_topics_supplement.jsonl ]]; then
    wc -l < datasets/arxiv_topics_supplement.jsonl | tr -d ' '
  else
    echo 0
  fi
}

EXTRA_ARGS=()
if [[ -n "${SINCE_DATE:-}" ]]; then
  EXTRA_ARGS+=(--since-date "$SINCE_DATE")
fi
if [[ -n "${UNTIL_DATE:-}" ]]; then
  EXTRA_ARGS+=(--until-date "$UNTIL_DATE")
fi
if [[ "${ALL_MISSING_TOPICS:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--all-missing-topics)
fi

log "Starting topic supplement batches (batch_size=$BATCH_SIZE, workers=$TOPIC_WORKERS)"
if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  log "Extra args: ${EXTRA_ARGS[*]}"
fi
before_total="$(count_supplement)"
log "Supplement records before: $before_total"

batch_num=0
while true; do
  batch_num=$((batch_num + 1))
  if [[ "$MAX_BATCHES" -gt 0 && "$batch_num" -gt "$MAX_BATCHES" ]]; then
    log "Reached MAX_BATCHES=$MAX_BATCHES, stopping"
    break
  fi

  prev="$(count_supplement)"
  log "=== Batch $batch_num: up to $BATCH_SIZE papers (supplement=$prev) ==="

  if ! "$PY" -m rwcite.cli.update_corpus \
    --skip-fetch \
    --skip-embeddings \
    --max-topics "$BATCH_SIZE" \
    --topic-batch-size "$TOPIC_BATCH_SIZE" \
    --topic-workers "$TOPIC_WORKERS" \
    --topic-sleep "$TOPIC_SLEEP" \
    "${EXTRA_ARGS[@]}" \
    2>&1 | tee -a "$LOG_FILE"; then
    log "Batch $batch_num failed"
    exit 1
  fi

  after="$(count_supplement)"
  added=$((after - prev))
  log "Batch $batch_num done: +$added records (total=$after)"

  if [[ "$added" -eq 0 ]]; then
    log "No new records appended; all missing topics processed"
    break
  fi
done

log "Finished. Supplement total: $(count_supplement) (started from $before_total)"
