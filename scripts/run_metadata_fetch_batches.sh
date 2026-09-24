#!/usr/bin/env bash
# Background-friendly incremental OAI metadata fetch until latest (date windows).
#
# Usage:
#   bash scripts/run_metadata_fetch_batches.sh
#   BATCH_DAYS=31 MAX_BATCHES=0 nohup bash scripts/run_metadata_fetch_batches.sh \
#     > logs/metadata_fetch_batches.nohup 2>&1 &
set -euo pipefail
ROOT="${RWCITE_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$ROOT"
export RWCITE_ROOT="$ROOT"
# shellcheck source=rwcite_python.sh
source "$(dirname "$0")/rwcite_python.sh"

BATCH_DAYS="${BATCH_DAYS:-31}"
MAX_BATCHES="${MAX_BATCHES:-0}"  # 0 = unlimited (until today)
BATCH_SLEEP="${BATCH_SLEEP:-5}"
PAGE_SLEEP="${PAGE_SLEEP:-3}"
TIMEOUT_SEC="${TIMEOUT_SEC:-180}"
LOG_DIR="${LOG_DIR:-logs}"
LOG_FILE="${LOG_FILE:-$LOG_DIR/metadata_fetch_batches.log}"

mkdir -p "$LOG_DIR"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

log "Starting incremental metadata fetch (batch_days=$BATCH_DAYS)"

ARGS=(
  --incremental
  --batch-days "$BATCH_DAYS"
  --batch-sleep "$BATCH_SLEEP"
  --page-sleep "$PAGE_SLEEP"
  --timeout-sec "$TIMEOUT_SEC"
)

if [[ "$MAX_BATCHES" -gt 0 ]]; then
  ARGS+=(--max-batches "$MAX_BATCHES")
fi
if [[ -n "${SINCE_DATE:-}" ]]; then
  ARGS+=(--since-date "$SINCE_DATE")
fi
if [[ -n "${UNTIL_DATE:-}" ]]; then
  ARGS+=(--until-date "$UNTIL_DATE")
fi

"$PY" -m rwcite.cli.fetch_metadata "${ARGS[@]}" 2>&1 | tee -a "$LOG_FILE"
log "Finished incremental metadata fetch"
