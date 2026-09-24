#!/usr/bin/env bash
# Usage: GPUS=4,5,6,7 bash scripts/launch_rwcite_comparison.sh CONFIG OUTPUT [SMOKE_COUNT]
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$ROOT"
PY="${PY:-$ROOT/.venv/bin/python}"
CONFIG="${1:?Pass the reviewed comparison config}"
OUT="${2:?Pass a new output directory inside this checkout}"
SMOKE="${3:-0}"
GPUS="${GPUS:-4,5,6,7}"
OUT="$("$PY" -c 'import sys; from pathlib import Path; from experiments.ewm_top10.rwcite_contract import safe_output; print(safe_output(Path.cwd(), sys.argv[1]))' "$OUT")"
[[ -f "$CONFIG" ]] || { echo "Config missing: $CONFIG" >&2; exit 2; }
[[ ! -d "$OUT/.running" ]] || { echo "Run is locked; inspect existing processes before restarting." >&2; exit 2; }
mkdir -p "$OUT/worker_logs" "$OUT/launcher_logs"
IFS=',' read -r -a GPU_LIST <<< "$GPUS"
for ((i=0; i<${#GPU_LIST[@]}; i++)); do
  touch "$OUT/worker_logs/shard$i.log"
done
STAMP="$(date +%Y%m%dT%H%M%S)_$$"
LOG="$OUT/launcher_logs/$STAMP.log"
nohup env PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
  "$PY" -u -m experiments.ewm_top10.run_rwcite_comparison run \
  --config "$CONFIG" --out "$OUT" --gpus "$GPUS" --max-queries "$SMOKE" \
  >"$LOG" 2>&1 &
PID=$!
printf '%s\n' "$PID" | tee "$OUT/launcher_logs/$STAMP.pid"
printf 'Launcher: %s\nOutput: %s\nCtrl-C stops log-following, not the nohup job.\n' "$LOG" "$OUT"
tail -n 60 -F "$LOG" "$OUT"/worker_logs/shard*.log
