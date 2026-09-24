#!/usr/bin/env bash
set -euo pipefail
ROOT="${RWCITE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ROOT"
export RWCITE_ROOT="$ROOT"
mkdir -p logs
pkill -f 'scripts/download_qwen3_32b.py' 2>/dev/null || true
sleep 2
# Keep partial *.incomplete for resume; low parallelism; then one-by-one fill gaps.
source "$ROOT/scripts/rwcite_python.sh"
nohup "$PY" -u "$ROOT/scripts/download_qwen3_32b.py" \
  --source modelscope --min-free-gb 40 --retries 10 --max-workers 2 --one-by-one \
  >"$ROOT/logs/download_qwen3_32b_ms_retry.log" 2>&1 &
echo $! | tee "$ROOT/logs/download_qwen3_32b_ms_retry.pid"
sleep 8
ps -p "$(cat "$ROOT/logs/download_qwen3_32b_ms_retry.pid")" -o pid,etime,cmd || true
head -15 "$ROOT/logs/download_qwen3_32b_ms_retry.log" || true
du -sh "$ROOT/models/base/Qwen3-32B" || true
