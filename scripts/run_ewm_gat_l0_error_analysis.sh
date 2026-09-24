#!/usr/bin/env bash
# L0 error analysis vs E4 / fair CE (CPU). Prefer after run_ewm_gat_l0_default.sh.
set -euo pipefail
ROOT="${RWCITE_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$ROOT"
PY="${RWCITE_PYTHON:-python3}"
if [[ -x "${ROOT}/.venv/bin/python" && -z "${RWCITE_PYTHON:-}" && "${PY}" == "python3" ]]; then
  PY="${ROOT}/.venv/bin/python"
fi
export RWCITE_ROOT="$ROOT"
export PYTHONUNBUFFERED=1
LOG=logs/gat_mvp
mkdir -p "$LOG"

echo "START L0 error analysis $(date)" | tee -a "$LOG/l0_error_analysis.log"
$PY -u -m rwcite.gat.analyze_l0 "$@" 2>&1 | tee -a "$LOG/l0_error_analysis.log"
echo "DONE L0 error analysis $(date)" | tee -a "$LOG/l0_error_analysis.log"
