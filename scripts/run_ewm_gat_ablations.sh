#!/usr/bin/env bash
# Run the paper-aligned EWM graph-attention ablation entry point.
set -euo pipefail
ROOT="${RWCITE_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$ROOT"
PY="${RWCITE_PYTHON:-python3}"
if [[ -x "${ROOT}/.venv/bin/python" && -z "${RWCITE_PYTHON:-}" && "${PY}" == "python3" ]]; then
  PY="${ROOT}/.venv/bin/python"
fi
export RWCITE_ROOT="$ROOT"
exec "$PY" -m rwcite.gat.ablations "$@"
