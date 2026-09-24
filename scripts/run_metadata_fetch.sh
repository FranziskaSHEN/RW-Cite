#!/usr/bin/env bash
# arXiv metadata: full snapshot download or OAI incremental fetch.
#
# Usage:
#   bash scripts/run_metadata_fetch.sh --download-snapshot
#   bash scripts/run_metadata_fetch.sh --incremental
#   bash scripts/run_metadata_fetch.sh --since-date 2026-01-01 --until-date 2026-01-31
#   MAX_BATCHES=2 bash scripts/run_metadata_fetch.sh --incremental   # smoke
set -euo pipefail
ROOT="${RWCITE_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$ROOT"
export RWCITE_ROOT="$ROOT"
# shellcheck source=rwcite_python.sh
source "$(dirname "$0")/rwcite_python.sh"

if [[ $# -eq 0 ]]; then
  sed -n '2,10p' "$0"
  exit 0
fi

exec "$PY" -m rwcite.cli.fetch_metadata "$@"
