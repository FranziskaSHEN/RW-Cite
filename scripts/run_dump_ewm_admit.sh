#!/usr/bin/env bash
# Thin wrapper → run_dump_admit.sh (DOMAIN=ewm).
set -euo pipefail
ROOT="${RWCITE_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
export DOMAIN=ewm
exec bash "$ROOT/scripts/run_dump_admit.sh" "$@"
