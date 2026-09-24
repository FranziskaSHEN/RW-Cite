#!/usr/bin/env bash
# Thin wrapper → run_gat_train.sh (DOMAIN=ewm).
set -euo pipefail
ROOT="${RWCITE_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
export DOMAIN="${DOMAIN:-ewm}"
exec bash "$ROOT/scripts/run_gat_train.sh" "$@"
