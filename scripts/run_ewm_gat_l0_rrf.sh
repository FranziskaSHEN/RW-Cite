#!/usr/bin/env bash
# Alias → run_gat_l0_default.sh (L0_rrf ≡ default).
set -euo pipefail
ROOT="${RWCITE_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
export DOMAIN="${DOMAIN:-ewm}"
exec bash "$ROOT/scripts/run_gat_l0_default.sh" "$@"
