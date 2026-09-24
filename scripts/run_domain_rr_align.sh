#!/usr/bin/env bash
# Thin wrapper → run_domain_rr_c2s.sh (Release B1 defaults).
# Prefer calling run_domain_rr_c2s.sh directly.
set -euo pipefail
ROOT="${RWCITE_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$ROOT"
exec bash scripts/run_domain_rr_c2s.sh "$@"
