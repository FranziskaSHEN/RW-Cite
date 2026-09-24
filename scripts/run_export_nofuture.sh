#!/usr/bin/env bash
# Export the split-masked graph for an enabled benchmark domain.
#   DOMAIN=ewm bash scripts/run_export_nofuture.sh
set -euo pipefail
ROOT="${RWCITE_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$ROOT"
PY="${RWCITE_PYTHON:-python3}"
if [[ -x "${ROOT}/.venv/bin/python" && -z "${RWCITE_PYTHON:-}" && "${PY}" == "python3" ]]; then
  PY="${ROOT}/.venv/bin/python"
fi
export RWCITE_ROOT="$ROOT"
# shellcheck source=scripts/lib_domain_env.sh
source "$ROOT/scripts/lib_domain_env.sh"
rwcite_load_domain_env
[[ -f "$GEXF_O2" ]] || { echo "missing $GEXF_O2" >&2; exit 1; }
[[ -d "$SPLIT_DIR" ]] || { echo "missing $SPLIT_DIR (run_dump_admit first)" >&2; exit 1; }
OUT="${GEXF_OUT:-$LANE/description/test_graph_rr.o2.nofuture.gexf}"
echo "== export_nofuture DOMAIN=$DOMAIN in=$GEXF_O2 out=$OUT =="
exec "$PY" -m rwcite.cli.export_nofuture_gexf \
  --domain "$DOMAIN" \
  --gexf-in "$GEXF_O2" \
  --gexf-out "$OUT" \
  --split-dir "$SPLIT_DIR"
