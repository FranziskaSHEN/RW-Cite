#!/usr/bin/env bash
# Evaluate the paper GAT for an enabled benchmark domain.
#   DOMAIN=ewm CUDA_VISIBLE_DEVICES=0 bash scripts/run_gat_eval.sh
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
TAG="${OUT_TAG:-$GAT_EVAL_TAG}"
MAX="${MAX_SAMPLES:-$N_TEST}"
if [[ -z "${N_TEST}" || "$N_TEST" == "0" ]]; then
  echo "ERROR: N_TEST unset/0 — run dump_admit first (refuse silent ewm 381 fallback)" >&2
  exit 1
fi
if [[ -z "$MAX" || "$MAX" == "0" ]]; then
  MAX="$N_TEST"
fi
echo "== gat eval DOMAIN=$DOMAIN tag=$TAG max=$MAX =="
exec "$PY" -m rwcite.gat.evaluate \
  --domain "$DOMAIN" \
  --out-dir "$GAT_MVP_DIR" \
  --out-tag "$TAG" \
  --max-samples "$MAX" \
  "$@"
