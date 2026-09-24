#!/usr/bin/env bash
# L0 default (= L0_rrf) for DOMAIN=ewm|sqc|gw.
#   DOMAIN=ewm CUDA_VISIBLE_DEVICES=0 bash scripts/run_gat_l0_default.sh
set -euo pipefail
ROOT="${RWCITE_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$ROOT"
PY="${RWCITE_PYTHON:-python3}"
if [[ -x "${ROOT}/.venv/bin/python" && -z "${RWCITE_PYTHON:-}" && "${PY}" == "python3" ]]; then
  PY="${ROOT}/.venv/bin/python"
fi
export RWCITE_ROOT="$ROOT"
export PYTHONUNBUFFERED=1
# shellcheck source=scripts/lib_domain_env.sh
source "$ROOT/scripts/lib_domain_env.sh"
rwcite_load_domain_env

export RR_CE_SENT_SHORT=1
export RR_CE_SENT_MAX_SENTS=3
export RR_CE_SENT_MAX_LEN=256
export RR_CE_SENT_CITELINK_BLEND=0
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

LOG="logs/gat_mvp_${DOMAIN}"
mkdir -p "$LOG"
ALPHA="${L0_ALPHA:-0.4}"
RRF_K="${L0_RRF_K:-20}"
RRF_W="${L0_RRF_W:-0.4}"
FORCE_FLAGS="${L0_FORCE_FLAGS:-}"

echo "START L0 default (=L0_rrf) DOMAIN=$DOMAIN $(date) alpha=${ALPHA} rrf_k=${RRF_K} rrf_w=${RRF_W}" | tee -a "$LOG/l0_default.log"
# shellcheck disable=SC2086
$PY -u -m rwcite.gat.l0_recipe \
  --domain "$DOMAIN" \
  --alpha "$ALPHA" --rrf-k "$RRF_K" --rrf-w "$RRF_W" $FORCE_FLAGS "$@" \
  2>&1 | tee -a "$LOG/l0_default.log"
echo "DONE L0 default DOMAIN=$DOMAIN $(date)" | tee -a "$LOG/l0_default.log"
