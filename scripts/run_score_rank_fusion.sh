#!/usr/bin/env bash
# Paper-defined score-and-rank fusion for an enabled benchmark domain.
#   DOMAIN=ewm CUDA_VISIBLE_DEVICES=0 bash scripts/run_score_rank_fusion.sh
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
ALPHA="${FUSION_ALPHA:-0.4}"
RRF_K="${FUSION_RANK_K:-20}"
RRF_W="${FUSION_RANK_WEIGHT:-0.4}"
FORCE_FLAGS="${FUSION_FORCE_FLAGS:-}"

echo "START score-and-rank fusion DOMAIN=$DOMAIN $(date) alpha=${ALPHA} rank_k=${RRF_K} rank_weight=${RRF_W}" | tee -a "$LOG/score_rank_fusion.log"
# shellcheck disable=SC2086
$PY -u -m rwcite.gat.score_rank_fusion \
  --domain "$DOMAIN" \
  --alpha "$ALPHA" --rrf-k "$RRF_K" --rrf-w "$RRF_W" $FORCE_FLAGS "$@" \
  2>&1 | tee -a "$LOG/score_rank_fusion.log"
echo "DONE score-and-rank fusion DOMAIN=$DOMAIN $(date)" | tee -a "$LOG/score_rank_fusion.log"
