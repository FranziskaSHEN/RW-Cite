#!/usr/bin/env bash
# GAT Phase 0: measure struct@400 window in-degree (no CE).
# Usage:
#   bash scripts/run_ewm_gat_phase0.sh
#   MAX_SAMPLES=40 bash scripts/run_ewm_gat_phase0.sh   # smoke
set -euo pipefail
ROOT="${RWCITE_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$ROOT"
export RWCITE_ROOT="$ROOT"

PY="${RWCITE_PYTHON:-python3}"
if [[ -x "${ROOT}/.venv/bin/python" && -z "${RWCITE_PYTHON:-}" && "${PY}" == "python3" ]]; then
  PY="${ROOT}/.venv/bin/python"
fi

# shellcheck disable=SC1091
source configs/env_domain_rr.sh ewm
export RR_STRUCT_COARSE=linear
export RR_RANKER_WITH_EMB=0
unset RR_RANKER_CE_SENT_PATH || true
unset RR_RANKER_CE_PATH || true
export RR_SPLIT_DIR="${RR_SPLIT_DIR:-$ROOT/embodied_world_model_retrieval/data/splits}"

WHICH="${WHICH:-test}"
WINDOW="${WINDOW:-400}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
OUT="${OUT:-embodied_world_model_retrieval/ranker/gat_mvp/phase0_window_indeg_${WHICH}.json}"
GPU="${GPU:-0}"

mkdir -p "$(dirname "$OUT")" logs
LOG="logs/ewm_gat_phase0_${WHICH}_$(date +%Y%m%d_%H%M%S).log"

echo "== Phase 0 window indeg: which=${WHICH} window=${WINDOW} max=${MAX_SAMPLES} gpu=${GPU} =="
echo "  log → $LOG"
export CUDA_VISIBLE_DEVICES="$GPU"
export RWCITE_EMBEDDER_DEVICE=cuda

EXTRA=()
if [[ "$MAX_SAMPLES" != "0" ]]; then
  EXTRA+=(--max-samples "$MAX_SAMPLES")
fi

"$PY" -m rwcite.gat.phase0_window_indeg \
  --domain ewm \
  --which "$WHICH" \
  --window "$WINDOW" \
  --out "$OUT" \
  "${EXTRA[@]}" \
  2>&1 | tee "$LOG"

echo "== done: $OUT =="
