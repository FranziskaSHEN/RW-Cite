#!/usr/bin/env bash
# True CE-sent (cite sentences, SHORT=1) × R2 E4 GAT L0 blend on frozen windows.
# Default CE = stage1_nofuture (HN stage2 retired for L0 after A3 flip).
set -euo pipefail
ROOT="${RWCITE_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$ROOT"
PY="${RWCITE_PYTHON:-python3}"
if [[ -x "${ROOT}/.venv/bin/python" && -z "${RWCITE_PYTHON:-}" && "${PY}" == "python3" ]]; then
  PY="${ROOT}/.venv/bin/python"
fi
export RWCITE_ROOT="$ROOT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export PYTHONUNBUFFERED=1
export RR_GEXF_OVERRIDE="${RR_GEXF_OVERRIDE:-$ROOT/embodied_world_model_retrieval/description/test_graph_rr.o2.nofuture.gexf}"
export RR_CE_SENT_SHORT=1
export RR_CE_SENT_MAX_SENTS=3
export RR_CE_SENT_MAX_LEN=256
export RR_CE_SENT_CITELINK_BLEND=0
CE_PATH="${CE_PATH:-$ROOT/embodied_world_model_retrieval/ranker/ce_sent/stage1_nofuture}"
CE_CACHE="${CE_CACHE:-embodied_world_model_retrieval/ranker/gat_mvp/ce_scores_test381_sent_s1.npz}"
OUT_PREFIX="${OUT_PREFIX:-ewm_gat_ce_sent_s1_blend_test381}"
REPORT_NAME="${REPORT_NAME:-ce_sent_s1_blend_report.json}"
LOG=logs/gat_mvp
mkdir -p "$LOG"

echo "START CE-sent s1 blend $(date) GPU=${CUDA_VISIBLE_DEVICES} CE=${CE_PATH} MAX_LEN=${RR_CE_SENT_MAX_LEN}" | tee -a "$LOG/fuse_ce_sent_s1.log"
$PY -u -m rwcite.gat.fuse_ce --step all --mode ce_sent --short 1 --max-sents 3 \
  --force-cache --fine-alpha \
  --ce-path "$CE_PATH" \
  --ce-cache "$CE_CACHE" \
  --out-prefix "$OUT_PREFIX" \
  --report-name "$REPORT_NAME" \
  2>&1 | tee -a "$LOG/fuse_ce_sent_s1.log"
echo "DONE CE-sent s1 blend $(date)" | tee -a "$LOG/fuse_ce_sent_s1.log"
