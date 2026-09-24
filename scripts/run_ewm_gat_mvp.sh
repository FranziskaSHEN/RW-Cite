#!/usr/bin/env bash
# EWM GAT experiment runner. It does not modify the configured production
# cross-encoder checkpoint.
#
# Contract: nofuture GEXF + frozen test_admit_v1 jsonl; infer = universe → struct@400 → GAT.
# By default this script validates the data contract and prints the available
# stages. Set DO_TRAIN or DO_EVAL to execute them.
#
# Usage:
#   bash scripts/run_ewm_gat_mvp.sh              # contract check + dry-run plan
#   bash scripts/run_ewm_gat_mvp.sh --check-only
#   DO_TRAIN=1 bash scripts/run_ewm_gat_mvp.sh
set -euo pipefail
ROOT="${RWCITE_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$ROOT"
export RWCITE_ROOT="$ROOT"

PY="${RWCITE_PYTHON:-python3}"
if [[ -x "${ROOT}/.venv/bin/python" && -z "${RWCITE_PYTHON:-}" && "${PY}" == "python3" ]]; then
  PY="${ROOT}/.venv/bin/python"
fi

WD=embodied_world_model_retrieval
CHECK_ONLY=0
for arg in "$@"; do
  case "$arg" in
    --check-only) CHECK_ONLY=1 ;;
  esac
done

echo "== EWM split-masked data contract =="
bash scripts/check_ewm_3_contract.sh

if [[ "$CHECK_ONLY" == "1" ]]; then
  echo "check-only: done"
  exit 0
fi

echo
echo "== Available GAT stages (no automatic checkpoint cutover) =="
echo "  1) preprocess: encode title/abstract on nofuture nodes + train.jsonl golds"
echo "  2) train:      2-layer GAT + MLP (BPR/InfoNCE) → ${WD}/ranker/gat_mvp/"
echo "  3) eval:       universe → struct Top-400 → GAT score → ${WD}/ranker/eval/ewm_gat_mvp_test381_merged.json"
echo "  4) compare:    python -m rwcite.gat.compare vs 2.0 nofuture full381 (bar: hits@10 > 2.735)"
echo

if [[ "${DO_TRAIN:-0}" == "1" ]]; then
  echo "DO_TRAIN=1: training the GAT ranker"
  "$PY" -m rwcite.gat.train --domain ewm
fi

if [[ "${DO_EVAL:-0}" == "1" ]]; then
  echo "DO_EVAL=1: evaluating the GAT ranker"
  "$PY" -m rwcite.gat.evaluate --domain ewm --max-samples "${MAX_SAMPLES:-381}" --window "${WINDOW:-400}"
fi

echo "GAT runner finished."
