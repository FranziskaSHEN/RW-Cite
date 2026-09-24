#!/usr/bin/env bash
# Paper cross-encoder training — single GPU only (no DDP).
# DOMAIN may be any enabled benchmark domain in configs/domains.yaml.
#
#   DOMAIN=ewm CUDA_VISIBLE_DEVICES=0 CE_GPUS=0 bash scripts/run_ce_stage1.sh
#   DOMAIN=driving CUDA_VISIBLE_DEVICES=2 CE_GPUS=2 bash scripts/run_ce_stage1.sh
#
# Requires CE_PAIRS (ce_base_pairs_nofuture.jsonl). Does NOT run stage2/HN.
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

# Pin to one physical GPU. Prefer CE_GPUS, else CE_GPU from pack, else CUDA_VISIBLE_DEVICES.
if [[ -n "${CE_GPUS:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="$CE_GPUS"
elif [[ -n "${CE_GPU:-}" && -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="$CE_GPU"
fi
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "ERROR: set CE_GPUS / CE_GPU or CUDA_VISIBLE_DEVICES to a single GPU id" >&2
  exit 2
fi
# Reject multi-GPU lists for this script
if [[ "$CUDA_VISIBLE_DEVICES" == *,* ]]; then
  echo "ERROR: run_ce_stage1.sh is single-GPU only (got CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES). No DDP." >&2
  exit 2
fi

PAIRS="${PAIRS:-$CE_PAIRS}"
BASE="${BASE:-$ROOT/models/base/scibert_scivocab_uncased}"
OUT="${OUT:-$CE_STAGE1}"
LOG_DIR="${LOG_DIR:-logs/ce_stage1_${DOMAIN}}"
mkdir -p "$LOG_DIR" "$OUT"

[[ -f "$PAIRS" ]] || { echo "missing pairs $PAIRS (build ce_base_pairs_nofuture first)" >&2; exit 1; }
[[ -d "$BASE" ]] || { echo "missing BASE $BASE" >&2; exit 1; }

echo "== CE stage1 SINGLE-GPU DOMAIN=$DOMAIN GPU=$CUDA_VISIBLE_DEVICES pairs=$PAIRS out=$OUT ==" | tee -a "$LOG_DIR/train.log"
# Explicitly avoid torch.distributed.run
"$PY" -u -m rwcite.cli.train_ce_sent \
  --pairs "$PAIRS" \
  --base "$BASE" \
  --out "$OUT" \
  --epochs "${EPOCHS:-2}" \
  --freeze-epochs "${FREEZE_EPOCHS:-0}" \
  --list-size "${LIST_SIZE:-64}" \
  --lr "${LR:-2e-5}" \
  --max-length "${MAX_LENGTH:-256}" \
  --warmup-ratio "${WARMUP_RATIO:-0.06}" \
  --seed "${SEED:-0}" \
  --train-only \
  2>&1 | tee -a "$LOG_DIR/train.log"

echo "OK: CE stage1 DOMAIN=$DOMAIN → $OUT"
