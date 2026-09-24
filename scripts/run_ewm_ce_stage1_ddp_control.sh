#!/usr/bin/env bash
# CE stage1_nofuture DDP control (isolated out-dir; does NOT overwrite stage1_nofuture).
# Compare to existing 1gpu stage1_nofuture (train_meta ddp_world_size=1).
# Protocol aligned with GAT R4: ddp_epoch_mode=full, NO √N lr.
# Usage:
#   GPUS=1,2,3,4,5 bash scripts/run_ewm_ce_stage1_ddp_control.sh   # 5gpu while GAT uses 0
#   GPUS=0,1,2,3,4,5 bash scripts/run_ewm_ce_stage1_ddp_control.sh
set -euo pipefail
ROOT="${RWCITE_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$ROOT"
PY="${RWCITE_PYTHON:-python3}"
if [[ -x "${ROOT}/.venv/bin/python" && -z "${RWCITE_PYTHON:-}" && "${PY}" == "python3" ]]; then
  PY="${ROOT}/.venv/bin/python"
fi
export RWCITE_ROOT="$ROOT"
export PYTHONUNBUFFERED=1

GPUS="${GPUS:-0,1,2,3,4,5}"
IFS=',' read -r -a GPU_ARR <<< "$GPUS"
N="${#GPU_ARR[@]}"
PAIRS="${PAIRS:-$ROOT/embodied_world_model_retrieval/ranker/pool/ce_base_pairs_nofuture.jsonl}"
BASE="${BASE:-$ROOT/models/base/scibert_scivocab_uncased}"
OUT="${OUT:-$ROOT/embodied_world_model_retrieval/ranker/ce_sent/stage1_nofuture_ddp${N}}"
LOG_DIR=logs/gat_fullU
mkdir -p "$LOG_DIR" "$OUT"

if [[ ! -f "$PAIRS" ]]; then
  echo "missing pairs $PAIRS" >&2
  exit 1
fi
if [[ ! -d "$BASE" ]]; then
  echo "missing base $BASE" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="$GPUS"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29541}"
export MASTER_PORT
unset RANK WORLD_SIZE LOCAL_RANK LOCAL_WORLD_SIZE GROUP_RANK ROLE_RANK \
  TORCHELASTIC_RUN_ID TORCHELASTIC_RESTART_COUNT ACCELERATE_MIXED_PRECISION || true

echo "== CE stage1 DDP control nproc=$N GPUS=$GPUS → $OUT (no √N, mode=full) ==" | tee -a "$LOG_DIR/ce_ddp_control.log"
"$PY" -m torch.distributed.run \
  --standalone \
  --nproc_per_node="$N" \
  --master_port="$MASTER_PORT" \
  -m rwcite.cli.train_ce_sent \
  --pairs "$PAIRS" \
  --base "$BASE" \
  --out "$OUT" \
  --epochs "${EPOCHS:-2}" \
  --freeze-epochs "${FREEZE_EPOCHS:-0}" \
  --list-size "${LIST_SIZE:-64}" \
  --lr "${LR:-2e-5}" \
  --max-length "${MAX_LENGTH:-256}" \
  --seed "${SEED:-0}" \
  --ddp-epoch-mode full \
  --no-lr-scale-ddp \
  --train-only \
  </dev/null 2>&1 | tee -a "$LOG_DIR/ce_ddp_control.log"

echo "OK: CE DDP control → $OUT"
