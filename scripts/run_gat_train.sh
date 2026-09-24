#!/usr/bin/env bash
# Train the paper GAT for an enabled benchmark domain.
#   DOMAIN=ewm CUDA_VISIBLE_DEVICES=0 bash scripts/run_gat_train.sh
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
export RR_STRUCT_MASK_QUERY_OUT="${RR_STRUCT_MASK_QUERY_OUT:-1}"
export RR_STRUCT_MASK_FUTURE_OUT="${RR_STRUCT_MASK_FUTURE_OUT:-1}"
if [[ -z "${N_TRAIN}" || "$N_TRAIN" == "0" || "$WINDOWS_TRAIN_TAG" == *XXX* ]]; then
  echo "ERROR: N_TRAIN unset — run dump_admit first" >&2
  exit 1
fi
WIN="${WINDOWS:-$GAT_MVP_DIR/windows_${WINDOWS_TRAIN_TAG}.npz}"
# Peer-safe: only one GPU train per domain (overlap vs sequential orch).
# .gpu_pipeline_started = claim during train; .gpu_pipeline_done / OK wave_gat = full train+eval.
GUARD_DIR="${RWCITE_ROOT}/logs/retrain_3.1/${DOMAIN}"
mkdir -p "$GUARD_DIR"
GPU_STAMP="$GUARD_DIR/.gpu_pipeline_started"
GPU_DONE="$GUARD_DIR/.gpu_pipeline_done"
GAT_LOG="$GUARD_DIR/wave_gat.log"
if [[ -f "$GPU_DONE" ]] || grep -q "OK wave_gat DOMAIN=${DOMAIN}" "$GAT_LOG" 2>/dev/null; then
  echo "== gat train skip DOMAIN=$DOMAIN (already OK) =="
  exit 0
fi
# Parallel orch (CE∥GAT): claim stamp without waiting on a peer that never writes OK.
if [[ "${SKIP_GAT_PEER_WAIT:-0}" == "1" ]]; then
  echo "pid=$$ ts=$(date -Is) skip_peer=1" > "$GPU_STAMP"
else
  wait_peer() {
    echo "== gat train wait peer DOMAIN=$DOMAIN =="
    while [[ ! -f "$GPU_DONE" ]] && ! grep -q "OK wave_gat DOMAIN=${DOMAIN}" "$GAT_LOG" 2>/dev/null; do
      sleep 15
    done
    echo "== gat train peer done DOMAIN=$DOMAIN =="
  }
  if [[ -f "$GPU_STAMP" ]]; then
    wait_peer
    exit 0
  fi
  if ! (set -o noclobber; echo "pid=$$ ts=$(date -Is)" > "$GPU_STAMP") 2>/dev/null; then
    wait_peer
    exit 0
  fi
fi
echo "== gat train DOMAIN=$DOMAIN GPU=${CUDA_VISIBLE_DEVICES:-unset} out=$GAT_MVP_DIR win=$WIN =="
set +e
"$PY" -m rwcite.gat.train \
  --domain "$DOMAIN" \
  --out-dir "$GAT_MVP_DIR" \
  --windows "$WIN" \
  "$@"
rc=$?
set -e
if [[ $rc -ne 0 ]]; then
  rm -f "$GPU_STAMP"
  echo "ERROR: gat train failed DOMAIN=$DOMAIN rc=$rc (released stamp)" >&2
  exit "$rc"
fi
# Note: do NOT touch .gpu_pipeline_done here — orch writes it after eval / OK wave_gat.
