#!/usr/bin/env bash
# Preprocess GAT windows and folds for an enabled benchmark domain.
# Usage:
#   DOMAIN=ewm CUDA_VISIBLE_DEVICES=0 bash scripts/run_gat_preprocess.sh
#   DOMAIN=sqc CUDA_VISIBLE_DEVICES=2 bash scripts/run_gat_preprocess.sh windows_train
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
export RR_STRUCT_COARSE="${RR_STRUCT_COARSE:-linear}"

STEP="${1:-core}"
SKIP_CONTRACT="${SKIP_CONTRACT:-0}"

run_core() {
  echo "== encode DOMAIN=$DOMAIN =="
  "$PY" -m rwcite.gat.preprocess --domain "$DOMAIN" --step encode
  echo "== whiten =="
  "$PY" -m rwcite.gat.preprocess --domain "$DOMAIN" --step whiten
  echo "== adj =="
  "$PY" -m rwcite.gat.preprocess --domain "$DOMAIN" --step adj
  echo "== neighbors =="
  "$PY" -m rwcite.gat.preprocess --domain "$DOMAIN" --step neighbors
  echo "== windows test =="
  "$PY" -m rwcite.gat.preprocess --domain "$DOMAIN" --step windows --windows test
}

run_windows_train() {
  # Shared-node default 2||2. Exclusive 128-core (sce next-round): prefer 6||6 + OMP=8
  # (see docs/RELEASE_3.1.md §12). Uncapped OMP at 8-way previously starved CPU (~40min/query).
  N="${GAT_WINDOW_SHARDS:-2}"
  PARALLEL="${GAT_WINDOW_PARALLEL:-2}"
  TAG="${WINDOWS_TRAIN_TAG}"
  if [[ "$TAG" == *XXX* || -z "${N_TRAIN}" || "$N_TRAIN" == "0" ]]; then
    echo "ERROR: N_TRAIN unset — run dump_admit first" >&2
    exit 1
  fi
  local merged="$GAT_MVP_DIR/windows_${TAG}.npz"
  if [[ "${FORCE_WINDOWS:-0}" != "1" && -f "$merged" ]]; then
    echo "== windows train reuse existing $merged (set FORCE_WINDOWS=1 to rebuild) =="
    return 0
  fi
  # Single-flight per domain: do not dual-build windows on CPU.
  local flockf="${RWCITE_ROOT}/logs/retrain_3.1/${DOMAIN}/.windows_build.flock"
  mkdir -p "${RWCITE_ROOT}/logs/retrain_3.1/${DOMAIN}"
  exec 9>"$flockf"
  if ! flock -n 9; then
    echo "== windows train wait peer DOMAIN=$DOMAIN (flock busy) =="
    while [[ ! -f "$merged" ]]; do sleep 30; done
    echo "== windows train peer ready $merged =="
    return 0
  fi
  echo "== windows train shards N=$N parallel=$PARALLEL tag=$TAG DOMAIN=$DOMAIN =="
  pids=()
  for i in $(seq 0 $((N - 1))); do
    echo "  shard ${i}/${N} (sequential wait if parallel cap reached)"
    "$PY" -m rwcite.gat.preprocess --domain "$DOMAIN" --step windows --windows train \
      --shard-idx "$i" --shard-count "$N" &
    pids+=($!)
    # Cap concurrent shards
    while (( ${#pids[@]} >= PARALLEL )); do
      wait "${pids[0]}"
      pids=("${pids[@]:1}")
    done
  done
  for pid in "${pids[@]}"; do
    wait "$pid"
  done
  echo "== merge train windows tag=$TAG =="
  "$PY" -m rwcite.gat.preprocess --domain "$DOMAIN" --step merge_windows \
    --merge-tag "$TAG" --shard-count "$N"
}

case "$STEP" in
  core) run_core ;;
  encode|whiten|adj|neighbors)
    "$PY" -m rwcite.gat.preprocess --domain "$DOMAIN" --step "$STEP"
    ;;
  windows_test)
    "$PY" -m rwcite.gat.preprocess --domain "$DOMAIN" --step windows --windows test
    ;;
  windows_train) run_windows_train ;;
  folds)
    echo "== folds DOMAIN=$DOMAIN =="
    "$PY" -m rwcite.gat.preprocess --domain "$DOMAIN" --step folds
    ;;
  all)
    run_core
    run_windows_train
    echo "== folds DOMAIN=$DOMAIN =="
    "$PY" -m rwcite.gat.preprocess --domain "$DOMAIN" --step folds
    ;;
  *)
    echo "unknown step: $STEP" >&2
    exit 2
    ;;
esac

echo "OK: preprocess DOMAIN=$DOMAIN step=$STEP"
if [[ "$DOMAIN" == "ewm" && "$SKIP_CONTRACT" != "1" ]]; then
  bash "$ROOT/scripts/check_ewm_3_contract.sh"
fi
