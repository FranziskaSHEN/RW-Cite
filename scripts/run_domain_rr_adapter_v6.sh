#!/usr/bin/env bash
# Release 3.1 L0_rrf pools → RR v6 Cite QLoRA (Qwen3-32B).
# Usage:
#   DOMAIN=ewm bash scripts/run_domain_rr_adapter_v6.sh dump_pools
#   DOMAIN=ewm bash scripts/run_domain_rr_adapter_v6.sh build
#   DOMAIN=ewm NPROC=4 bash scripts/run_domain_rr_adapter_v6.sh train
#   DOMAIN=ewm bash scripts/run_domain_rr_adapter_v6.sh eval
#   # full L0–L2 (AS protocol; default cite_one rewrite); avoid GPU0 if smoke occupies it:
#   CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NPROC=8 DOMAIN=ewm \
#     bash scripts/run_domain_rr_adapter_v6.sh eval --max-samples 0 \
#     --out embodied_world_model_retrieval/adapter/rr_v6/data/eval_cite_full370.json
#   # resume default on; shards at <out>.shards/rank*.jsonl
#   DOMAIN=ewm bash scripts/run_domain_rr_adapter_v6.sh all
set -euo pipefail
ROOT="${RWCITE_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$ROOT"
export RWCITE_ROOT="$ROOT"
PY="${RWCITE_PYTHON:-python3}"
if [[ -x "${ROOT}/.venv/bin/python" && -z "${RWCITE_PYTHON:-}" && "${PY}" == "python3" ]]; then
  PY="${ROOT}/.venv/bin/python"
fi
export PYTHONUNBUFFERED=1
export WANDB_MODE="${WANDB_MODE:-offline}"

# shellcheck source=scripts/lib_domain_env.sh
source "$ROOT/scripts/lib_domain_env.sh"
rwcite_load_domain_env

CMD="${1:-}"
if [[ -z "$CMD" ]]; then
  echo "Usage: DOMAIN=ewm|sqc|gw $0 {dump_pools|build|train|eval|all}" >&2
  exit 2
fi
shift || true

ADAPTER_DIR="$ROOT/$LANE/adapter/rr_v6"
export ADAPTER_DIR
mkdir -p "$ADAPTER_DIR/data" "$ADAPTER_DIR/model" "$ADAPTER_DIR/trainer_outputs"
POOL_N="${POOL_N:-50}"
NPROC="${NPROC:-1}"
CFG="${CFG:-configs/rr_adapter_v6.yaml}"

check_train_extras() {
  "$PY" - <<'PY'
import importlib
missing=[]
for m in ("peft","bitsandbytes","accelerate"):
    try: importlib.import_module(m)
    except ImportError: missing.append(m)
if missing:
    raise SystemExit("missing "+str(missing)+"; run: pip install -e '.[train]'")
print("train extras ok")
PY
}

run_dump() {
  echo "== dump_pools DOMAIN=$DOMAIN pool_n=$POOL_N =="
  "$PY" -u -m rwcite.cli.dump_rr_l0_pools --domain "$DOMAIN" --pool-n "$POOL_N" "$@"
}

run_build() {
  echo "== build DOMAIN=$DOMAIN =="
  "$PY" -u -m rwcite.cli.build_rr_adapter_v6 --domain "$DOMAIN" --pool-n "$POOL_N" "$@"
}

run_train() {
  check_train_extras
  if [[ ! -d "$ROOT/models/base/Qwen3-32B" ]]; then
    echo "ERROR: missing models/base/Qwen3-32B; run scripts/download_qwen3_32b.py" >&2
    exit 1
  fi
  echo "== train DOMAIN=$DOMAIN nproc=$NPROC =="
  if [[ "$NPROC" -gt 1 ]]; then
    "$PY" -m torch.distributed.run --nproc_per_node="$NPROC" \
      -m rwcite.cli.train_rr_adapter --domain "$DOMAIN" --config "$CFG" --fresh "$@"
  else
    "$PY" -u -m rwcite.cli.train_rr_adapter --domain "$DOMAIN" --config "$CFG" --fresh "$@"
  fi
}

run_eval() {
  check_train_extras
  echo "== eval DOMAIN=$DOMAIN nproc=$NPROC =="
  if [[ "$NPROC" -gt 1 ]]; then
    "$PY" -m torch.distributed.run --nproc_per_node="$NPROC" \
      -m rwcite.cli.eval_rr_adapter_cite --domain "$DOMAIN" "$@"
  else
    "$PY" -u -m rwcite.cli.eval_rr_adapter_cite --domain "$DOMAIN" "$@"
  fi
}

case "$CMD" in
  dump_pools) run_dump "$@" ;;
  build) run_build "$@" ;;
  train) run_train "$@" ;;
  eval) run_eval "$@" ;;
  all)
    run_dump
    run_build
    run_train
    run_eval
    ;;
  *)
    echo "Unknown command: $CMD" >&2
    exit 2
    ;;
esac

echo "ADAPTER_DIR=$ADAPTER_DIR"
echo "DONE $CMD DOMAIN=$DOMAIN $(date)"
