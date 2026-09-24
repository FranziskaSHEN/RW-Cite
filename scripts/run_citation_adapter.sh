#!/usr/bin/env bash
# Optional citation-sentence generator over frozen score-and-rank selections.
# Usage:
#   DOMAIN=ewm bash scripts/run_citation_adapter.sh dump_pools
#   DOMAIN=ewm bash scripts/run_citation_adapter.sh build
#   DOMAIN=ewm NPROC=4 bash scripts/run_citation_adapter.sh train
#   DOMAIN=ewm bash scripts/run_citation_adapter.sh eval
#   # Full evaluation with candidate-specific rewriting:
#   CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NPROC=8 DOMAIN=ewm \
#     bash scripts/run_citation_adapter.sh eval --max-samples 0
#   # resume default on; shards at <out>.shards/rank*.jsonl
#   DOMAIN=ewm bash scripts/run_citation_adapter.sh all
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

ADAPTER_DIR="$ROOT/$LANE/adapter/citation_generator"
export ADAPTER_DIR
mkdir -p "$ADAPTER_DIR/data" "$ADAPTER_DIR/model" "$ADAPTER_DIR/trainer_outputs"
POOL_N="${POOL_N:-50}"
NPROC="${NPROC:-1}"
CFG="${CFG:-configs/citation_adapter.yaml}"

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
  echo "== export frozen ranking pools DOMAIN=$DOMAIN pool_n=$POOL_N =="
  "$PY" -u -m rwcite.cli.export_ranking_pools --domain "$DOMAIN" --pool-n "$POOL_N" "$@"
}

run_build() {
  echo "== build citation-generation data DOMAIN=$DOMAIN =="
  "$PY" -u -m rwcite.cli.build_citation_adapter --domain "$DOMAIN" --pool-n "$POOL_N" "$@"
}

run_train() {
  check_train_extras
  if [[ ! -d "$ROOT/models/base/Qwen3-32B" ]]; then
    echo "ERROR: missing models/base/Qwen3-32B; run scripts/download_qwen3_32b.py" >&2
    exit 1
  fi
  echo "== train citation generator DOMAIN=$DOMAIN nproc=$NPROC =="
  if [[ "$NPROC" -gt 1 ]]; then
    "$PY" -m torch.distributed.run --nproc_per_node="$NPROC" \
      -m rwcite.cli.train_rr_adapter --domain "$DOMAIN" --config "$CFG" --fresh "$@"
  else
    "$PY" -u -m rwcite.cli.train_rr_adapter --domain "$DOMAIN" --config "$CFG" --fresh "$@"
  fi
}

run_eval() {
  check_train_extras
  echo "== evaluate citation generator DOMAIN=$DOMAIN nproc=$NPROC =="
  if [[ "$NPROC" -gt 1 ]]; then
    "$PY" -m torch.distributed.run --nproc_per_node="$NPROC" \
      -m rwcite.cli.evaluate_citation_adapter --domain "$DOMAIN" "$@"
  else
    "$PY" -u -m rwcite.cli.evaluate_citation_adapter --domain "$DOMAIN" "$@"
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
