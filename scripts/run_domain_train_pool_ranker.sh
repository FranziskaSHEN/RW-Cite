#!/usr/bin/env bash
# Domain structural shortlist ranker: dump features (+emb) + fit linear logistic.
#
# Writes:
#   <domain>_retrieval/ranker/struct/feats/features_shard*of*.npz
#   <domain>_retrieval/ranker/struct/model.npz
#   <domain>_retrieval/ranker/struct/train_report.json
#
# Standard recipe (AS v1 aligned): WITH_EMB=1 LOSS=logistic, then infer with
# RR_STRUCT_COARSE=linear (emb_sims=None at CE shortlist time).
#
# Usage:
#   DOMAIN=gw GPUS=0,1,2,3 scripts/run_domain_train_pool_ranker.sh
#   FORCE=1 WITH_EMB=1 LOSS=logistic DOMAIN=ewm_as GPUS=0,1,2,3 \
#     scripts/run_domain_train_pool_ranker.sh
#
# Env:
#   SKIP_POOL_RANKER=1     skip entirely
#   WITH_EMB=1|0           fill emb_sim at dump (default 1)
#   LOSS=logistic|ranknet  fit objective (default logistic)
#   POOL_RANKER_NUM_SHARDS shard count (default: number of GPUs in GPUS)
#   POOL_RANKER_SHARDS     optional subset, e.g. 0,1,2,3 for a partial wave
#   STRUCT_MODEL / STRUCT_FEATS / TRAIN_JSONL  override layout defaults
set -euo pipefail
ROOT="${RWCITE_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$ROOT"
export RWCITE_ROOT="$ROOT"

DOMAIN="${DOMAIN:?DOMAIN required}"
FORCE="${FORCE:-0}"
SKIP_POOL_RANKER="${SKIP_POOL_RANKER:-0}"
WITH_EMB="${WITH_EMB:-1}"
LOSS="${LOSS:-logistic}"
# Capture user overrides before lib_domain_env overwrites STRUCT_MODEL.
_user_struct_model="${STRUCT_MODEL:-}"
_user_struct_feats="${STRUCT_FEATS:-}"
_user_train_jsonl="${TRAIN_JSONL:-}"
_user_gpus="${GPUS:-}"

PY="${RWCITE_PYTHON:-${PY:-python3}}"
if [[ -x "${ROOT}/.venv/bin/python" && -z "${RWCITE_PYTHON:-}" && "${PY}" == "python3" ]]; then
  PY="${ROOT}/.venv/bin/python"
fi
module load cuda-12.8 2>/dev/null || true

# Release 3.1: ewm|sqc|gw|fno|wsi|cosmo|qopt|sce|hep|exo|radio|radseg|driving → nofuture struct lane via lib_domain_env.
_use_nofuture_lane=0
case "$DOMAIN" in
  ewm|sqc|gw|fno|wsi|cosmo|qopt|sce|hep|exo|radio|radseg|driving)
    # shellcheck source=scripts/lib_domain_env.sh
    source "$ROOT/scripts/lib_domain_env.sh"
    rwcite_load_domain_env
    _use_nofuture_lane=1
    ;;
esac

GPUS="${_user_gpus:-${PACK_GPUS:-0,1,2,3}}"

if [[ "$SKIP_POOL_RANKER" == "1" ]]; then
  echo "== 0c) SKIP_POOL_RANKER=1 =="
  exit 0
fi

if [[ "$_use_nofuture_lane" == "1" ]]; then
  export RR_GEXF_OVERRIDE="${RR_GEXF_OVERRIDE:-$ROOT/$GEXF_NOFUTURE}"
  if [[ "${RR_GEXF_OVERRIDE}" != /* ]]; then
    export RR_GEXF_OVERRIDE="$ROOT/$RR_GEXF_OVERRIDE"
  fi
  # Always use nofuture model path for 3.1 lane. Inherited STRUCT_MODEL from a
  # prior domain env load must not redirect fit to ranker/struct/.
  STRUCT_MODEL="${STRUCT_MODEL_OVERRIDE:-$STRUCT_NOFUTURE_MODEL}"
  STRUCT_FEATS="${STRUCT_FEATS_OVERRIDE:-${_user_struct_feats:-$LANE/ranker/struct_nofuture/feats}}"
  TRAIN_JSONL="${_user_train_jsonl:-$JSONL_DIR/train.jsonl}"
  export RR_STRUCT_MASK_QUERY_OUT="${RR_STRUCT_MASK_QUERY_OUT:-1}"
  export RR_STRUCT_MASK_FUTURE_OUT="${RR_STRUCT_MASK_FUTURE_OUT:-1}"
  mkdir -p "$STRUCT_FEATS"
elif [[ -z "${_user_struct_model}" || -z "${_user_struct_feats}" || -z "${_user_train_jsonl}" ]]; then
  mapfile -t _paths < <("$PY" - <<PY
from pathlib import Path
from rwcite.graph.domain_paths import domain_layout, working_dir_for_domain

root = Path(".")
dom = "$DOMAIN"
wd = working_dir_for_domain(dom, root)
p = domain_layout(wd)
print(p["struct_model"])
print(p["struct_feats"])
print(p["jsonl_dir"] + "train.jsonl")
PY
  )
  STRUCT_MODEL="${_user_struct_model:-${_paths[0]}}"
  STRUCT_FEATS="${_user_struct_feats:-${_paths[1]}}"
  TRAIN_JSONL="${_user_train_jsonl:-${_paths[2]}}"
else
  STRUCT_MODEL="$_user_struct_model"
  STRUCT_FEATS="$_user_struct_feats"
  TRAIN_JSONL="$_user_train_jsonl"
fi

[[ -f "$TRAIN_JSONL" ]] || { echo "missing TRAIN_JSONL $TRAIN_JSONL" >&2; exit 1; }

IFS=',' read -r -a GPU_ARR <<< "$GPUS"
NUM_SHARDS="${POOL_RANKER_NUM_SHARDS:-${#GPU_ARR[@]}}"
LOG_DIR="${LOG_DIR:-logs/train_pool_ranker_${DOMAIN}}"

# Loss-aware fit defaults (override via POOL_RANKER_EPOCHS / POOL_RANKER_LR).
if [[ "$LOSS" == "logistic" ]]; then
  POOL_RANKER_EPOCHS="${POOL_RANKER_EPOCHS:-150}"
  POOL_RANKER_LR="${POOL_RANKER_LR:-0.25}"
else
  POOL_RANKER_EPOCHS="${POOL_RANKER_EPOCHS:-400}"
  POOL_RANKER_LR="${POOL_RANKER_LR:-0.05}"
fi

_abs_model() {
  readlink -f -- "$1"
}

if [[ -f "$STRUCT_MODEL" && "$FORCE" != "1" ]]; then
  echo "== 0c) reuse struct ranker $STRUCT_MODEL =="
  export RR_RANKER_PATH="$(_abs_model "$STRUCT_MODEL")"
  exit 0
fi

mkdir -p "$(dirname "$STRUCT_MODEL")" "$STRUCT_FEATS" "$LOG_DIR"

_feats_complete() {
  local _si
  for ((_si = 0; _si < NUM_SHARDS; _si++)); do
    [[ -f "$STRUCT_FEATS/features_shard${_si}of${NUM_SHARDS}.npz" ]] || return 1
  done
}

need_dump=0
if [[ "$FORCE" == "1" ]]; then
  need_dump=1
elif ! _feats_complete; then
  need_dump=1
fi

if [[ "$need_dump" == "1" ]]; then
  echo "== 0c-a) dump struct features: domain=$DOMAIN shards=$NUM_SHARDS gpus=$GPUS with_emb=$WITH_EMB =="
  if [[ "$FORCE" == "1" ]]; then
    rm -f "$STRUCT_FEATS"/features_shard*of*.npz
  fi
  # Launch shards in waves when NUM_SHARDS > |GPUS|.
  wave=0
  while (( wave < NUM_SHARDS )); do
    shard_list=()
    for ((idx = 0; idx < ${#GPU_ARR[@]} && wave + idx < NUM_SHARDS; idx++)); do
      shard_list+=("$((wave + idx))")
    done
    SHARDS="$(IFS=,; echo "${shard_list[*]}")"
    echo "  wave shards=$SHARDS"
    DOMAIN="$DOMAIN" OUT_DIR="$STRUCT_FEATS" TRAIN_JSONL="$TRAIN_JSONL" \
      NUM_SHARDS="$NUM_SHARDS" SHARDS="$SHARDS" GPUS="$GPUS" \
      LOG_DIR="$LOG_DIR" MAX_SOURCES="${MAX_SOURCES:-0}" \
      WITH_EMB="$WITH_EMB" \
      bash scripts/run_train_pool_ranker_4gpu.sh
    wave=$((wave + ${#shard_list[@]}))
  done
  _feats_complete || { echo "ERROR: feature dump incomplete under $STRUCT_FEATS" >&2; exit 1; }
else
  echo "== 0c-a) reuse feature shards under $STRUCT_FEATS =="
fi

echo "== 0c-b fit struct ranker loss=$LOSS -> $STRUCT_MODEL =="
"$PY" -m rwcite.cli.train_pool_ranker \
  --stage fit \
  --domain "$DOMAIN" \
  --train-jsonl "$TRAIN_JSONL" \
  --out-dir "$STRUCT_FEATS" \
  --out "$STRUCT_MODEL" \
  --loss "$LOSS" \
  --dev-frac "${POOL_RANKER_DEV_FRAC:-0.15}" \
  --epochs "$POOL_RANKER_EPOCHS" \
  --lr "$POOL_RANKER_LR" \
  --pairs-per-query "${POOL_RANKER_PAIRS_PER_QUERY:-256}" \
  --seed "${POOL_RANKER_SEED:-0}"

export RR_RANKER_PATH="$(_abs_model "$STRUCT_MODEL")"
echo "== 0c) struct ranker ready RR_RANKER_PATH=$RR_RANKER_PATH loss=$LOSS with_emb=$WITH_EMB =="
