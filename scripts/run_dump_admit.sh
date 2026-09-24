#!/usr/bin/env bash
# Dump the frozen query split and recommendation JSONL for an enabled domain.
# F defaults: gold outdeg q[0.10, 0.99]. Backs up prior splits/jsonl.
#
#   DOMAIN=ewm bash scripts/run_dump_admit.sh
#   DOMAIN=sqc bash scripts/run_dump_admit.sh
set -euo pipefail
ROOT="${RWCITE_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$ROOT"
export RWCITE_ROOT="$ROOT"
PY="${RWCITE_PYTHON:-${PY:-python3}}"
if [[ -x "${ROOT}/.venv/bin/python" && -z "${RWCITE_PYTHON:-}" && "${PY}" == "python3" ]]; then
  PY="${ROOT}/.venv/bin/python"
fi
# shellcheck source=scripts/lib_domain_env.sh
source "$ROOT/scripts/lib_domain_env.sh"
rwcite_load_domain_env

GEXF="$GEXF_O2"
RET="$RET_NODES"
[[ -f "$GEXF" ]] || { echo "ERROR: missing full O2 $GEXF" >&2; exit 1; }
[[ -f "$RET" ]] || { echo "ERROR: missing $RET" >&2; exit 1; }

case "$SPLIT_DIR$JSONL_DIR" in
  *embodied_world_model_retrieval_rw*) echo "ERROR: refused ewm_rw path" >&2; exit 1 ;;
esac

TS="$(date +%Y%m%d_%H%M%S)"
BAK_ROOT="$LANE/data/_admit_backup_${TS}"
if [[ -d "$SPLIT_DIR" || -d "$JSONL_DIR" ]]; then
  echo "== backup prior admit → $BAK_ROOT =="
  mkdir -p "$BAK_ROOT"
  [[ -d "$SPLIT_DIR" ]] && mv "$SPLIT_DIR" "$BAK_ROOT/splits"
  [[ -d "$JSONL_DIR" ]] && mv "$JSONL_DIR" "$BAK_ROOT/reference_recommend"
fi
mkdir -p "$SPLIT_DIR" "$JSONL_DIR"

echo "== dump test_admit_v1 (F q[0.10,0.99]) DOMAIN=$DOMAIN CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} =="
"$PY" -m rwcite.cli.dump_split \
  --gexf "$GEXF" \
  --retrieval-nodes "$RET" \
  --out-dir "$SPLIT_DIR"

echo "== build RR jsonl → $JSONL_DIR =="
"$PY" -m rwcite.cli.build_rr_jsonl \
  --gexf "$GEXF" \
  --split-file "$SPLIT_DIR" \
  --out-dir "$JSONL_DIR" \
  --which both

rwcite_load_domain_env
echo "== done DOMAIN=$DOMAIN train/test/front=$N_TRAIN/$N_TEST/$N_FRONTIER =="
"$PY" -c "
import json
m=json.load(open('$SPLIT_DIR/split_meta.json'))
print(m['split_id'], m.get('gold_quantile_lo'), m.get('gold_quantile_hi'),
      m['n_train_sources'], m['n_test_sources'], m.get('cutoff_source'),
      m.get('n_drop_gold_quantile'), m.get('gold_outdeg_lo'), m.get('gold_outdeg_hi'),
      m.get('n_frontier_sources'))
"
