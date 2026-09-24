#!/usr/bin/env bash
# Sweep the CE shortlist window on the ewm_rw dev split and print hits@10 /
# pool_over_u per setting. Parameters get chosen here, never on the test split.
#
# Only the window is swept: max_sents and max_len are baked into the CE's
# training text, so changing them at inference time breaks the CE.
#
# Usage:
#   GEN=p3 CE=embodied_world_model_retrieval_rw/ranker/ce_sent/c2s-hn-scibert_scivocab_uncased.p3 \
#     STRUCT_MODEL=models/rr-pool-ranker-ewm-rw-p3/model.npz \
#     WINDOWS="400 600 800" scripts/sweep_ewm_rw_window.sh
set -euo pipefail
ROOT="${RWCITE_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$ROOT"
export RWCITE_ROOT="$ROOT"

LANE="embodied_world_model_retrieval_rw"
GEN="${GEN:?GEN required}"
CE="${CE:?CE required}"
STRUCT_MODEL="${STRUCT_MODEL:?STRUCT_MODEL required}"
WINDOWS="${WINDOWS:-400 600 800}"
DEV_JSONL="${DEV_JSONL:-$LANE/data/reference_recommend_$GEN/dev400.jsonl}"
SAMPLES="${SAMPLES:-400}"
GPUS="${GPUS:-0,1,2,3}"

for win in $WINDOWS; do
  tag="ewm_rw_${GEN}_devwin${win}"
  echo "== window=$win -> $tag =="
  TAG="$tag" MAX_SAMPLES="$SAMPLES" GPUS="$GPUS" \
    TEST_JSONL="$DEV_JSONL" \
    RANKER_CE_SENT="$CE" \
    STRUCT_MODEL="$STRUCT_MODEL" \
    RR_CE_SENT_WINDOW="$win" \
    scripts/run_eval_ewm_rw.sh >"logs/${tag}.log" 2>&1
  "${RWCITE_PYTHON:-.venv/bin/python}" - "$LANE/ranker/eval/${tag}_merged.json" "$win" <<'PY'
import json, sys
s = json.loads(open(sys.argv[1]).read())
s = s.get("summary", s)
print(
    f"  window={sys.argv[2]} hits@10={s['mean_hits_at_10']:.4f} "
    f"hits@30={s['mean_hits_at_30']:.4f} pool_over_u={s['mean_pool_over_u']:.4f} "
    f"U={s['mean_universe_recall']:.4f} n={s['n_samples']}"
)
PY
done
