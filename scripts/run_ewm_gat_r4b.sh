#!/usr/bin/env bash
# R4b: freeze-struct + L4 train top-64 / eval top-32 (no expand_hop).
# Parallel to R4 expand_hop on another GPU. Does NOT attach to breakthrough_pipeline.
set -euo pipefail
ROOT="${RWCITE_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$ROOT"
PY="${RWCITE_PYTHON:-python3}"
if [[ -x "${ROOT}/.venv/bin/python" && -z "${RWCITE_PYTHON:-}" && "${PY}" == "python3" ]]; then
  PY="${ROOT}/.venv/bin/python"
fi
export RWCITE_ROOT="$ROOT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export PYTHONUNBUFFERED=1
export RR_GEXF_OVERRIDE="${RR_GEXF_OVERRIDE:-$ROOT/embodied_world_model_retrieval/description/test_graph_rr.o2.nofuture.gexf}"

OUT=embodied_world_model_retrieval/ranker/gat_mvp
LOG=logs/gat_mvp
mkdir -p "$LOG"

echo "START R4b $(date) GPU=${CUDA_VISIBLE_DEVICES}" | tee -a "$LOG/r4b_all.log"

$PY -u -m rwcite.gat.train \
  --ablation E4 \
  --freeze-struct \
  --struct-residual-scale 1.0 \
  --l4-train-k 64 \
  --l4-eval-k 32 \
  --expand-hop 0 \
  --epochs 20 --min-epochs 5 --patience 4 \
  --ckpt-name ckpt_e4_r4b.pt \
  --report-name train_report_e4_r4b.json \
  2>&1 | tee -a "$LOG/train_e4_r4b.log"

echo "EVAL R4b $(date)" | tee -a "$LOG/r4b_all.log"
$PY -u -m rwcite.gat.evaluate \
  --ablation E4 \
  --ckpt "$OUT/ckpt_e4_r4b.pt" \
  --l4-eval-k 32 \
  --out-tag ewm_gat_mvp_test381_r4b \
  --compare-name compare_e4_r4b.json \
  --dump-scores \
  2>&1 | tee -a "$LOG/eval_e4_r4b.log"

# vs R2 E4 (if present)
if [[ -f "$OUT/ckpt_e4.pt" ]]; then
  $PY - <<'PY' 2>&1 | tee -a "$LOG/r4b_all.log"
import json
from pathlib import Path
from rwcite.gat.compare import compare
from rwcite.gat import BASELINE_FULL381_HITS_AT_10
root = Path(".")
base = root / "embodied_world_model_retrieval/ranker/eval/ewm_gat_mvp_test381_e4_merged.json"
cand = root / "embodied_world_model_retrieval/ranker/eval/ewm_gat_mvp_test381_r4b_e4_merged.json"
if base.is_file() and cand.is_file():
    cmp = compare(base, cand, bar=BASELINE_FULL381_HITS_AT_10)
    out = root / "embodied_world_model_retrieval/ranker/gat_mvp/compare_r4b_vs_e4.json"
    out.write_text(json.dumps(cmp, indent=2))
    print("r4b_vs_e4", json.dumps(cmp, indent=2))
PY
fi

echo "R4b DONE $(date)" | tee -a "$LOG/r4b_all.log"
