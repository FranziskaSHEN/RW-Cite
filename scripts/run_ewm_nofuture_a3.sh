#!/usr/bin/env bash
# Minimal A3 on ewm nofuture: paired A0 (HN) + A3 (stage1) under fair masks.
# Usage:
#   GPUS=0,1,2,3 bash scripts/run_ewm_nofuture_a3.sh
#   MAX_SAMPLES=381 bash scripts/run_ewm_nofuture_a3.sh   # after n80 |Δ|≥0.15
set -euo pipefail
ROOT="${RWCITE_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$ROOT"
export RWCITE_ROOT="$ROOT"

MAX_SAMPLES="${MAX_SAMPLES:-80}"
GPUS="${GPUS:-0,1,2,3}"

echo "== A0 HN (nofuture fair) max_samples=$MAX_SAMPLES =="
bash scripts/run_ablation_2.0.sh --domain ewm --nofuture --cell A0 \
  --max-samples "$MAX_SAMPLES" --gpus "$GPUS"

echo "== A3 stage1 (nofuture fair) max_samples=$MAX_SAMPLES =="
bash scripts/run_ablation_2.0.sh --domain ewm --nofuture --cell A3 \
  --max-samples "$MAX_SAMPLES" --gpus "$GPUS"

tag="n80"
if [[ "$MAX_SAMPLES" == "381" ]]; then tag="full"; elif [[ "$MAX_SAMPLES" != "80" ]]; then tag="n${MAX_SAMPLES}"; fi
OUT=embodied_world_model_retrieval/ranker/eval
A0="$OUT/abl2_ewm_nf_A0_${tag}_merged.json"
A3="$OUT/abl2_ewm_nf_A3_${tag}_merged.json"
python3 - <<PY
import json
from pathlib import Path
a0 = json.loads(Path("$A0").read_text())["summary"]["mean_hits_at_10"]
a3 = json.loads(Path("$A3").read_text())["summary"]["mean_hits_at_10"]
d = a3 - a0
print(f"A0_HN @10={a0:.4f}")
print(f"A3_stage1 @10={a3:.4f}")
print(f"Δ@10={d:+.4f}  (≤-0.15 keep stage2; |Δ|<0.15 noise; ≥+0.15 investigate)")
PY
