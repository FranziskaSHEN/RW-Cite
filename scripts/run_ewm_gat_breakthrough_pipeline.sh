#!/usr/bin/env bash
set -euo pipefail
ROOT="${RWCITE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1
export RWCITE_ROOT="$ROOT"
PY=python
OUT=embodied_world_model_retrieval/ranker/gat_mvp
LOG=logs/gat_mvp
mkdir -p "$LOG"

echo "START E4 $(date)" | tee -a "$LOG/breakthrough_all.log"
$PY -u -m rwcite.gat.train --ablation E4 --epochs 20 --min-epochs 5 --patience 4 \
  --ckpt-name ckpt_e4.pt --report-name train_report_e4.json 2>&1 | tee -a "$LOG/train_e4.log"

echo "EVAL E4 $(date)" | tee -a "$LOG/breakthrough_all.log"
$PY -u -m rwcite.gat.evaluate --ablation E4 --ckpt "$OUT/ckpt_e4.pt" 2>&1 | tee -a "$LOG/eval_e4.log"

echo "START E4 freeze $(date)" | tee -a "$LOG/breakthrough_all.log"
$PY -u -m rwcite.gat.train --ablation E4 --freeze-struct --epochs 20 --min-epochs 5 --patience 4 \
  --ckpt-name ckpt_e4_freeze_struct.pt --report-name train_report_e4_freeze.json 2>&1 | tee -a "$LOG/train_e4_freeze.log"

echo "START F5 $(date)" | tee -a "$LOG/breakthrough_all.log"
$PY -u -m rwcite.gat.train --ablation F5 --epochs 20 --min-epochs 5 --patience 4 \
  --ckpt-name ckpt_f5.pt --report-name train_report_f5.json 2>&1 | tee -a "$LOG/train_f5.log"

echo "START E5 $(date)" | tee -a "$LOG/breakthrough_all.log"
$PY -u -m rwcite.gat.train --ablation E5 --epochs 20 --min-epochs 5 --patience 4 \
  --ckpt-name ckpt_e5.pt --report-name train_report_e5.json 2>&1 | tee -a "$LOG/train_e5.log"

echo "START E3 $(date)" | tee -a "$LOG/breakthrough_all.log"
$PY -u -m rwcite.gat.train --ablation E3 --struct-only-train --epochs 20 --min-epochs 5 --patience 4 \
  --ckpt-name ckpt_e3.pt --report-name train_report_e3.json 2>&1 | tee -a "$LOG/train_e3.log"

echo "FAIR ABLATIONS $(date)" | tee -a "$LOG/breakthrough_all.log"
$PY -u -m rwcite.gat.ablations --runs E1,E2,E3,E4,E5,F5 --ckpt-e4 "$OUT/ckpt_e4.pt" 2>&1 | tee -a "$LOG/fair_ablations.log"

DEC=$(python -c "import json; print(json.load(open('$OUT/route_decision.json')).get('decision',''))")
echo "decision=$DEC" | tee -a "$LOG/breakthrough_all.log"

if echo "$DEC" | grep -Eq 'mixed_continue_r4|partial|plan_b_or_simplify|promote_gat'; then
  # Always run R4 enrich when continuing GAT or inconclusive; skip only if drop-GNN finalized
  true
fi
# R4 when gate is mixed / simplify / promote (strengthen) or keep_struct (still try enrich once)
if echo "$DEC" | grep -Eq 'mixed_continue_r4|partial|plan_b_or_simplify|promote_gat|insufficient'; then
  echo "START R4 $(date)" | tee -a "$LOG/breakthrough_all.log"
  $PY -u -m rwcite.gat.train --ablation E4 --expand-hop 1 --epochs 15 --min-epochs 4 --patience 3 \
    --ckpt-name ckpt_e4_r4.pt --report-name train_report_e4_r4.json 2>&1 | tee -a "$LOG/train_e4_r4.log"
  $PY -u -m rwcite.gat.evaluate --ablation E4 --ckpt "$OUT/ckpt_e4_r4.pt" --expand-hop 1 \
    --out-tag ewm_gat_mvp_test381_r4 2>&1 | tee -a "$LOG/eval_e4_r4.log"
  python - <<'PY'
import json
from pathlib import Path
out = Path("embodied_world_model_retrieval/ranker/gat_mvp")
route = json.loads((out / "route_decision.json").read_text())
p = Path("embodied_world_model_retrieval/ranker/eval/ewm_gat_mvp_test381_r4_e4_merged.json")
if p.is_file():
    ev = json.loads(p.read_text())
    route["r4"] = ev["summary"]
    route["r4_note"] = "expand_hop=1 E4 after fair gate"
    (out / "route_decision.json").write_text(json.dumps(route, indent=2))
    print("r4 @10", ev["summary"].get("mean_hits_at_10"))
PY
fi

echo "PIPELINE DONE $(date)" | tee -a "$LOG/breakthrough_all.log"
