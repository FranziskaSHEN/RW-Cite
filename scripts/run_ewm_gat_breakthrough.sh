#!/usr/bin/env bash
# GAT 续冲: folds → E4 (+freeze) → F5/E5/E3 → fair ablations → optional R4
set -euo pipefail
ROOT="${RWCITE_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$ROOT"
PY="${RWCITE_PYTHON:-python3}"
if [[ -x "${ROOT}/.venv/bin/python" && -z "${RWCITE_PYTHON:-}" && "${PY}" == "python3" ]]; then
  PY="${ROOT}/.venv/bin/python"
fi
export RWCITE_ROOT="$ROOT" PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
OUT="$ROOT/embodied_world_model_retrieval/ranker/gat_mvp"
LOG="$ROOT/logs/gat_mvp"
mkdir -p "$LOG"

STEP="${1:-all}"

run_folds() {
  echo "== R1 folds =="
  "$PY" -m rwcite.gat.preprocess --step folds
}

run_e4() {
  echo "== R2 E4 B-clean =="
  "$PY" -m rwcite.gat.train --ablation E4 --epochs 20 --min-epochs 5 --patience 4 \
    --ckpt-name ckpt_e4.pt --report-name train_report_e4.json "$@"
  "$PY" -m rwcite.gat.evaluate --ablation E4 --ckpt "$OUT/ckpt_e4.pt"
}

run_e4_freeze() {
  echo "== R2 E4 freeze-struct =="
  "$PY" -m rwcite.gat.train --ablation E4 --freeze-struct --epochs 20 --min-epochs 5 --patience 4 \
    --ckpt-name ckpt_e4_freeze_struct.pt --report-name train_report_e4_freeze.json "$@"
}

run_f5() {
  echo "== R3 F5 independent =="
  "$PY" -m rwcite.gat.train --ablation F5 --epochs 20 --min-epochs 5 --patience 4 \
    --ckpt-name ckpt_f5.pt --report-name train_report_f5.json "$@"
}

run_e5() {
  echo "== R3 E5 independent =="
  "$PY" -m rwcite.gat.train --ablation E5 --epochs 20 --min-epochs 5 --patience 4 \
    --ckpt-name ckpt_e5.pt --report-name train_report_e5.json "$@"
}

run_e3() {
  echo "== R3 true E3 struct-only =="
  "$PY" -m rwcite.gat.train --ablation E3 --struct-only-train --epochs 20 --min-epochs 5 --patience 4 \
    --ckpt-name ckpt_e3.pt --report-name train_report_e3.json "$@"
}

run_fair() {
  echo "== R3 fair ablations =="
  "$PY" -m rwcite.gat.ablations --runs E1,E2,E3,E4,E5,F5 --ckpt-e4 "$OUT/ckpt_e4.pt"
}

run_r4() {
  echo "== R4 enrich hop=1 =="
  "$PY" -m rwcite.gat.train --ablation E4 --expand-hop 1 --epochs 15 --min-epochs 4 --patience 3 \
    --ckpt-name ckpt_e4_r4.pt --report-name train_report_e4_r4.json "$@"
  "$PY" -m rwcite.gat.evaluate --ablation E4 --ckpt "$OUT/ckpt_e4_r4.pt" --expand-hop 1 --out-tag ewm_gat_mvp_test381_r4
}

case "$STEP" in
  folds) run_folds ;;
  e4) run_e4 ;;
  e4_freeze) run_e4_freeze ;;
  f5) run_f5 ;;
  e5) run_e5 ;;
  e3) run_e3 ;;
  fair) run_fair ;;
  r4) run_r4 ;;
  all)
    run_folds
    run_e4
    run_e4_freeze
    run_f5
    run_e5
    run_e3
    run_fair
    # R4 if fair route says mixed_continue_r4 or F5 inconclusive
    if python -c "import json; d=json.load(open('$OUT/route_decision.json')); print(d.get('decision',''))" | grep -Eq 'mixed_continue_r4|partial'; then
      run_r4
      # re-eval E4 r4 into route note
    fi
    ;;
  *)
    echo "usage: $0 {folds|e4|e4_freeze|f5|e5|e3|fair|r4|all}" >&2
    exit 2
    ;;
esac
echo "OK step=$STEP"
