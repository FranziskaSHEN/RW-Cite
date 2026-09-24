#!/usr/bin/env bash
# L1: CE-sent scalar into L5 MLP; freeze GNN; init from R2 E4.
# Gate: only run after CE-sent L0 best @10 > 3.480.
#
# NOTE (2026-09-10): defaults follow Release 3.0 stage1 CE arm (HN retired for L0).
# Historical HN-negative L1 result remains in route_decision / RELEASE; re-running
# this script scores stage1 CE features — not a replay of the archived HN L1.
set -euo pipefail
ROOT="${RWCITE_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$ROOT"
PY="${RWCITE_PYTHON:-python3}"
if [[ -x "${ROOT}/.venv/bin/python" && -z "${RWCITE_PYTHON:-}" && "${PY}" == "python3" ]]; then
  PY="${ROOT}/.venv/bin/python"
fi
export RWCITE_ROOT="$ROOT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1
export RR_GEXF_OVERRIDE="${RR_GEXF_OVERRIDE:-$ROOT/embodied_world_model_retrieval/description/test_graph_rr.o2.nofuture.gexf}"
export RR_CE_SENT_SHORT=1
export RR_CE_SENT_MAX_SENTS=3
export RR_CE_SENT_MAX_LEN=256
export RR_CE_SENT_CITELINK_BLEND=0

OUT=embodied_world_model_retrieval/ranker/gat_mvp
LOG=logs/gat_mvp
CE_PATH="${CE_PATH:-$ROOT/embodied_world_model_retrieval/ranker/ce_sent/stage1_nofuture}"
TRAIN_CE="${TRAIN_CE:-$OUT/ce_scores_train3423_sent_s1.npz}"
TEST_CE="${TEST_CE:-$OUT/ce_scores_test381_sent_s1.npz}"
L0_REPORT="${L0_REPORT:-$OUT/ce_sent_s1_blend_report.json}"
mkdir -p "$LOG"

# Gate: best L0 > E4 (stage1 report)
if [[ -f "$L0_REPORT" ]]; then
  L0_REPORT="$L0_REPORT" $PY - <<'PY'
import json, os, sys
from pathlib import Path
r = json.loads(Path(os.environ["L0_REPORT"]).read_text())
ok = bool(r.get("beats_e4")) and float(r.get("best_hits_at_10") or 0) > 3.480
print(f"L0 gate report={os.environ['L0_REPORT']} alpha0={r.get('alpha0_hits_at_10')} "
      f"beats_e4={r.get('beats_e4')} best={r.get('best_hits_at_10')}")
sys.exit(0 if ok else 2)
PY
fi

echo "START L1 cefeat $(date) GPU=${CUDA_VISIBLE_DEVICES} CE=${CE_PATH}" | tee -a "$LOG/l1_cefeat.log"

if [[ ! -f "$TRAIN_CE" ]]; then
  echo "cache train CE-sent $(date)" | tee -a "$LOG/l1_cefeat.log"
  $PY -u -m rwcite.gat.fuse_ce --step cache_ce --mode ce_sent --short 1 --max-sents 3 \
    --force-cache \
    --windows "$OUT/windows_train3423.npz" \
    --ce-path "$CE_PATH" \
    --ce-cache "$TRAIN_CE" \
    2>&1 | tee -a "$LOG/l1_cefeat.log"
fi

if [[ ! -f "$TEST_CE" ]]; then
  echo "missing $TEST_CE — run CE-sent L0 blend first (stage1)" | tee -a "$LOG/l1_cefeat.log"
  exit 1
fi

$PY -u -m rwcite.gat.train \
  --ablation E4 \
  --use-ce-feat --freeze-gnn --freeze-struct \
  --init-ckpt "$OUT/ckpt_e4.pt" \
  --ce-cache "$TRAIN_CE" \
  --windows "$OUT/windows_train3423.npz" \
  --epochs 15 --min-epochs 3 --patience 3 \
  --ckpt-name ckpt_e4_cefeat.pt \
  --report-name train_report_e4_cefeat.json \
  2>&1 | tee -a "$LOG/train_e4_cefeat.log"

echo "EVAL L1 cefeat $(date)" | tee -a "$LOG/l1_cefeat.log"
$PY -u -m rwcite.gat.evaluate \
  --ablation E4 \
  --ckpt "$OUT/ckpt_e4_cefeat.pt" \
  --ce-cache "$TEST_CE" \
  --out-tag ewm_gat_mvp_test381_cefeat \
  --compare-name compare_e4_cefeat.json \
  2>&1 | tee -a "$LOG/eval_e4_cefeat.log"

L0_REPORT="$L0_REPORT" $PY - <<'PY' 2>&1 | tee -a "$LOG/l1_cefeat.log"
import json, os
from pathlib import Path
from rwcite.gat.compare import compare
from rwcite.gat import BASELINE_FULL381_HITS_AT_10
root = Path(".")
out = root / "embodied_world_model_retrieval/ranker/gat_mvp"
eval_dir = root / "embodied_world_model_retrieval/ranker/eval"
base = eval_dir / "ewm_gat_mvp_test381_e4_merged.json"
cand = eval_dir / "ewm_gat_mvp_test381_cefeat_e4_merged.json"
if not cand.is_file():
    for p in eval_dir.glob("ewm_gat_mvp_test381_cefeat*_merged.json"):
        cand = p
        break
l0 = Path(os.environ.get("L0_REPORT") or out / "ce_sent_s1_blend_report.json")
report = {"ckpt": "ckpt_e4_cefeat.pt", "ce_arm": "stage1_nofuture"}
if base.is_file() and cand.is_file():
    cmp = compare(base, cand, bar=BASELINE_FULL381_HITS_AT_10)
    (out / "compare_cefeat_vs_e4.json").write_text(json.dumps(cmp, indent=2))
    report["vs_e4"] = cmp
    report["mean_hits_at_10"] = (json.loads(cand.read_text()).get("summary") or {}).get("mean_hits_at_10")
if l0.is_file():
    r = json.loads(l0.read_text())
    report["best_l0"] = r.get("best_hits_at_10")
    report["l1_beats_l0"] = bool(
        report.get("mean_hits_at_10") is not None
        and float(report["mean_hits_at_10"]) > float(r.get("best_hits_at_10") or 0)
    )
    report["prefer"] = "L1" if report.get("l1_beats_l0") else "L0_blend"
(out / "cefeat_l1_report.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
PY

echo "L1 cefeat DONE $(date)" | tee -a "$LOG/l1_cefeat.log"
