#!/usr/bin/env bash
# Merge sharded train CE-sent caches → ce_scores_train3423_sent_s1.npz then L1 train/eval.
# Assumes shards already written by fuse_ce --start-query/--max-queries with stage1 CE.
# Override TRAIN_CE / shard glob for HN-era archives if needed.
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
export RR_CE_SENT_SHORT=1
export RR_CE_SENT_MAX_SENTS=3
export RR_CE_SENT_MAX_LEN=256

OUT=embodied_world_model_retrieval/ranker/gat_mvp
LOG=logs/gat_mvp
TRAIN_CE="${TRAIN_CE:-$OUT/ce_scores_train3423_sent_s1.npz}"
TEST_CE="${TEST_CE:-$OUT/ce_scores_test381_sent_s1.npz}"
SHARD_GLOB="${SHARD_GLOB:-ce_scores_train3423_sent_s1_shard*.npz}"
L0_REPORT="${L0_REPORT:-$OUT/ce_sent_s1_blend_report.json}"
mkdir -p "$LOG"

echo "MERGE train CE shards $(date) glob=$SHARD_GLOB -> $TRAIN_CE" | tee -a "$LOG/l1_cefeat.log"
SHARD_GLOB="$SHARD_GLOB" TRAIN_CE="$TRAIN_CE" $PY - <<'PY'
import json, os
from pathlib import Path
import numpy as np
out = Path("embodied_world_model_retrieval/ranker/gat_mvp")
pattern = os.environ.get("SHARD_GLOB", "ce_scores_train3423_sent_s1_shard*.npz")
train_ce = Path(os.environ.get("TRAIN_CE", str(out / "ce_scores_train3423_sent_s1.npz")))
shards = sorted(out.glob(pattern))
if len(shards) < 1:
    # fallback HN-era shard names
    shards = sorted(out.glob("ce_scores_train3423_sent_shard*.npz"))
if len(shards) < 1:
    raise SystemExit(f"no shards matching {pattern} in {out}")
qids = []
scores = []
meta = None
for p in shards:
    d = np.load(p, allow_pickle=True)
    qids.extend([str(x) for x in d["query_ids"].tolist()])
    scores.append(d["ce_scores"])
    meta = d
    print(f"  {p.name}: n={d['ce_scores'].shape[0]}")
sc = np.concatenate(scores, axis=0)
assert sc.shape[0] == len(qids)
# order by windows_train3423 query order
win = np.load(out / "windows_train3423.npz", allow_pickle=True)
order = {qid: i for i, qid in enumerate(qids)}
w = sc.shape[1]
aligned = np.full((len(win["query_ids"]), w), np.nan, dtype=np.float32)
aq = []
miss = 0
for i, qid in enumerate(win["query_ids"]):
    qid = str(qid)
    aq.append(qid)
    j = order.get(qid)
    if j is None:
        miss += 1
        continue
    aligned[i] = sc[j]
if miss:
    raise SystemExit(f"missing {miss} queries in shards")
dest = train_ce
np.savez_compressed(
    dest,
    query_ids=np.asarray(aq, dtype=object),
    ce_scores=aligned,
    windows=str(out / "windows_train3423.npz"),
    ce_path=str(meta["ce_path"]) if "ce_path" in meta else "",
    gexf=str(meta["gexf"]) if "gexf" in meta else "",
    mode=str(meta["mode"]) if "mode" in meta else "ce_sent",
    short=np.asarray([1]),
    max_sents=np.asarray([3]),
    max_length=np.asarray([256]),
)
print(json.dumps({"n": int(aligned.shape[0]), "path": str(dest), "miss": miss}, indent=2))
PY

if [[ ! -f "$TEST_CE" ]]; then
  echo "missing $TEST_CE" | tee -a "$LOG/l1_cefeat.log"
  exit 1
fi

echo "TRAIN L1 cefeat $(date) GPU=${CUDA_VISIBLE_DEVICES}" | tee -a "$LOG/l1_cefeat.log"
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
cands = list(eval_dir.glob("ewm_gat_mvp_test381_cefeat*_merged.json"))
cand = cands[0] if cands else None
l0 = Path(os.environ.get("L0_REPORT") or out / "ce_sent_s1_blend_report.json")
report = {"ckpt": "ckpt_e4_cefeat.pt", "ce_arm": "stage1_nofuture"}
if base.is_file() and cand and cand.is_file():
    cmp = compare(base, cand, bar=BASELINE_FULL381_HITS_AT_10)
    (out / "compare_cefeat_vs_e4.json").write_text(json.dumps(cmp, indent=2))
    report["vs_e4"] = cmp
    report["mean_hits_at_10"] = (json.loads(cand.read_text()).get("summary") or {}).get("mean_hits_at_10")
    report["eval_path"] = str(cand)
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
