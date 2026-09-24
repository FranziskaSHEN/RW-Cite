#!/usr/bin/env bash
# Promote ewm nofuture graph models into production path names.
# Prerequisites: domains.yaml already points at test_graph_rr.o2.nofuture.gexf
# Usage: bash scripts/cutover_ewm_nofuture_production.sh
set -euo pipefail
ROOT="${RWCITE_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$ROOT"
TS="${TS:-$(date +%Y%m%d)}"
WD=embodied_world_model_retrieval

[[ -d "$WD/ranker/struct_nofuture" ]] || { echo "missing struct_nofuture" >&2; exit 1; }
[[ -d "$WD/ranker/citelink_nofuture" ]] || { echo "missing citelink_nofuture" >&2; exit 1; }
[[ -d "$WD/ranker/ce_sent/c2s-hn-scibert_scivocab_uncased_nofuture" ]] || {
  echo "missing ce nofuture" >&2; exit 1
}
[[ -f "$WD/description/test_graph_rr.o2.nofuture.gexf" ]] || {
  echo "missing nofuture gexf" >&2; exit 1
}

echo "== backup leaky full-graph production → *_pre_nofuture_${TS} =="
[[ -e "$WD/ranker/struct_pre_nofuture_$TS" ]] || mv "$WD/ranker/struct" "$WD/ranker/struct_pre_nofuture_$TS"
[[ -e "$WD/ranker/citelink_pre_nofuture_$TS" ]] || mv "$WD/ranker/citelink" "$WD/ranker/citelink_pre_nofuture_$TS"
[[ -e "$WD/ranker/ce_sent/c2s-hn-scibert_scivocab_uncased_pre_nofuture_$TS" ]] || \
  mv "$WD/ranker/ce_sent/c2s-hn-scibert_scivocab_uncased" \
     "$WD/ranker/ce_sent/c2s-hn-scibert_scivocab_uncased_pre_nofuture_$TS"

echo "== promote nofuture → production names =="
cp -a "$WD/ranker/struct_nofuture" "$WD/ranker/struct"
cp -a "$WD/ranker/citelink_nofuture" "$WD/ranker/citelink"
cp -a "$WD/ranker/ce_sent/c2s-hn-scibert_scivocab_uncased_nofuture" \
      "$WD/ranker/ce_sent/c2s-hn-scibert_scivocab_uncased"

if [[ -f "$WD/ranker/eval/c2s_report_scibert_scivocab_uncased.json" && \
      ! -f "$WD/ranker/eval/c2s_report_scibert_scivocab_uncased.pre_nofuture_$TS.json" ]]; then
  cp -a "$WD/ranker/eval/c2s_report_scibert_scivocab_uncased.json" \
        "$WD/ranker/eval/c2s_report_scibert_scivocab_uncased.pre_nofuture_$TS.json"
fi

python3 - <<PY
import hashlib, json
from pathlib import Path
from datetime import datetime, timezone
wd = Path("$WD")
nf = wd / "description/test_graph_rr.o2.nofuture.gexf"
sha = hashlib.sha256(nf.read_bytes()).hexdigest()
src = json.loads((wd / "ranker/eval/c2s_report_nofuture_scibert_scivocab_uncased.json").read_text())
src["gexf"] = "embodied_world_model_retrieval/description/test_graph_rr.o2.nofuture.gexf"
src["gexf_sha256"] = sha
src["protocol_note"] = (
    "nofuture cutover: strip test∪frontier out-edges; admit split/jsonl frozen from full O2. "
    "Closes citation-edge leakage that inflated CE scores on the full graph."
)
src["cutover_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
full = json.loads((wd / "ranker/eval/ewm_v5_scibert_c2s_nofuture_test381_merged.json").read_text())
src["full_admit_test"] = full.get("summary")
out = wd / "ranker/eval/c2s_report_scibert_scivocab_uncased.json"
out.write_text(json.dumps(src, indent=2) + "\n", encoding="utf-8")
print("report", out, "sha", sha[:24], "n80", src["hn_pure_ce"]["mean_hits_at_10"])
PY

# Ensure domains.yaml points at nofuture (idempotent)
python3 - <<'PY'
from pathlib import Path
p = Path("configs/domains.yaml")
t = p.read_text()
old = "gexf: embodied_world_model_retrieval/description/test_graph_rr.o2.gexf"
new = "gexf: embodied_world_model_retrieval/description/test_graph_rr.o2.nofuture.gexf"
if old in t and new not in t:
    t = t.replace(old, new, 1)
    p.write_text(t)
    print("domains.yaml → nofuture")
else:
    print("domains.yaml already nofuture or unexpected")
PY

echo "== done =="
echo "  gexf default: $WD/description/test_graph_rr.o2.nofuture.gexf"
echo "  backups: struct/citelink/ce *_pre_nofuture_$TS"
echo "  Do NOT re-dump admit on nofuture (test/frontier outdeg=0); keep existing splits/jsonl."
