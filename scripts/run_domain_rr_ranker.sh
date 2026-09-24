#!/usr/bin/env bash
# Ablation only: time_elig10 → RR jsonl → CE pairs → train WITHOUT citelink HN → pure-CE n80.
# Release deploy path is scripts/run_domain_rr_c2s.sh (sclean + SciBERT + full citelink + C₂s).
# Same SciBERT cold-start as Release; only difference is no citelink HN stage.
#
# Usage:
#   bash scripts/run_domain_rr_ranker.sh --domain <tag>
#   bash scripts/run_domain_rr_ranker.sh --domain <tag> --skip-split --skip-build-data
#   bash scripts/run_domain_rr_ranker.sh --domain <tag> --eval-only
set -euo pipefail

ROOT="${RWCITE_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$ROOT"

DOMAIN="${DOMAIN:-}"
SKIP_SPLIT="${SKIP_SPLIT:-0}"
SKIP_BUILD_DATA="${SKIP_BUILD_DATA:-0}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
EVAL_ONLY="${EVAL_ONLY:-0}"
GPUS="${GPUS:-0,1,2,3}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --domain) DOMAIN="$2"; shift 2 ;;
    --skip-split) SKIP_SPLIT=1; shift ;;
    --skip-build-data) SKIP_BUILD_DATA=1; shift ;;
    --skip-train) SKIP_TRAIN=1; shift ;;
    --eval-only) EVAL_ONLY=1; SKIP_SPLIT=1; SKIP_BUILD_DATA=1; SKIP_TRAIN=1; shift ;;
    -h|--help)
      sed -n '2,9p' "$0"
      exit 0
      ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$DOMAIN" ]]; then
  echo "ERROR: --domain required (key in configs/domains.yaml)" >&2
  exit 2
fi

PY="${RWCITE_PYTHON:-${PY:-python3}}"
if [[ -x "${ROOT}/.venv/bin/python" && -z "${RWCITE_PYTHON:-}" && "${PY}" == "python3" ]]; then
  PY="${ROOT}/.venv/bin/python"
fi

module load cuda-12.8 2>/dev/null || true
export RWCITE_ROOT="$ROOT"

mapfile -t _paths < <("$PY" - <<PY
from pathlib import Path
from rwcite.graph.domain_paths import domain_layout, working_dir_for_domain

root = Path("$ROOT")
dom = "$DOMAIN"
wd = working_dir_for_domain(dom, root)
p = domain_layout(wd, base_tag="")
for k in (
    "root",
    "splits_dir",
    "jsonl_dir",
    "ablation_pairs",
    "ce_ablation",
    "eval_dir",
    "retrain_report",
):
    print(p[k])
PY
)
DOMAIN_ROOT="${_paths[0]}"
SPLIT_DIR="${_paths[1]}"
JSONL_DIR="${_paths[2]}"
PAIRS="${_paths[3]}"
OUT="${_paths[4]}"
EVAL_DIR="${_paths[5]}"
REPORT="${_paths[6]}"

mapfile -t _gexf < <("$PY" - <<PY
import yaml
from pathlib import Path
root = Path("$ROOT")
dom = "$DOMAIN"
block = yaml.safe_load((root / "configs/domains.yaml").read_text())["domains"][dom]
print(block["gexf"])
print(block.get("retrieval_nodes", ""))
PY
)
GEXF="${_gexf[0]}"
RET_NODES="${_gexf[1]}"
TAG_PURE="${DOMAIN}_v5_pure_n80"
BASE="${BASE:-models/base/scibert_scivocab_uncased}"
LOG_DIR="logs/rr_ltr_${DOMAIN}"

mkdir -p "$SPLIT_DIR" "$JSONL_DIR" "$LOG_DIR" "$(dirname "$PAIRS")" "$OUT" "$EVAL_DIR" "${DOMAIN_ROOT}data" "${DOMAIN_ROOT}ranker"

if [[ ! -f "$GEXF" ]]; then
  echo "ERROR: missing gexf $GEXF — run run_domain_graph_build.sh --domain $DOMAIN --cutover first" >&2
  exit 1
fi
if [[ ! -d "$BASE" ]]; then
  echo "ERROR: missing CE init base $BASE" >&2
  exit 1
fi

echo "== domain RR ranker: domain=${DOMAIN} base=${BASE} (no foreign ce-sent warm-start) =="
echo "  gexf:   $GEXF"
echo "  split:  $SPLIT_DIR"
echo "  jsonl:  $JSONL_DIR"
echo "  pairs:  $PAIRS"
echo "  out:    $OUT"

if [[ "$SKIP_SPLIT" != "1" ]]; then
  echo "== dump test_admit_v1 split =="
  "$PY" -m rwcite.cli.dump_split \
    --elig-k 10 \
    --test-frac 0.10 \
    --gexf "$GEXF" \
    --retrieval-nodes "$RET_NODES" \
    --out-dir "$SPLIT_DIR" \
    2>&1 | tee "$LOG_DIR/dump_split.log"
fi

if [[ "$SKIP_BUILD_DATA" != "1" ]]; then
  echo "== build RR jsonl =="
  "$PY" -m rwcite.cli.build_rr_jsonl \
    --gexf "$GEXF" \
    --split-file "$SPLIT_DIR" \
    --out-dir "$JSONL_DIR" \
    --which both \
    2>&1 | tee "$LOG_DIR/build_jsonl.log"
fi

TRAIN_JSONL="$JSONL_DIR/train.jsonl"
TEST_JSONL="$JSONL_DIR/test.jsonl"
if [[ ! -f "$TRAIN_JSONL" || ! -f "$TEST_JSONL" ]]; then
  echo "ERROR: missing $TRAIN_JSONL or $TEST_JSONL" >&2
  exit 1
fi

N_TRAIN="$("$PY" -c "import json; print(json.load(open('$SPLIT_DIR/split_meta.json'))['n_train_sources'])")"
N_TEST="$("$PY" -c "import json; print(json.load(open('$SPLIT_DIR/split_meta.json'))['n_test_sources'])")"
MAX_QUERIES="${MAX_QUERIES:-$N_TRAIN}"
# n80 protocol; if test smaller, eval all
MAX_SAMPLES="${MAX_SAMPLES:-80}"
if (( N_TEST < MAX_SAMPLES )); then
  MAX_SAMPLES="$N_TEST"
fi
N_POOL="${N_POOL:-50}"

if [[ "$SKIP_TRAIN" != "1" ]]; then
  echo "== pairs + train (HARD_NEG_CITELINK empty; BASE=$BASE) =="
  DOMAIN="$DOMAIN" \
  TRAIN_JSONL="$TRAIN_JSONL" \
  PAIRS="$PAIRS" \
  OUT="$OUT" \
  BASE="$BASE" \
  MAX_QUERIES="$MAX_QUERIES" \
  HARD_NEG_CITELINK= \
  REBUILD=1 DO_TRAIN=1 DO_EVAL=0 \
  LOG_DIR="$LOG_DIR" \
  BUILD_LOG_PREFIX="v5_${DOMAIN}_build" \
  TRAIN_LOG="v5_${DOMAIN}_ce_sent_train.log" \
  GPUS="$GPUS" \
  bash scripts/run_rr_ce_sent_v4b_4gpu.sh
fi

if [[ ! -d "$OUT" ]]; then
  echo "ERROR: missing model $OUT" >&2
  exit 1
fi

echo "== pure CE n80/n_eval domain=${DOMAIN} n=${MAX_SAMPLES} N=${N_POOL} =="
export RR_CE_SENT_SHORT=1
export RR_CE_SENT_WINDOW=400
export RR_CE_SENT_MAX_SENTS=3
export RR_CE_SENT_MAX_LEN=256
DOMAIN="$DOMAIN" \
RR_CE_SENT_CITELINK_BLEND=0 \
RANKER_CE_SENT="$OUT" \
RANKER_CITELINK= \
RANKER_SENT= RANKER_FUSION= RANKER_LTR= \
TAG="$TAG_PURE" \
MAX_SAMPLES="$MAX_SAMPLES" \
SEED=42 \
N="$N_POOL" \
TEST_JSONL="$TEST_JSONL" \
LOG_DIR="$LOG_DIR" \
GPUS="$GPUS" \
OUT_DIR="$EVAL_DIR" \
bash scripts/run_eval_rr_pool_ranker_4gpu.sh

MERGED="${EVAL_DIR}/${TAG_PURE}_merged.json"
"$PY" - <<PY
import json
from pathlib import Path
from datetime import datetime, timezone

split_meta = json.loads(Path("$SPLIT_DIR/split_meta.json").read_text())
merged_path = Path("$MERGED")
summary = {}
if merged_path.exists():
    summary = json.loads(merged_path.read_text()).get("summary") or {}

test = set(json.loads(Path("$SPLIT_DIR/test_sources.json").read_text()))
train_ids = set()
with Path("$TRAIN_JSONL").open() as f:
    for line in f:
        if not line.strip():
            continue
        train_ids.add(str(json.loads(line)["source_id"]))
inter = sorted(train_ids & test)
report = {
    "domain": "$DOMAIN",
    "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "gexf": "$GEXF",
    "gexf_sha256": split_meta.get("gexf_sha256"),
    "split_id": split_meta.get("split_id"),
    "n_train_sources": split_meta.get("n_train_sources"),
    "n_test_sources": split_meta.get("n_test_sources"),
    "base": "$BASE",
    "warm_start_domain": None,
    "hard_neg_citelink": None,
    "model": "$OUT",
    "pairs": "$PAIRS",
    "n_eval": int("$MAX_SAMPLES"),
    "N": int("$N_POOL"),
    "eval_tag": "$TAG_PURE",
    "merged": str(merged_path),
    "summary": summary,
    "train_jsonl_cap_test": len(inter),
    "train_jsonl_cap_test_ids_sample": inter[:10],
}
Path("$REPORT").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
if inter:
    raise SystemExit(f"LEAK: {len(inter)} train.jsonl source_ids in test sources")
PY

echo "== done domain=${DOMAIN} report=$REPORT merged=$MERGED =="
