#!/usr/bin/env bash
# Ranker v5 Release recipe = B1 / §3.5 (sclean + SciBERT + full citelink + C₂s):
#   0) optional sclean rewrite + dump split + RR jsonl  (DO_PREP=1)
#   0c) domain struct shortlist ranker (dump feats + RankNet fit)
#   1) full-domain citelink (MAX_QUERIES=0) + quality gate
#   2) SHORT + random-neg pairs → stage1 CE (new head on SciBERT)
#   3) SHORT + citelink-HN pairs → stage2 CE warm from stage1
#   4) n80 pure CE (β=0); blend optional (SKIP_BLEND=0)
#
# Defaults match V5 tech report §3.2.1-F / §3.5:
#   BASE=models/base/scibert_scivocab_uncased  (new CE head)
#   DATA_TAG=sclean  stage1=2/1  stage2=2/0  max_length=256  CE_GPUS single-card
#
# Usage:
#   GPUS=0,1,2,3 CE_GPUS=0 bash scripts/run_domain_rr_c2s.sh --domain <tag>
#   DO_PREP=1 GPUS=0,1,2,3 CE_GPUS=0 bash scripts/run_domain_rr_c2s.sh --domain <tag>
set -euo pipefail
ROOT="${RWCITE_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$ROOT"

DOMAIN="${DOMAIN:-}"
GPUS="${GPUS:-0,1,2,3}"
CE_GPUS="${CE_GPUS:-0}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --domain) DOMAIN="$2"; shift 2 ;;
    -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "Unknown: $1" >&2; exit 2 ;;
  esac
done
[[ -n "$DOMAIN" ]] || { echo "--domain required" >&2; exit 2; }

PY="${RWCITE_PYTHON:-${PY:-python3}}"
if [[ -x "${ROOT}/.venv/bin/python" && -z "${RWCITE_PYTHON:-}" && "${PY}" == "python3" ]]; then
  PY="${ROOT}/.venv/bin/python"
fi
module load cuda-12.8 2>/dev/null || true
export RWCITE_ROOT="$ROOT"

DATA_TAG="${DATA_TAG:-sclean}"

BASE="${BASE:-models/base/scibert_scivocab_uncased}"
BASE_TAG="${BASE_TAG:-$(basename "$BASE")}"
BASE_TAG="${BASE_TAG//\//-}"

LOG_DIR="${LOG_DIR:-logs/rr_ltr_${DOMAIN}_c2s_${BASE_TAG}}"
# Domain-local artifact paths (data/, ranker/, …)
mapfile -t _paths < <("$PY" - <<PY
from pathlib import Path
from rwcite.graph.domain_paths import domain_layout, working_dir_for_domain

root = Path(".")
dom = "$DOMAIN"
wd = working_dir_for_domain(dom, root)
p = domain_layout(wd, base_tag="$BASE_TAG")
for k in (
    "root",
    "splits_dir",
    "jsonl_dir",
    "citelink_model",
    "citelink_cache",
    "base_pairs",
    "hn_pairs",
    "ce_stage1",
    "ce_c2s_hn",
    "eval_dir",
    "citelink_gate",
    "c2s_report",
    "struct_model",
    "struct_feats",
):
    print(p[k])
PY
)
DOMAIN_ROOT="${_paths[0]}"
SPLIT_DIR="${SPLIT_DIR:-${_paths[1]}}"
JSONL_DIR="${JSONL_DIR:-${_paths[2]}}"
CITELINK_OUT="${CITELINK_OUT:-${_paths[3]}}"
CITELINK_CACHE="${CITELINK_CACHE:-${_paths[4]}}"
BASE_PAIRS="${BASE_PAIRS:-${_paths[5]}}"
HN_PAIRS="${HN_PAIRS:-${_paths[6]}}"
OUT_STAGE1="${OUT_STAGE1:-${_paths[7]}}"
OUT="${OUT:-${_paths[8]}}"
EVAL_DIR="${EVAL_DIR:-${_paths[9]}}"
GATE_OUT="${GATE_OUT:-${_paths[10]}}"
REPORT="${REPORT:-${_paths[11]}}"
STRUCT_MODEL="${STRUCT_MODEL:-${_paths[12]}}"
STRUCT_FEATS="${STRUCT_FEATS:-${_paths[13]}}"
TRAIN_JSONL="${TRAIN_JSONL:-$JSONL_DIR/train.jsonl}"
TEST_JSONL="${TEST_JSONL:-$JSONL_DIR/test.jsonl}"

# Resolve gexf / retrieval from domains.yaml
mapfile -t _gexf < <("$PY" - <<PY
import yaml
from pathlib import Path
root = Path(".")
dom = "$DOMAIN"
block = yaml.safe_load((root / "configs/domains.yaml").read_text())["domains"][dom]
print(block["gexf"])
print(block.get("retrieval_nodes", ""))
PY
)
GEXF="${GEXF:-${_gexf[0]}}"
RET_NODES="${RET_NODES:-${_gexf[1]}}"

SKIP_CITELINK="${SKIP_CITELINK:-0}"
REBUILD_PAIRS="${REBUILD_PAIRS:-1}"
SKIP_BLEND="${SKIP_BLEND:-1}"
FORCE="${FORCE:-0}"
DO_COVERAGE_DIAG="${DO_COVERAGE_DIAG:-1}"
DO_PREP="${DO_PREP:-0}"
SKIP_REWRITE="${SKIP_REWRITE:-0}"
SKIP_POOL_RANKER="${SKIP_POOL_RANKER:-0}"
MASTER_PORT="${MASTER_PORT:-29520}"
DDP_EPOCH_MODE="${DDP_EPOCH_MODE:-full}"
LR_SCALE_DDP="${LR_SCALE_DDP:-0}"
ANCHOR_HITS_AT_10="${ANCHOR_HITS_AT_10:-}"
MAX_LENGTH="${MAX_LENGTH:-256}"
STAGE1_EPOCHS="${STAGE1_EPOCHS:-2}"
STAGE1_FREEZE="${STAGE1_FREEZE:-1}"
STAGE2_EPOCHS="${STAGE2_EPOCHS:-2}"
STAGE2_FREEZE="${STAGE2_FREEZE:-0}"

mkdir -p "$LOG_DIR" "$(dirname "$CITELINK_OUT")" "$(dirname "$HN_PAIRS")" "$OUT_STAGE1" "$OUT" "$SPLIT_DIR" "$JSONL_DIR" "$EVAL_DIR" "${DOMAIN_ROOT}data" "${DOMAIN_ROOT}ranker/pool" "$(dirname "$STRUCT_MODEL")" "$STRUCT_FEATS"

# gexf / retrieval resolved above with DOMAIN_ROOT

if [[ "$DO_PREP" == "1" ]]; then
  CLEAN_REPORT="${GEXF%.gexf}.sentences_clean_report.json"
  if [[ "$SKIP_REWRITE" != "1" ]]; then
    if [[ -f "$CLEAN_REPORT" && -f "${GEXF%.gexf}.pre_sentence_clean.gexf.bak" ]]; then
      echo "== 0) sclean report present; skip rewrite =="
    else
      echo "== 0) offline cite_sentence_clean on $GEXF =="
      "$PY" -m rwcite.cli.rewrite_sentences \
        --gexf "$GEXF" --backup --also-install
    fi
  fi
  echo "== 0a) dump test_admit_v1 split =="
  "$PY" -m rwcite.cli.dump_split \
    --elig-k 10 --test-frac 0.10 \
    --gexf "$GEXF" --retrieval-nodes "$RET_NODES" \
    --out-dir "$SPLIT_DIR"
  echo "== 0b) build RR jsonl =="
  "$PY" -m rwcite.cli.build_rr_jsonl \
    --gexf "$GEXF" --split-file "$SPLIT_DIR" \
    --out-dir "$JSONL_DIR" --which both
fi

[[ -f "$TRAIN_JSONL" ]] || { echo "missing $TRAIN_JSONL (set DO_PREP=1 or build jsonl first)" >&2; exit 1; }
[[ -f "$SPLIT_DIR/split_meta.json" ]] || { echo "missing $SPLIT_DIR/split_meta.json" >&2; exit 1; }
[[ -d "$BASE" ]] || { echo "missing BASE $BASE (bootstrap SciBERT or set BASE=)" >&2; exit 1; }

N_TRAIN="$("$PY" -c "import json; print(json.load(open('$SPLIT_DIR/split_meta.json'))['n_train_sources'])")"
N_TEST="$("$PY" -c "import json; print(json.load(open('$SPLIT_DIR/split_meta.json'))['n_test_sources'])")"
MAX_SAMPLES="${MAX_SAMPLES:-80}"
# Prefer if/fi over ``((…)) &&`` so a false comparison never trips ``set -e``.
if (( N_TEST < MAX_SAMPLES )); then
  MAX_SAMPLES="$N_TEST"
fi

if [[ "$SKIP_POOL_RANKER" != "1" && -z "${RR_RANKER_PATH:-}" ]]; then
  DOMAIN="$DOMAIN" TRAIN_JSONL="$TRAIN_JSONL" GPUS="$GPUS" FORCE="$FORCE" \
    STRUCT_MODEL="$STRUCT_MODEL" STRUCT_FEATS="$STRUCT_FEATS" LOG_DIR="$LOG_DIR" \
    bash scripts/run_domain_train_pool_ranker.sh
elif [[ -n "${RR_RANKER_PATH:-}" ]]; then
  echo "== 0c) RR_RANKER_PATH preset ($RR_RANKER_PATH); skip domain struct train =="
elif [[ "$SKIP_POOL_RANKER" == "1" ]]; then
  echo "== 0c) SKIP_POOL_RANKER=1 =="
fi
# Child ``export`` does not propagate; always resolve the shortlist path here.
if [[ -z "${RR_RANKER_PATH:-}" && -f "$STRUCT_MODEL" ]]; then
  RR_RANKER_PATH="$(readlink -f -- "$STRUCT_MODEL")"
  export RR_RANKER_PATH
fi
if [[ -n "${RR_RANKER_PATH:-}" ]]; then
  echo "   struct shortlist=$RR_RANKER_PATH"
fi

TAG_PURE="${TAG_PURE:-${DOMAIN}_v5_${BASE_TAG}_c2s_pure_n80}"
TAG_BLEND="${TAG_BLEND:-${DOMAIN}_v5_${BASE_TAG}_c2s_blend_n80}"
GATE_OUT="${GATE_OUT:-$EVAL_DIR/citelink_gate.json}"
CITELINK_EVAL_TAG="${CITELINK_EVAL_TAG:-citelink_${DOMAIN}_n80}"

echo "== domain RR C_2s (B1/sclean/SciBERT): domain=${DOMAIN} data=${DATA_TAG:-none} base=${BASE} tag=${BASE_TAG} ce_gpus=${CE_GPUS} =="
echo "   train=$TRAIN_JSONL citelink=$CITELINK_OUT out=$OUT"
echo "   max_length=${MAX_LENGTH} stage1=${STAGE1_EPOCHS}/${STAGE1_FREEZE} stage2=${STAGE2_EPOCHS}/${STAGE2_FREEZE}"

if [[ "$DO_COVERAGE_DIAG" == "1" ]]; then
  "$PY" - <<PY | tee "$LOG_DIR/coverage_diag.json"
import json
from pathlib import Path
import networkx as nx
import yaml
root = Path(".")
dom = "$DOMAIN"
cfg = yaml.safe_load((root/"configs/domains.yaml").read_text())["domains"][dom]
gexf = cfg["gexf"]
ret_path = cfg.get("retrieval_nodes") or ""
ret = set(json.loads((root/ret_path).read_text())) if ret_path and (root/ret_path).is_file() else set()
g = nx.read_gexf(str(root/gexf))
miss = sorted(ret - set(g.nodes())) if ret else []
pap = root / Path(gexf).parts[0] / "research_papers"
cand = root / Path(gexf).parent.parent / "research_papers"
if cand.is_dir():
    pap = cand
have_tex = sum(1 for i in miss if (pap/i).is_dir()) if pap.is_dir() else 0
print(json.dumps({
  "domain": dom, "gexf": gexf, "n_ret": len(ret),
  "ret_in_graph": len(ret & set(g.nodes())) if ret else None,
  "ret_in_graph_frac": round(len(ret & set(g.nodes()))/max(1,len(ret)), 4) if ret else None,
  "ret_missing": len(miss), "missing_with_tex": have_tex,
}, indent=2))
PY
fi

CITELINK_META="$(dirname "$CITELINK_OUT")/meta.json"
CITELINK_MERGED="${EVAL_DIR}/${CITELINK_EVAL_TAG}_merged.json"

run_citelink_gate() {
  "$PY" -m rwcite.cli.gate_citelink \
    --meta "$CITELINK_META" \
    --eval-merged "$CITELINK_MERGED" \
    --max-val-loss "${GATE_MAX_VAL_LOSS:-0.40}" \
    --min-hits-at-10 "${GATE_MIN_HITS:-2.0}" \
    --out "$GATE_OUT" \
    || { echo "citelink quality gate FAILED — refuse HN" >&2; exit 1; }
}

if [[ "$SKIP_CITELINK" == "1" ]]; then
  echo "== 1) SKIP_CITELINK=1 (using $CITELINK_OUT) =="
  [[ -f "$CITELINK_OUT" ]] || { echo "missing citelink $CITELINK_OUT" >&2; exit 1; }
  if [[ "$FORCE" == "1" || ! -f "$GATE_OUT" ]]; then
    if [[ "$FORCE" == "1" || ! -f "$CITELINK_MERGED" ]]; then
      echo "== 1b) citelink n80 eval (reuse model) =="
      DOMAIN="$DOMAIN" TRAIN_JSONL="$TRAIN_JSONL" TEST_JSONL="$TEST_JSONL" \
      CACHE="$CITELINK_CACHE" OUT="$CITELINK_OUT" \
      MAX_QUERIES=0 WINDOW=400 NEG_PER_LIST=64 \
      REBUILD=0 DO_TRAIN=0 DO_EVAL=1 \
      TAG="$CITELINK_EVAL_TAG" \
      MAX_SAMPLES="$MAX_SAMPLES" \
      LOG_DIR="$LOG_DIR" BUILD_LOG_PREFIX="citelink_${DOMAIN}_build" \
      GPUS="$GPUS" OUT_DIR="$EVAL_DIR" \
      bash scripts/run_rr_citelink_v3c_4gpu.sh
    fi
    run_citelink_gate
  else
    echo "== 1) reuse gate $GATE_OUT =="
  fi
elif [[ "$FORCE" != "1" && -f "$CITELINK_OUT" && -f "$CITELINK_META" ]]; then
  echo "== 1) reuse citelink $CITELINK_OUT (skip pair rebuild + MLP train) =="
  if [[ ! -f "$CITELINK_MERGED" ]]; then
    echo "== 1b) citelink n80 eval (reuse model) =="
    DOMAIN="$DOMAIN" TRAIN_JSONL="$TRAIN_JSONL" TEST_JSONL="$TEST_JSONL" \
    CACHE="$CITELINK_CACHE" OUT="$CITELINK_OUT" \
    MAX_QUERIES=0 WINDOW=400 NEG_PER_LIST=64 \
    REBUILD=0 DO_TRAIN=0 DO_EVAL=1 \
    TAG="$CITELINK_EVAL_TAG" \
    MAX_SAMPLES="$MAX_SAMPLES" \
    LOG_DIR="$LOG_DIR" BUILD_LOG_PREFIX="citelink_${DOMAIN}_build" \
    GPUS="$GPUS" OUT_DIR="$EVAL_DIR" \
    bash scripts/run_rr_citelink_v3c_4gpu.sh
  else
    echo "== 1b) skip citelink n80 (exists $CITELINK_MERGED) =="
  fi
  if [[ "$FORCE" == "1" || ! -f "$GATE_OUT" ]]; then
    run_citelink_gate
  else
    echo "== 1c) reuse gate $GATE_OUT =="
  fi
else
  echo "== 1) train domain citelink (MAX_QUERIES=0 full) =="
  DOMAIN="$DOMAIN" TRAIN_JSONL="$TRAIN_JSONL" TEST_JSONL="$TEST_JSONL" \
  CACHE="$CITELINK_CACHE" OUT="$CITELINK_OUT" \
  MAX_QUERIES=0 WINDOW=400 NEG_PER_LIST=64 \
  EPOCHS="${CITELINK_EPOCHS:-12}" \
  REBUILD=1 DO_TRAIN=1 DO_EVAL=1 \
  TAG="$CITELINK_EVAL_TAG" \
  MAX_SAMPLES="$MAX_SAMPLES" \
  LOG_DIR="$LOG_DIR" BUILD_LOG_PREFIX="citelink_${DOMAIN}_build" \
  GPUS="$GPUS" OUT_DIR="$EVAL_DIR" \
  bash scripts/run_rr_citelink_v3c_4gpu.sh

  [[ -f "$CITELINK_OUT" ]] || { echo "missing citelink $CITELINK_OUT" >&2; exit 1; }
  run_citelink_gate
fi

if [[ "$REBUILD_PAIRS" == "1" ]]; then
  echo "== 2a) CE base pairs (SHORT + random neg) =="
  DOMAIN="$DOMAIN" TRAIN_JSONL="$TRAIN_JSONL" \
  PAIRS="$BASE_PAIRS" OUT="$OUT_STAGE1" BASE="$BASE" \
  HARD_NEG_CITELINK="" CITELINK="" \
  MAX_QUERIES="$N_TRAIN" \
  REBUILD=1 DO_TRAIN=0 DO_EVAL=0 DO_BLEND_EVAL=0 \
  LOG_DIR="$LOG_DIR" \
  BUILD_LOG_PREFIX="v5_${DOMAIN}_base_build" \
  GPUS="$GPUS" \
  bash scripts/run_rr_ce_sent_v4b_4gpu.sh

  echo "== 2c-prep) CE HN pairs (domain citelink hard-negs) =="
  DOMAIN="$DOMAIN" TRAIN_JSONL="$TRAIN_JSONL" \
  PAIRS="$HN_PAIRS" OUT="$OUT" BASE="$OUT_STAGE1" \
  HARD_NEG_CITELINK="$CITELINK_OUT" \
  CITELINK="$CITELINK_OUT" \
  MAX_QUERIES="$N_TRAIN" \
  REBUILD=1 DO_TRAIN=0 DO_EVAL=0 DO_BLEND_EVAL=0 \
  LOG_DIR="$LOG_DIR" \
  BUILD_LOG_PREFIX="v5_${DOMAIN}_hn_build" \
  GPUS="$GPUS" \
  bash scripts/run_rr_ce_sent_v4b_4gpu.sh
else
  echo "== 2a/2c) REBUILD_PAIRS=0 reuse base=$BASE_PAIRS hn=$HN_PAIRS =="
  [[ -f "$BASE_PAIRS" ]] || { echo "missing $BASE_PAIRS" >&2; exit 1; }
  [[ -f "$HN_PAIRS" ]] || { echo "missing $HN_PAIRS" >&2; exit 1; }
fi

# Note: invoke CE behind </dev/null so a pipe/stdin-launched parent cannot have
# its remaining script lines stolen by the trainer (classic `) ` syntax error after stage1).
if [[ "$FORCE" == "1" || ! -f "$OUT_STAGE1/train_meta.json" ]]; then
  echo "== 2b. CE stage1 train BASE->stage1 EPOCHS=${STAGE1_EPOCHS} FREEZE=${STAGE1_FREEZE} MAX_LENGTH=${MAX_LENGTH} =="
  MASTER_PORT=$((MASTER_PORT + 1))
  DOMAIN="$DOMAIN" TRAIN_JSONL="$TRAIN_JSONL" TEST_JSONL="$TEST_JSONL" \
  PAIRS="$BASE_PAIRS" OUT="$OUT_STAGE1" BASE="$BASE" \
  HARD_NEG_CITELINK="" CITELINK="" \
  MAX_QUERIES="$N_TRAIN" EPOCHS="$STAGE1_EPOCHS" FREEZE_EPOCHS="$STAGE1_FREEZE" \
  MAX_LENGTH="$MAX_LENGTH" \
  REBUILD=0 DO_TRAIN=1 DO_EVAL=0 DO_BLEND_EVAL=0 \
  DDP_EPOCH_MODE="$DDP_EPOCH_MODE" LR_SCALE_DDP="$LR_SCALE_DDP" \
  LOG_DIR="$LOG_DIR" TRAIN_LOG="c2s_stage1_train.log" \
  GPUS="$CE_GPUS" MASTER_PORT="$MASTER_PORT" \
  bash scripts/run_rr_ce_sent_v4b_4gpu.sh </dev/null
else
  echo "== 2b. skip stage1 meta=${OUT_STAGE1}/train_meta.json =="
fi

if [[ "$FORCE" == "1" || ! -f "$OUT/train_meta.json" ]]; then
  echo "== 2d. CE stage2 train stage1->HN EPOCHS=${STAGE2_EPOCHS} FREEZE=${STAGE2_FREEZE} MAX_LENGTH=${MAX_LENGTH} =="
  MASTER_PORT=$((MASTER_PORT + 1))
  DOMAIN="$DOMAIN" TRAIN_JSONL="$TRAIN_JSONL" TEST_JSONL="$TEST_JSONL" \
  PAIRS="$HN_PAIRS" OUT="$OUT" BASE="$OUT_STAGE1" \
  HARD_NEG_CITELINK="" CITELINK="" \
  MAX_QUERIES="$N_TRAIN" EPOCHS="$STAGE2_EPOCHS" FREEZE_EPOCHS="$STAGE2_FREEZE" \
  MAX_LENGTH="$MAX_LENGTH" \
  REBUILD=0 DO_TRAIN=1 DO_EVAL=0 DO_BLEND_EVAL=0 \
  DDP_EPOCH_MODE="$DDP_EPOCH_MODE" LR_SCALE_DDP="$LR_SCALE_DDP" \
  LOG_DIR="$LOG_DIR" TRAIN_LOG="c2s_stage2_train.log" \
  GPUS="$CE_GPUS" MASTER_PORT="$MASTER_PORT" \
  bash scripts/run_rr_ce_sent_v4b_4gpu.sh </dev/null
else
  echo "== 2d. skip stage2 meta=${OUT}/train_meta.json =="
fi

export RR_CE_SENT_SHORT=1 RR_CE_SENT_WINDOW=400 RR_CE_SENT_MAX_SENTS=3 RR_CE_SENT_MAX_LEN="$MAX_LENGTH"

if [[ "$FORCE" == "1" || ! -f "${EVAL_DIR}/${TAG_PURE}_merged.json" ]]; then
  echo "== 3) n80 pure CE =="
  DOMAIN="$DOMAIN" RR_CE_SENT_CITELINK_BLEND=0 \
  RANKER_CE_SENT="$OUT" RANKER_CITELINK= \
  TAG="$TAG_PURE" MAX_SAMPLES="$MAX_SAMPLES" SEED=42 N=50 \
  TEST_JSONL="$TEST_JSONL" LOG_DIR="$LOG_DIR" GPUS="$CE_GPUS" \
  OUT_DIR="$EVAL_DIR" \
  bash scripts/run_eval_rr_pool_ranker_4gpu.sh
else
  echo "== 3. skip pure eval; exists ${TAG_PURE}_merged.json =="
fi

if [[ "$SKIP_BLEND" != "1" ]]; then
  if [[ "$FORCE" == "1" || ! -f "${EVAL_DIR}/${TAG_BLEND}_merged.json" ]]; then
    echo "== 4) n80 CE⊕citelink blend=0.25 =="
    DOMAIN="$DOMAIN" RR_CE_SENT_CITELINK_BLEND=0.25 \
    RANKER_CE_SENT="$OUT" RANKER_CITELINK="$CITELINK_OUT" \
    TAG="$TAG_BLEND" MAX_SAMPLES="$MAX_SAMPLES" SEED=42 N=50 \
    TEST_JSONL="$TEST_JSONL" LOG_DIR="$LOG_DIR" GPUS="$CE_GPUS" \
    OUT_DIR="$EVAL_DIR" \
    bash scripts/run_eval_rr_pool_ranker_4gpu.sh
  else
    echo "== 4) skip blend eval =="
  fi
fi

echo "== 5) write $REPORT =="
"$PY" - <<PY
import json
from pathlib import Path
from datetime import datetime, timezone

def summ(tag):
    p = Path("$EVAL_DIR") / f"{tag}_merged.json"
    if not p.exists():
        return None
    return json.loads(p.read_text()).get("summary")

split = json.loads(Path("$SPLIT_DIR/split_meta.json").read_text())
pure = summ("$TAG_PURE")
blend = summ("$TAG_BLEND") if "$SKIP_BLEND" != "1" else None
hits = (pure or {}).get("mean_hits_at_10")
anchor = "$ANCHOR_HITS_AT_10".strip()
deploy = None
if anchor and hits is not None:
    a = float(anchor)
    deploy = "propose_cutover" if hits >= a else "keep_anchor_deploy"

cov = Path("$LOG_DIR/coverage_diag.json")
struct_report = Path("$STRUCT_MODEL").parent / "train_report.json"
report = {
    "domain": "$DOMAIN",
    "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "protocol": "sclean_scibert_c2s_two_stage",
    "data_tag": ("$DATA_TAG" or None),
    "split_id": split.get("split_id"),
    "gexf_sha256": split.get("gexf_sha256"),
    "n_train_sources": split.get("n_train_sources"),
    "n_test_sources": split.get("n_test_sources"),
    "citelink": "$CITELINK_OUT",
    "citelink_max_queries": 0,
    "ce_base": "$BASE",
    "ce_base_tag": "$BASE_TAG",
    "ce_base_recommended": "models/base/scibert_scivocab_uncased",
    "ce_gpus": "$CE_GPUS",
    "max_length": int("$MAX_LENGTH"),
    "stage1_epochs": int("$STAGE1_EPOCHS"),
    "stage1_freeze": int("$STAGE1_FREEZE"),
    "stage2_epochs": int("$STAGE2_EPOCHS"),
    "stage2_freeze": int("$STAGE2_FREEZE"),
    "skip_citelink": "$SKIP_CITELINK" == "1",
    "rebuild_pairs": "$REBUILD_PAIRS" == "1",
    "stage1_out": "$OUT_STAGE1",
    "ce_out": "$OUT",
    "base_pairs": "$BASE_PAIRS",
    "hn_pairs": "$HN_PAIRS",
    "struct_ranker": ("$RR_RANKER_PATH" or None),
    "struct_ranker_report": str(struct_report) if struct_report.is_file() else None,
    "n_eval": int("$MAX_SAMPLES"),
    "hn_pure_ce": pure,
    "hn_blend_0_25": blend,
    "anchor_hits_at_10": float(anchor) if anchor else None,
    "deploy_decision": deploy,
    "coverage_diag": json.loads(cov.read_text()) if cov.exists() else None,
}
Path("$REPORT").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
PY

echo "== done C_2s report=$REPORT =="
