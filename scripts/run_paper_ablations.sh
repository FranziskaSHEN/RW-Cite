#!/usr/bin/env bash
# Paper ablations over the frozen benchmark assets.
# Window and model variants rebuild or retrain the component they change.
#
# Usage:
#   bash scripts/run_paper_ablations.sh --domain ewm --cell phase1
#   bash scripts/run_paper_ablations.sh --domain ewm --cell C1
#   bash scripts/run_paper_ablations.sh --domain sqc --cell gate
#   bash scripts/run_paper_ablations.sh --domain ewm --cell phase2
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export RWCITE_ROOT="$ROOT"
export PYTHONUNBUFFERED=1

PY="${RWCITE_PYTHON:-python3}"
if [[ -x "${ROOT}/.venv/bin/python" && -z "${RWCITE_PYTHON:-}" && "${PY}" == "python3" ]]; then
  PY="${ROOT}/.venv/bin/python"
fi

DOMAIN="ewm"
CELL=""
MAX_SAMPLES="${MAX_SAMPLES:-80}"
SEED="${SEED:-42}"
DRY_RUN=0
FORCE_E4=0
FORCE_CE=0

PHASE1_CELLS=(C0 C1 C2 C3 C4a C4b C5a C5b C6a C6b C7)
GATE_CELLS=(C0 C1 C2 C3 C5a)
PHASE2_CELLS=(C9w200 C9w700 C10 C11e3 C11e5)

usage() {
  sed -n '2,14p' "$0"
  exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --domain) DOMAIN="$2"; shift 2 ;;
    --cell) CELL="$2"; shift 2 ;;
    --max-samples) MAX_SAMPLES="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --force-e4) FORCE_E4=1; shift ;;
    --force-ce) FORCE_CE=1; shift ;;
    -h|--help) usage 0 ;;
    *)
      echo "Unknown arg: $1" >&2
      usage 1
      ;;
  esac
done

if [[ -z "$CELL" ]]; then
  echo "ERROR: --cell required (phase1|phase2|gate|C0|…)" >&2
  usage 1
fi

# shellcheck source=scripts/lib_domain_env.sh
source "$ROOT/scripts/lib_domain_env.sh"
rwcite_load_domain_env

LOG_DIR="${LOG_DIR:-logs/paper_ablations/${DOMAIN}}"
mkdir -p "$LOG_DIR"
MVP="$ROOT/${GAT_MVP_DIR}"
LANE_RANKER="$(dirname "$MVP")"
# prefer env from lib_domain_env
N_TEST="${N_TEST:-0}"
N_TRAIN="${N_TRAIN:-0}"

symlink_mvp_core() {
  local dest="$1"
  mkdir -p "$dest"
  local f
  for f in id_map.json node_scibert.npy node_scibert_whitened.npy whiten.npz \
           adj_in.npz adj_out.npz adj_cocite.npz neighbors_k32.npz fold_masks.json \
           preprocess_report.json; do
    if [[ -e "$MVP/$f" && ! -e "$dest/$f" ]]; then
      ln -s "$MVP/$f" "$dest/$f"
    fi
  done
  # fold adjs if present
  shopt -s nullglob
  for f in "$MVP"/adj_*_fold*.npz; do
    bn="$(basename "$f")"
    [[ -e "$dest/$bn" ]] || ln -s "$f" "$dest/$bn"
  done
  shopt -u nullglob
}

run_fuse_cell() {
  local cell="$1"
  local asset="${2:-}"
  local win="${3:-}"
  local extra=()
  [[ "$FORCE_E4" == "1" ]] && extra+=(--force-e4)
  [[ "$FORCE_CE" == "1" ]] && extra+=(--force-ce)
  [[ -n "$asset" ]] && extra+=(--asset-dir "$asset")
  [[ -n "$win" ]] && extra+=(--windows "$win")
  echo "======== abl31 fuse cell=$cell domain=$DOMAIN max_samples=$MAX_SAMPLES ========"
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "  (dry-run) skip"
    return 0
  fi
  "$PY" -u -m rwcite.gat.paper_ablations \
    --domain "$DOMAIN" \
    --cell "$cell" \
    --max-samples "$MAX_SAMPLES" \
    --seed "$SEED" \
    "${extra[@]}" \
    2>&1 | tee -a "$LOG_DIR/${cell}.log"
}

build_windows_side() {
  local cell="$1"
  local W="$2"
  local coarse="${3:-linear}"
  local dest="${LANE_RANKER}/gat_abl31_${cell}"
  symlink_mvp_core "$dest"
  export RR_STRUCT_COARSE="$coarse"
  export RR_STRUCT_MASK_QUERY_OUT="${RR_STRUCT_MASK_QUERY_OUT:-1}"
  export RR_STRUCT_MASK_FUTURE_OUT="${RR_STRUCT_MASK_FUTURE_OUT:-1}"
  echo "== phase2 windows cell=$cell W=$W coarse=$coarse dest=$dest =="
  if [[ "$DRY_RUN" == "1" ]]; then
    return 0
  fi
  local test_npz="$dest/windows_test${N_TEST}.npz"
  local train_npz="$dest/windows_train${N_TRAIN}.npz"
  if [[ ! -f "$test_npz" || "${FORCE_WINDOWS:-0}" == "1" ]]; then
    "$PY" -m rwcite.gat.preprocess --domain "$DOMAIN" --step windows --windows test \
      --window "$W" --out-dir "$dest" 2>&1 | tee -a "$LOG_DIR/${cell}_prep.log"
  else
    echo "reuse $test_npz"
  fi
  if [[ ! -f "$train_npz" || "${FORCE_WINDOWS:-0}" == "1" ]]; then
    local N="${GAT_WINDOW_SHARDS:-2}"
    local PARALLEL="${GAT_WINDOW_PARALLEL:-2}"
    echo "== train windows shards N=$N parallel=$PARALLEL =="
    local i=0
    local running=0
    for ((i=0; i<N; i++)); do
      (
        "$PY" -m rwcite.gat.preprocess --domain "$DOMAIN" --step windows --windows train \
          --window "$W" --out-dir "$dest" --shard-idx "$i" --shard-count "$N"
      ) >> "$LOG_DIR/${cell}_prep.log" 2>&1 &
      running=$((running+1))
      if [[ "$running" -ge "$PARALLEL" ]]; then
        wait -n || true
        running=$((running-1))
      fi
    done
    wait
    "$PY" -m rwcite.gat.preprocess --domain "$DOMAIN" --step merge_windows \
      --out-dir "$dest" --merge-tag "train${N_TRAIN}" --shard-count "$N" \
      2>&1 | tee -a "$LOG_DIR/${cell}_prep.log"
  else
    echo "reuse $train_npz"
  fi
  # default E4 ckpt via symlink for scoring
  [[ -e "$dest/ckpt_e4.pt" ]] || ln -s "$MVP/ckpt_e4.pt" "$dest/ckpt_e4.pt"
  unset ABL31_CKPT ABL31_ABLATION || true
  export ABL31_ABLATION=E4
  # CE + E4 scores on new windows then L0_rrf fuse
  FORCE_CE=1 FORCE_E4=1 run_fuse_cell "$cell" "$dest" "$test_npz"
}

retrain_head_side() {
  local cell="$1"
  local abl="$2"   # E3 or E5
  local dest="${LANE_RANKER}/gat_abl31_${cell}"
  symlink_mvp_core "$dest"
  # reuse production 3.1 windows
  local tw="$MVP/windows_train${N_TRAIN}.npz"
  local te="$MVP/windows_test${N_TEST}.npz"
  [[ -f "$tw" ]] || { echo "ERROR: missing $tw" >&2; exit 1; }
  [[ -f "$te" ]] || { echo "ERROR: missing $te" >&2; exit 1; }
  [[ -e "$dest/windows_train${N_TRAIN}.npz" ]] || ln -s "$tw" "$dest/windows_train${N_TRAIN}.npz"
  [[ -e "$dest/windows_test${N_TEST}.npz" ]] || ln -s "$te" "$dest/windows_test${N_TEST}.npz"
  local ckpt_name="ckpt_${abl,,}.pt"
  echo "== phase2 retrain cell=$cell ablation=$abl dest=$dest =="
  if [[ "$DRY_RUN" == "1" ]]; then
    return 0
  fi
  if [[ ! -f "$dest/$ckpt_name" || "${FORCE_TRAIN:-0}" == "1" ]]; then
    local extra=()
    if [[ "$abl" == "E3" ]]; then
      extra+=(--struct-only-train)
    fi
    "$PY" -u -m rwcite.gat.train \
      --domain "$DOMAIN" \
      --out-dir "$dest" \
      --windows "$dest/windows_train${N_TRAIN}.npz" \
      --ablation "$abl" \
      --ckpt-name "$ckpt_name" \
      --report-name "train_report_${abl,,}.json" \
      --epochs 20 --min-epochs 5 --patience 4 \
      "${extra[@]}" \
      2>&1 | tee -a "$LOG_DIR/${cell}_train.log"
  else
    echo "reuse $dest/$ckpt_name"
  fi
  export ABL31_CKPT="$dest/$ckpt_name"
  export ABL31_ABLATION="$abl"
  FORCE_CE=0 FORCE_E4=1 run_fuse_cell "$cell" "$dest" "$dest/windows_test${N_TEST}.npz"
  unset ABL31_CKPT ABL31_ABLATION || true
}

run_phase2_cell() {
  local cell="$1"
  case "$cell" in
    C9w200) build_windows_side C9w200 200 linear ;;
    C9w700) build_windows_side C9w700 700 linear ;;
    C10) build_windows_side C10 400 heuristic ;;
    C11e3) retrain_head_side C11e3 E3 ;;
    C11e5) retrain_head_side C11e5 E5 ;;
    *)
      echo "ERROR: not a phase2 cell: $cell" >&2
      exit 1
      ;;
  esac
}

cells=()
case "$CELL" in
  phase1) cells=("${PHASE1_CELLS[@]}") ;;
  gate) cells=("${GATE_CELLS[@]}") ;;
  phase2) cells=("${PHASE2_CELLS[@]}") ;;
  C9w200|C9w700|C10|C11e3|C11e5) cells=("$CELL") ;;
  *) cells=("$CELL") ;;
esac

echo "START ablation_3.1 DOMAIN=$DOMAIN CELL=$CELL N_TEST=$N_TEST N_TRAIN=$N_TRAIN $(date -Is)" | tee -a "$LOG_DIR/driver.log"

for c in "${cells[@]}"; do
  case "$c" in
    C9w200|C9w700|C10|C11e3|C11e5)
      run_phase2_cell "$c"
      ;;
    *)
      unset ABL31_CKPT ABL31_ABLATION || true
      run_fuse_cell "$c"
      ;;
  esac
done

echo "DONE ablation_3.1 DOMAIN=$DOMAIN CELL=$CELL $(date -Is)" | tee -a "$LOG_DIR/driver.log"
echo "Merged reports under ${LANE_RANKER}/eval/abl31_${DOMAIN}_*"
echo "Fill docs/ABLATION_3.1.md from summary.mean_hits_at_10 (3.1 evidence only)."
