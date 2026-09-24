#!/usr/bin/env bash
# Domain RR ranker env for RW-Cite (B1 / sclean SciBERT C₂s preferred).
# Usage: source configs/env_domain_rr.sh <domain>
# Optional: ALIGN=1 to prefer …-c2s-hn / …-time_*-hn checkpoints.
_DOM="${1:-}"
if [[ -z "$_DOM" ]]; then
  echo "Usage: source configs/env_domain_rr.sh <domain>" >&2
  return 2 2>/dev/null || exit 2
fi
_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export RWCITE_ROOT="${RWCITE_ROOT:-$_ROOT}"

_PY="${RWCITE_PYTHON:-python3}"
if [[ -x "${_ROOT}/.venv/bin/python" && -z "${RWCITE_PYTHON:-}" && "${_PY}" == "python3" ]]; then
  _PY="${_ROOT}/.venv/bin/python"
fi

_DOMAIN_ROOT="$("$_PY" - <<PY
import yaml
from pathlib import Path
from rwcite.graph.domain_paths import normalize_working_dir

root = Path("$_ROOT")
dom = "$_DOM"
block = yaml.safe_load((root / "configs/domains.yaml").read_text())["domains"][dom]
ycfg = yaml.safe_load((root / block["domain_config"]).read_text())
print(normalize_working_dir(ycfg["data_downloading"]["download_directory"]))
PY
)"

_pick=""
# Prefer domain-local SciBERT C₂s HN. ewm production CE is nofuture-trained (cutover 2026-09-08).
_ce_cands=()
if [[ "$_DOM" == "ewm" ]]; then
  _ce_cands+=(
    "${_ROOT}/${_DOMAIN_ROOT}ranker/ce_sent/c2s-hn-scibert_scivocab_uncased"
    "${_ROOT}/${_DOMAIN_ROOT}ranker/ce_sent/c2s-hn-scibert_scivocab_uncased_nofuture"
  )
else
  _ce_cands+=(
    "${_ROOT}/${_DOMAIN_ROOT}ranker/ce_sent/c2s-hn-scibert_scivocab_uncased"
  )
fi
_ce_cands+=(
  "${_ROOT}/${_DOMAIN_ROOT}ranker/ce_sent/c2s-hn-"*
  "${_ROOT}/models/rr-pool-ranker-ce-sent-v5-${_DOM}-base-scibert_scivocab_uncased-c2s-hn"
  "${_ROOT}/models/rr-pool-ranker-ce-sent-v5-${_DOM}-base-"*"-c2s-hn"
)
for cand in "${_ce_cands[@]}"; do
  if [[ -d "$cand" ]]; then
    _pick="$cand"
    break
  fi
done
if [[ -z "$_pick" ]]; then
  echo "WARN: no CE model found for domain=${_DOM}; set RR_RANKER_CE_SENT_PATH manually" >&2
else
  export RR_RANKER_CE_SENT_PATH="$_pick"
fi

_struct="${_ROOT}/${_DOMAIN_ROOT}ranker/struct/model.npz"
_struct_nf="${_ROOT}/${_DOMAIN_ROOT}ranker/struct_nofuture/model.npz"
if [[ -f "$_struct" ]]; then
  export RR_RANKER_PATH="$_struct"
elif [[ "$_DOM" == "ewm" && -f "$_struct_nf" ]]; then
  export RR_RANKER_PATH="$_struct_nf"
elif [[ -z "${RR_RANKER_PATH:-}" ]]; then
  echo "WARN: no domain struct ranker at $_struct; CE shortlist falls back to heuristic" >&2
fi

# ewm default graph is nofuture (domains.yaml); keep split dir for future-source masks if set.
if [[ "$_DOM" == "ewm" ]]; then
  export RR_SPLIT_DIR="${RR_SPLIT_DIR:-${_ROOT}/${_DOMAIN_ROOT}data/splits}"
fi

unset RR_RANKER_CITELINK_PATH
export RR_CE_SENT_CITELINK_BLEND=0
export RR_CE_SENT_SHORT=1
export RR_CE_SENT_WINDOW=400
export RR_CE_SENT_MAX_SENTS=3
export RR_CE_SENT_MAX_LEN=256

# Optional MLP override path (standard coarse is linear model.npz).
_mlp="${_ROOT}/${_DOMAIN_ROOT}ranker/struct/mlp.npz"
if [[ -f "$_mlp" ]]; then
  export RR_RANKER_MLP_PATH="$_mlp"
else
  unset RR_RANKER_MLP_PATH || true
fi

export RR_STRUCT_COARSE="${RR_STRUCT_COARSE:-linear}"
export RR_RANKER_STRUCT_PREFILTER="${RR_RANKER_STRUCT_PREFILTER:-800}"
export RR_RANKER_FULL_EMB=0
export RR_RANKER_WITH_EMB=0
export RR_RANKER_DIRECT_TOP10=1
export RR_CAND_N="${RR_CAND_N:-50}"
echo "Domain RR env ($_DOM): CE=${RR_RANKER_CE_SENT_PATH:-unset} struct=${RR_RANKER_PATH:-unset} mlp=${RR_RANKER_MLP_PATH:-unset} coarse=$RR_STRUCT_COARSE prefilter=$RR_RANKER_STRUCT_PREFILTER blend=$RR_CE_SENT_CITELINK_BLEND DIRECT_TOP10=1"
