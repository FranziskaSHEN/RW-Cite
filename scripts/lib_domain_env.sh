#!/usr/bin/env bash
# Shared DOMAIN→path exports for multi-domain benchmark scripts.
# Usage (after ROOT/PY set):
#   # shellcheck source=scripts/lib_domain_env.sh
#   source "$ROOT/scripts/lib_domain_env.sh"
#   rwcite_load_domain_env   # requires DOMAIN=ewm|sqc|gw|fno|wsi|cosmo|qopt|sce|hep|exo|radio|radseg|driving
#
# Exports: DOMAIN LANE GEXF_O2 GEXF_NOFUTURE RR_GEXF_OVERRIDE SPLIT_DIR JSONL_DIR
#   RET_NODES GAT_MVP_DIR EVAL_DIR STRUCT_MODEL STRUCT_NOFUTURE_MODEL CE_STAGE1 CE_PAIRS
#   PACK_GPUS CE_GPU N_TRAIN N_TEST N_FRONTIER WINDOWS_*_TAG L0_CANONICAL_TAG GAT_EVAL_TAG
set -euo pipefail

rwcite_load_domain_env() {
  local _domain="${DOMAIN:?DOMAIN required (ewm|sqc|gw|fno|wsi|cosmo|qopt|sce|hep|exo|radio|radseg|driving)}"
  case "$_domain" in
    ewm|sqc|gw|fno|wsi|cosmo|qopt|sce|hep|exo|radio|radseg|driving) ;;
    *)
      echo "ERROR: DOMAIN must be ewm|sqc|gw|fno|wsi|cosmo|qopt|sce|hep|exo|radio|radseg|driving (got $_domain)" >&2
      return 2
      ;;
  esac
  local _py="${PY:-python3}"
  # Stale RR_GEXF_OVERRIDE from a prior domain must not leak into shell_exports /
  # effective_nofuture_gexf (parallel multi-domain runs).
  unset RR_GEXF_OVERRIDE RR_SPLIT_DIR || true
  # shellcheck disable=SC1090
  eval "$("$_py" - <<PY
from rwcite.gat.domain_layout import shell_exports
print(shell_exports("$_domain"))
PY
)"
  export DOMAIN LANE GEXF_O2 GEXF_NOFUTURE RR_GEXF_OVERRIDE SPLIT_DIR JSONL_DIR
  export RET_NODES GAT_MVP_DIR EVAL_DIR STRUCT_MODEL STRUCT_NOFUTURE_MODEL
  export CE_STAGE1 CE_PAIRS PACK_GPUS CE_GPU N_TRAIN N_TEST N_FRONTIER
  export WINDOWS_TEST_TAG WINDOWS_TRAIN_TAG L0_CANONICAL_TAG GAT_EVAL_TAG
  # Always pin override to this domain's nofuture GEXF (ignore any inherited value).
  if [[ "${GEXF_NOFUTURE}" = /* ]]; then
    export RR_GEXF_OVERRIDE="$GEXF_NOFUTURE"
  else
    export RR_GEXF_OVERRIDE="$ROOT/$GEXF_NOFUTURE"
  fi
  if [[ "${SPLIT_DIR}" = /* ]]; then
    export RR_SPLIT_DIR="$SPLIT_DIR"
  else
    export RR_SPLIT_DIR="$ROOT/$SPLIT_DIR"
  fi
  echo "== domain env DOMAIN=$DOMAIN LANE=$LANE gexf=$(basename "$RR_GEXF_OVERRIDE") N_train/test=$N_TRAIN/$N_TEST pack=$PACK_GPUS ce_gpu=$CE_GPU =="
}
