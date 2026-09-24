#!/usr/bin/env bash
# Pack GitHub-ready RW-Cite sources (aligned with .gitignore).
# Domain working dirs (*_retrieval, *_retrieval_as/rw, …) are never included —
# ship those via scripts/pack_domain_release.sh.
#
# Usage (argument is the tarball version label, not a pack mode):
#   bash scripts/pack_github_release.sh 3.1
#   bash scripts/pack_github_release.sh r3.1
#   OUT_DIR=/tmp bash scripts/pack_github_release.sh 3.1
#
# Writes (default OUT_DIR = releases/github):
#   releases/github/RW-Cite/          upload/publish-ready source tree
#   releases/github/RW-Cite-rX.Y.tar.gz
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NAME="$(basename "$ROOT")"
PARENT="$(cd "$ROOT/.." && pwd)"
OUT_DIR="${OUT_DIR:-$ROOT/releases/github}"

usage() {
  sed -n '2,11p' "$0"
  exit "${1:-0}"
}

[[ $# -ge 1 ]] || usage 1
[[ "${1:-}" == "-h" || "${1:-}" == "--help" ]] && usage 0

raw="$1"
ver="${raw#RW-Cite-}"
ver="${ver#rw-cite-}"
ver="${ver#v}"
ver="${ver#r}"
if [[ ! "$ver" =~ ^[0-9]+(\.[0-9]+)*$ ]]; then
  echo "ERROR: version must look like 1.1 or r1.1 (got: $raw)" >&2
  exit 1
fi

OUT="${OUT_DIR}/${NAME}-r${ver}.tar.gz"
RELEASE_STAGE="${OUT_DIR}/${NAME}"
tmpdir="$(mktemp -d)"
trap 'rm -rf "$tmpdir"' EXIT

echo "Packing ${NAME} -> ${OUT}"
echo "  source: ${ROOT}"

STAGE="$tmpdir/$NAME"
mkdir -p "$STAGE"

if command -v rsync >/dev/null 2>&1; then
  rsync -a \
    --exclude='.venv/' \
    --exclude='__pycache__/' \
    --exclude='*.py[cod]' \
    --exclude='*.egg-info/' \
    --exclude='rwcite.egg-info/' \
    --exclude='.eggs/' \
    --exclude='dist/' \
    --exclude='build/' \
    --exclude='.pytest_cache/' \
    --exclude='.mypy_cache/' \
    --exclude='.ruff_cache/' \
    --exclude='.git/' \
    --exclude='RW-Cite-Baseline/' \
    --exclude='releases/github/RW-Cite/' \
    --exclude='releases/github/*.tar.gz' \
    --exclude='releases/huggingface/corpus/RW-Cite-datasets-*/' \
    --exclude='releases/huggingface/corpus/*.tar.gz' \
    --exclude='releases/huggingface/domains/RW-Cite-domain-*/' \
    --exclude='releases/huggingface/domains/*.tar.gz' \
    --exclude='bash' \
    --exclude='configs/rr_adapter_v7*.yaml' \
    --exclude='configs/rr_v7_*' \
    --exclude='docs/RELEASE_*.md' \
    --exclude='docs/ABLATION_3.1*.md' \
    --exclude='docs/GAT solution.md' \
    --exclude='docs/ADAPTER_RR_v6.md' \
    --exclude='rwcite/adapter/dataset_v7.py' \
    --exclude='rwcite/gat/ablation_3_1.py' \
    --exclude='rwcite/cli/*rr_adapter_v7*.py' \
    --exclude='scripts/*rr_adapter_v7.sh' \
    --exclude='scripts/watch_rr_v7_*' \
    --exclude='scripts/check_ewm_3_contract.sh' \
    --exclude='scripts/run_ablation_2.0.sh' \
    --exclude='scripts/run_ablation_3.1.sh' \
    --exclude='scripts/run_domain_ranker_3.1_*.sh' \
    --exclude='tests/test_v7_*' \
    --exclude='.env' \
    --exclude='*.bak' \
    --exclude='*_retrieval/' \
    --exclude='*_retrieval_*/' \
    --exclude='datasets/' \
    --exclude='models/' \
    --exclude='logs/' \
    --exclude='outputs/' \
    "$ROOT"/ "$STAGE"/
else
  tar -C "$PARENT" -cf - \
    --exclude="${NAME}/.venv" \
    --exclude="*/__pycache__" \
    --exclude="${NAME}/rwcite.egg-info" \
    --exclude="${NAME}/*.egg-info" \
    --exclude="${NAME}/.eggs" \
    --exclude="${NAME}/dist" \
    --exclude="${NAME}/build" \
    --exclude="${NAME}/.pytest_cache" \
    --exclude="${NAME}/.mypy_cache" \
    --exclude="${NAME}/.ruff_cache" \
    --exclude="${NAME}/.git" \
    --exclude="${NAME}/RW-Cite-Baseline" \
    --exclude="${NAME}/releases/github/RW-Cite" \
    --exclude="${NAME}/releases/github/*.tar.gz" \
    --exclude="${NAME}/releases/huggingface/corpus/RW-Cite-datasets-*" \
    --exclude="${NAME}/releases/huggingface/corpus/*.tar.gz" \
    --exclude="${NAME}/releases/huggingface/domains/RW-Cite-domain-*" \
    --exclude="${NAME}/releases/huggingface/domains/*.tar.gz" \
    --exclude="${NAME}/bash" \
    --exclude="${NAME}/configs/rr_adapter_v7*.yaml" \
    --exclude="${NAME}/configs/rr_v7_*" \
    --exclude="${NAME}/docs/RELEASE_*.md" \
    --exclude="${NAME}/docs/ABLATION_3.1*.md" \
    --exclude="${NAME}/docs/GAT solution.md" \
    --exclude="${NAME}/docs/ADAPTER_RR_v6.md" \
    --exclude="${NAME}/rwcite/adapter/dataset_v7.py" \
    --exclude="${NAME}/rwcite/gat/ablation_3_1.py" \
    --exclude="${NAME}/rwcite/cli/*rr_adapter_v7*.py" \
    --exclude="${NAME}/scripts/*rr_adapter_v7.sh" \
    --exclude="${NAME}/scripts/watch_rr_v7_*" \
    --exclude="${NAME}/scripts/check_ewm_3_contract.sh" \
    --exclude="${NAME}/scripts/run_ablation_2.0.sh" \
    --exclude="${NAME}/scripts/run_ablation_3.1.sh" \
    --exclude="${NAME}/scripts/run_domain_ranker_3.1_*.sh" \
    --exclude="${NAME}/tests/test_v7_*" \
    --exclude="${NAME}/.env" \
    --exclude="${NAME}/*_retrieval" \
    --exclude="${NAME}/*_retrieval_*" \
    --exclude="${NAME}/datasets" \
    --exclude="${NAME}/models" \
    --exclude="${NAME}/logs" \
    --exclude="${NAME}/outputs" \
    "$NAME" | tar -C "$tmpdir" -xf -
fi

# Placeholders only (runtime assets stay out of the tarball)
for d in datasets models logs; do
  mkdir -p "$STAGE/$d"
  if [[ -f "$ROOT/$d/.gitkeep" ]]; then
    cp -a "$ROOT/$d/.gitkeep" "$STAGE/$d/.gitkeep"
  else
    : >"$STAGE/$d/.gitkeep"
  fi
done

mkdir -p "$OUT_DIR"
tar -C "$tmpdir" -czf "$OUT" "$NAME"

# Keep the exact tree that was archived so it can be pushed to GitHub or
# copied into an anonymous mirror without unpacking the tarball.
if [[ "$RELEASE_STAGE" == "$ROOT" || "$RELEASE_STAGE" == "/" ]]; then
  echo "ERROR: unsafe release stage: $RELEASE_STAGE" >&2
  exit 1
fi
rm -rf "$RELEASE_STAGE"
mv "$STAGE" "$RELEASE_STAGE"

echo "Done: $OUT"
echo "Staged source: $RELEASE_STAGE"
ls -lh "$OUT"
echo "Entries: $(tar tzf "$OUT" | wc -l)"
echo "gitkeeps:"
tar tzf "$OUT" | grep gitkeep || true
# Fail if any domain working tree leaked (base or *_retrieval_* lanes)
bad="$(tar tzf "$OUT" | grep -E '(^|/)[^/]*_retrieval(/|_|$)' | head -10 || true)"
if [[ -n "$bad" ]]; then
  echo "WARNING: unexpected paths (incl. domain dirs):" >&2
  echo "$bad" >&2
  exit 1
fi
bad2="$(tar tzf "$OUT" | grep -E '\.venv/|__pycache__|egg-info|arxiv-metadata|bge-large|AliMaatouk|\.pytest_cache' | head -5 || true)"
if [[ -n "$bad2" ]]; then
  echo "WARNING: unexpected paths:" >&2
  echo "$bad2" >&2
  exit 1
fi
echo "Sanity: OK (code-only tree; no domain *_retrieval* assets)"
