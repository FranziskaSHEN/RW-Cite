#!/usr/bin/env bash
# Build the two independent public deliverables described in releases/README.md.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

VERSION=""
DATA_TAG=""
DATA_TIER="full"
BUILD_CODE=1
BUILD_DATA=1
BUILD_CORPUS=0
BUILD_DOMAINS=1
PREPACKED_DOMAIN_ROOT="$ROOT/RW-Cite-Baseline"

usage() {
  cat <<'EOF'
Usage: bash scripts/prepare_public_releases.sh --version X.Y [options]

Options:
  --data-tag YYYYMMDD   Dataset revision tag (default: corpus state or today)
  --data-tier TIER      full or delta (default: full)
  --code-only           Build only releases/github/RW-Cite
  --data-only           Build only the Hugging Face deliverables
  --with-corpus         Include the optional global arXiv corpus pack
  --corpus-only         Build only the Hugging Face corpus pack
  --domains-only        Build only the ten Hugging Face domain packs
  --prepacked-domains P Read complete domain packs from P
  --checksums           Compute SHA-256 values for large dataset files
EOF
}

CHECKSUMS=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --version) VERSION="$2"; shift 2 ;;
    --data-tag) DATA_TAG="$2"; shift 2 ;;
    --data-tier) DATA_TIER="$2"; shift 2 ;;
    --code-only) BUILD_DATA=0; shift ;;
    --data-only) BUILD_CODE=0; shift ;;
    --with-corpus) BUILD_CORPUS=1; shift ;;
    --corpus-only) BUILD_CODE=0; BUILD_CORPUS=1; BUILD_DOMAINS=0; shift ;;
    --domains-only) BUILD_CODE=0; BUILD_CORPUS=0; shift ;;
    --prepacked-domains) PREPACKED_DOMAIN_ROOT="$2"; shift 2 ;;
    --checksums) CHECKSUMS=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "ERROR: unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ "$BUILD_CODE" == "1" && -z "$VERSION" ]]; then
  echo "ERROR: --version is required for the GitHub release" >&2
  exit 2
fi
if [[ "$DATA_TIER" != "full" && "$DATA_TIER" != "delta" ]]; then
  echo "ERROR: --data-tier must be full or delta" >&2
  exit 2
fi

if [[ "$BUILD_CODE" == "1" ]]; then
  OUT_DIR="$ROOT/releases/github" \
    bash scripts/pack_github_release.sh "$VERSION"
fi

if [[ "$BUILD_DATA" == "1" ]]; then
  if [[ "$BUILD_CORPUS" == "1" ]]; then
    args=(--tier "$DATA_TIER" --out-dir "$ROOT/releases/huggingface/corpus")
    [[ -n "$DATA_TAG" ]] && args+=(--tag "$DATA_TAG")
    [[ "$CHECKSUMS" == "1" ]] && args+=(--checksums)
    bash scripts/pack_datasets_release.sh "${args[@]}"
  fi
  if [[ "$BUILD_DOMAINS" == "1" ]]; then
    if compgen -G "$PREPACKED_DOMAIN_ROOT/RW-Cite-domain-*" >/dev/null; then
      py="${RWCITE_PYTHON:-python3}"
      args=(--source-root "$PREPACKED_DOMAIN_ROOT" --output-root "$ROOT/releases/huggingface/domains")
      [[ -n "$DATA_TAG" ]] && args+=(--tag "$DATA_TAG")
      [[ "$CHECKSUMS" == "1" ]] && args+=(--checksums)
      "$py" scripts/stage_prepacked_domains.py "${args[@]}"
    else
      args=(--all-paper-domains --out-dir "$ROOT/releases/huggingface/domains")
      [[ -n "$DATA_TAG" ]] && args+=(--tag "$DATA_TAG")
      [[ "$CHECKSUMS" == "1" ]] && args+=(--checksums)
      bash scripts/pack_domain_release.sh "${args[@]}"
    fi
  fi
fi

echo "Release directories ready under:"
[[ "$BUILD_CODE" == "1" ]] && echo "  $ROOT/releases/github"
[[ "$BUILD_DATA" == "1" ]] && echo "  $ROOT/releases/huggingface"
