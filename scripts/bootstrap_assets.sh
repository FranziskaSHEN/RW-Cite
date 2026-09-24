#!/usr/bin/env bash
# Link shared assets needed by graph build + Ranker B1 into RW-Cite.
#
# Preferred: set RWCITE_ASSET_SOURCE to a data root that already contains
# arxiv metadata/topics/embeds, BGE, and SciBERT.
#
# Usage:
#   RWCITE_ASSET_SOURCE=/path/to/data_root bash scripts/bootstrap_assets.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

ASSET_SRC="${RWCITE_ASSET_SOURCE:-}"

link_one() {
  local rel="$1"
  local src="$ASSET_SRC/$rel"
  local dst="$ROOT/$rel"
  mkdir -p "$(dirname "$dst")"
  if [[ -e "$dst" || -L "$dst" ]]; then
    echo "  skip exists: $rel"
    return 0
  fi
  if [[ ! -e "$src" ]]; then
    echo "  MISSING source: $src"
    return 0
  fi
  ln -s "$src" "$dst"
  echo "  linked: $rel -> $src"
}

echo "== RW-Cite bootstrap_assets =="
echo "  root:   $ROOT"
echo "  source: ${ASSET_SRC:-NONE}"

if [[ -z "$ASSET_SRC" ]]; then
  echo "ERROR: set RWCITE_ASSET_SOURCE to a data root containing:" >&2
  echo "  datasets/arxiv-metadata-oai-snapshot.json" >&2
  echo "  datasets/arxiv_topics/" >&2
  echo "  datasets/topic_level_embeds/..." >&2
  echo "  models/base/bge-large-en-v1.5/" >&2
  echo "  models/base/scibert_scivocab_uncased/   (B1 CE cold-start)" >&2
  exit 1
fi

link_one "datasets/arxiv-metadata-oai-snapshot.json"
link_one "datasets/arxiv_topics"
link_one "datasets/arxiv_topics_supplement.jsonl"
link_one "datasets/topic_level_embeds"
link_one "models/base/bge-large-en-v1.5"
link_one "models/base/scibert_scivocab_uncased"

echo "== done. Metadata: run_metadata_fetch.sh | Graph: run_domain_graph_build.sh | Ranker: run_domain_rr_c2s.sh =="
