#!/usr/bin/env bash
# Pack latest arXiv metadata / Topics supplement / embeddings for separate release
# (not part of the GitHub source tarball). See docs/DATASETS_HF_UPLOAD.md.
#
# Usage:
#   bash scripts/pack_datasets_release.sh --tier delta
#   bash scripts/pack_datasets_release.sh --tier full --tag 2026-08-25
#   OUT_DIR=/data/releases bash scripts/pack_datasets_release.sh --tier delta --tag 20260825
#
# Tiers:
#   delta  — topics supplement + embeddings supplement + corpus_state (+ MANIFEST)
#   full   — delta + arxiv-metadata-oai-snapshot.json
#
# Outputs under OUT_DIR (default: releases/huggingface):
#   RW-Cite-datasets-r<tag>-<tier>/     staged tree (HF-upload friendly)
#   RW-Cite-datasets-r<tag>-<tier>.tar.gz
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${OUT_DIR:-$ROOT/releases/huggingface/corpus}"
TIER="delta"
TAG=""
MAKE_TAR="${MAKE_TAR:-1}"
CHECKSUMS="${CHECKSUMS:-0}"

usage() {
  sed -n '2,20p' "$0"
  exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tier) TIER="${2,,}"; shift 2 ;;
    --tag) TAG="$2"; shift 2 ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    --no-tar) MAKE_TAR=0; shift ;;
    --checksums) CHECKSUMS=1; shift ;;
    -h|--help) usage 0 ;;
    *)
      echo "Unknown arg: $1" >&2
      usage 1
      ;;
  esac
done

if [[ "$TIER" != "delta" && "$TIER" != "full" ]]; then
  echo "ERROR: --tier must be delta or full" >&2
  exit 1
fi

cd "$ROOT"
export RWCITE_ROOT="$ROOT"

if [[ -z "$TAG" ]]; then
  if [[ -f datasets/corpus_state.json ]]; then
    TAG="$(python3 -c "import json; print(json.load(open('datasets/corpus_state.json')).get('last_metadata_until','')[:10].replace('-',''))" 2>/dev/null || true)"
  fi
  if [[ -z "$TAG" ]]; then
    TAG="$(date -u +%Y%m%d)"
  fi
fi
# Normalize 2026-08-25 -> 20260825; drop other punctuation
TAG="${TAG//-/}"
TAG="$(printf '%s' "$TAG" | tr -cd '0-9A-Za-z')"
if [[ -z "$TAG" ]]; then
  echo "ERROR: empty tag after normalize" >&2
  exit 1
fi

META="datasets/arxiv-metadata-oai-snapshot.json"
TOPICS="datasets/arxiv_topics_supplement.jsonl"
EMBEDS="datasets/topic_level_embeds/supplement_embeddings.parquet"
STATE="datasets/corpus_state.json"

need=("$TOPICS" "$EMBEDS")
if [[ "$TIER" == "full" ]]; then
  need+=("$META")
fi
for f in "${need[@]}"; do
  if [[ ! -f "$f" ]]; then
    echo "ERROR: missing required file: $f" >&2
    exit 1
  fi
done

STAGE_NAME="RW-Cite-datasets-r${TAG}-${TIER}"
STAGE="${OUT_DIR}/${STAGE_NAME}"
ARCHIVE="${OUT_DIR}/${STAGE_NAME}.tar.gz"

echo "Packing datasets release"
echo "  tier=$TIER tag=$TAG"
echo "  stage=$STAGE"

rm -rf "$STAGE"
mkdir -p "$STAGE/datasets/topic_level_embeds"

copy_one() {
  local src="$1" dst="$2"
  mkdir -p "$(dirname "$dst")"
  # Prefer hardlink on same filesystem; else copy
  if ln "$src" "$dst" 2>/dev/null; then
    echo "  link $src"
  else
    cp -a "$src" "$dst"
    echo "  copy $src"
  fi
}

copy_one "$TOPICS" "$STAGE/datasets/arxiv_topics_supplement.jsonl"
copy_one "$EMBEDS" "$STAGE/datasets/topic_level_embeds/supplement_embeddings.parquet"
if [[ -f "$STATE" ]]; then
  copy_one "$STATE" "$STAGE/datasets/corpus_state.json"
fi
if [[ "$TIER" == "full" ]]; then
  copy_one "$META" "$STAGE/datasets/arxiv-metadata-oai-snapshot.json"
fi

# Short README inside the pack
cat >"$STAGE/README.md" <<EOF
---
pretty_name: RW-Cite arXiv corpus
task_categories:
  - text-retrieval
  - feature-extraction
language:
  - en
license: other
---

# RW-Cite datasets pack (\`${TIER}\`, tag \`r${TAG}\`)

This archive is **data only**. Code: install RW-Cite from GitHub / source tarball.

## Contents

| Path | Role |
|------|------|
| \`datasets/arxiv_topics_supplement.jsonl\` | Incremental L1/L2/L3 topics (GLM) |
| \`datasets/topic_level_embeds/supplement_embeddings.parquet\` | Incremental BGE topic-level embeddings |
| \`datasets/corpus_state.json\` | Update window / counts |
$([ "$TIER" = "full" ] && echo "| \`datasets/arxiv-metadata-oai-snapshot.json\` | Full merged OAI metadata JSONL |")

## Base corpora (not included)

Keep Hugging Face base assets separate (read-only):

- Topics: \`AliMaatouk/arXiv_Topics\` → local \`datasets/arxiv_topics/\`
- Embeddings: \`AliMaatouk/arXiv-Topics-Embeddings\` → local \`datasets/topic_level_embeds/\`

RW-Cite merges HF base + these supplements at runtime.

## Install into an RW-Cite tree

\`\`\`bash
# from extracted pack root
rsync -a datasets/ /path/to/RW-Cite/datasets/
\`\`\`

See \`docs/DATASETS_HF_UPLOAD.md\` in the RW-Cite repo for Hugging Face Hub upload.
EOF

# MANIFEST.json
python3 - "$STAGE" "$TIER" "$TAG" "$CHECKSUMS" <<'PY'
import hashlib, json, os, sys
from pathlib import Path

stage = Path(sys.argv[1])
tier = sys.argv[2]
tag = sys.argv[3]
do_hash = sys.argv[4] == "1"

files = []
for path in sorted(stage.rglob("*")):
    if not path.is_file() or path.name in ("MANIFEST.json", "README.md"):
        continue
    rel = path.relative_to(stage).as_posix()
    st = path.stat()
    entry = {
        "path": rel,
        "bytes": st.st_size,
        "mtime_utc": __import__("datetime").datetime.utcfromtimestamp(st.st_mtime).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    if do_hash:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024 * 8), b""):
                h.update(chunk)
        entry["sha256"] = h.hexdigest()
    files.append(entry)

state = {}
sp = stage / "datasets" / "corpus_state.json"
if sp.exists():
    state = json.loads(sp.read_text(encoding="utf-8"))

manifest = {
    "name": "RW-Cite-datasets",
    "tier": tier,
    "tag": f"r{tag}",
    "created_utc": __import__("datetime").datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
    "base_topics": "AliMaatouk/arXiv_Topics",
    "base_embeddings": "AliMaatouk/arXiv-Topics-Embeddings",
    "embedder": "BAAI/bge-large-en-v1.5",
    "topic_generator": "GLM-5.1 (OpenAI-compatible) / heuristic fallback",
    "corpus_state": state,
    "files": files,
    "notes": [
        "Does not include HF base Topics/Embeddings caches.",
        "delta = supplements only; full = supplements + metadata snapshot.",
    ],
}
(stage / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
print(f"Wrote MANIFEST.json ({len(files)} files)")
PY

cp -a "$ROOT/docs/DATASETS_HF_UPLOAD.md" "$STAGE/DATASETS_HF_UPLOAD.md" 2>/dev/null || true

mkdir -p "$OUT_DIR"
if [[ "$MAKE_TAR" == "1" ]]; then
  echo "Creating $ARCHIVE ..."
  tar -C "$OUT_DIR" -czf "$ARCHIVE" "$STAGE_NAME"
  ls -lh "$ARCHIVE"
fi

echo "Done."
echo "  staged: $STAGE"
[[ "$MAKE_TAR" == "1" ]] && echo "  archive: $ARCHIVE"
echo "Upload guide: docs/DATASETS_HF_UPLOAD.md"
