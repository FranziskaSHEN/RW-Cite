#!/usr/bin/env bash
# BGE-encode supplement topics that lack embeddings yet.
#
#   CUDA_VISIBLE_DEVICES=0 nohup bash scripts/run_supplement_embeddings.sh \
#     > logs/supplement_embeddings.nohup 2>&1 &
set -euo pipefail
ROOT="${RWCITE_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$ROOT"
export RWCITE_ROOT="$ROOT"
module load cuda-12.8 2>/dev/null || true
# shellcheck source=rwcite_python.sh
source "$(dirname "$0")/rwcite_python.sh"

EMBED_BATCH_SIZE="${EMBED_BATCH_SIZE:-256}"

exec "$PY" -m rwcite.cli.update_corpus \
  --skip-fetch \
  --skip-topics \
  --embed-batch-size "$EMBED_BATCH_SIZE" \
  "$@"
