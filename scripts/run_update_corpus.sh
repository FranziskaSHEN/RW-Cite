#!/usr/bin/env bash
# Topics supplement (GLM) + BGE embeddings for papers missing from HF Topics.
#
# After metadata incremental:
#   bash scripts/run_update_corpus.sh --skip-fetch
# Topics only / embeds only:
#   bash scripts/run_update_corpus.sh --skip-fetch --skip-embeddings
#   bash scripts/run_update_corpus.sh --skip-fetch --skip-topics
# Heuristic (no LLM):
#   bash scripts/run_update_corpus.sh --skip-fetch --topic-backend heuristic
set -euo pipefail
ROOT="${RWCITE_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$ROOT"
export RWCITE_ROOT="$ROOT"
module load cuda-12.8 2>/dev/null || true
# shellcheck source=rwcite_python.sh
source "$(dirname "$0")/rwcite_python.sh"

# Prefer RWCITE_LLM_API_KEY; accept LitBench key name if already exported.
if [[ -z "${RWCITE_LLM_API_KEY:-}" && -n "${LITBENCH_LLM_API_KEY:-}" ]]; then
  export RWCITE_LLM_API_KEY="$LITBENCH_LLM_API_KEY"
fi

exec "$PY" -m rwcite.cli.update_corpus "$@"
