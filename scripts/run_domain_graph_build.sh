#!/usr/bin/env bash
# Unified domain graph build: retrieve (year∧N∧τ) → incremental download → native seed.
# Default: --seed-only (no expand). Corpus stays under domain retrieval root (reuse zips).
#
# Usage:
#   bash scripts/run_domain_graph_build.sh --domain sqc [--cutover] [--seed-only|--expand]
#   DOMAIN=sqc CUTOVER=1 bash scripts/run_domain_graph_build.sh
#   # Full rebuild after wiping domain corpus (e.g. ewm):
#   bash scripts/run_domain_graph_build.sh --domain ewm --cutover --seed-only --reset
set -euo pipefail

ROOT="${RWCITE_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$ROOT"

DOMAIN="${DOMAIN:-}"
CUTOVER="${CUTOVER:-0}"
SEED_ONLY="${SEED_ONLY:-1}"
SKIP_RETRIEVAL="${SKIP_RETRIEVAL:-0}"
SKIP_DOWNLOAD="${SKIP_DOWNLOAD:-0}"
ALLOW_DOWNLOAD="${ALLOW_DOWNLOAD:-1}"
RESET="${RESET:-0}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --domain) DOMAIN="$2"; shift 2 ;;
    --cutover) CUTOVER=1; shift ;;
    --seed-only) SEED_ONLY=1; shift ;;
    --expand) SEED_ONLY=0; shift ;;
    --skip-retrieval) SKIP_RETRIEVAL=1; shift ;;
    --skip-download) SKIP_DOWNLOAD=1; shift ;;
    --no-allow-download) ALLOW_DOWNLOAD=0; shift ;;
    --reset) RESET=1; shift ;;
    -h|--help)
      sed -n '2,10p' "$0"
      exit 0
      ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$DOMAIN" ]]; then
  echo "ERROR: --domain required (key in configs/domains.yaml)" >&2
  exit 2
fi

PY="${RWCITE_PYTHON:-${PY:-python3}}"
if [[ -x "${ROOT}/.venv/bin/python" && -z "${RWCITE_PYTHON:-}" && "${PY}" == "python3" ]]; then
  PY="${ROOT}/.venv/bin/python"
fi

module load cuda-12.8 2>/dev/null || true
export RWCITE_ROOT="$ROOT"
export RWCITE_ASSET_SOURCE="${RWCITE_ASSET_SOURCE:-${RWCITE_DATA_ROOT:-}}"
export CUDA_VISIBLE_DEVICES
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

# Resolve domain paths from domains.yaml + config
mapfile -t _paths < <("$PY" - <<PY
import yaml
from pathlib import Path
root = Path("$ROOT")
dom = "$DOMAIN"
dy = yaml.safe_load((root / "configs/domains.yaml").read_text())
block = (dy.get("domains") or {}).get(dom)
if not block:
    raise SystemExit(f"unknown domain: {dom}")
cfg = block["domain_config"]
gexf_prod = block["gexf"]
ret_nodes = block.get("retrieval_nodes", "")
cfg_path = root / cfg
ycfg = yaml.safe_load(cfg_path.read_text())
working = ycfg["data_downloading"]["download_directory"].rstrip("/") + "/"
rr_gexf = working + "description/test_graph_rr.gexf"
print(cfg)
print(working)
print(ret_nodes)
print(gexf_prod)
print(rr_gexf)
PY
)

CONFIG="${_paths[0]}"
WORKING_DIR="${_paths[1]}"
RET_NODES="${_paths[2]}"
GEXF_PROD="${_paths[3]}"
OUT_GEXF="${_paths[4]}"
WORK_DIR="${WORKING_DIR}description/rebuild_year10"
LOG_DIR="$ROOT/logs"
LOG_FILE="$LOG_DIR/domain_graph_build_${DOMAIN}.log"
mkdir -p "$LOG_DIR" "$WORK_DIR" "$(dirname "$OUT_GEXF")"

ZIP_DIR="${WORKING_DIR}research_papers_zip"
PAPERS_DIR="${WORKING_DIR}research_papers"
n_zip=0
n_papers=0
[[ -d "$ZIP_DIR" ]] && n_zip=$(find "$ZIP_DIR" -maxdepth 1 \( -name '*.tar.gz' -o -name '*.zip' \) 2>/dev/null | wc -l | tr -d ' ')
[[ -d "$PAPERS_DIR" ]] && n_papers=$(find "$PAPERS_DIR" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l | tr -d ' ')

echo "== domain graph build: domain=${DOMAIN} =="
echo "  config:     $CONFIG"
echo "  working:    $WORKING_DIR  (corpus root; skip_if_downloaded)"
echo "  zip reuse:  $ZIP_DIR (n≈${n_zip})"
echo "  papers:     $PAPERS_DIR (n≈${n_papers})"
echo "  work-dir:   $WORK_DIR"
echo "  out-gexf:   $OUT_GEXF"
echo "  seed-only:  $SEED_ONLY  cutover=$CUTOVER  reset=$RESET"
echo "  log:        $LOG_FILE"

EXTRA=()
[[ "$SKIP_RETRIEVAL" == "1" ]] && EXTRA+=(--skip-retrieval)
[[ "$SKIP_DOWNLOAD" == "1" ]] && EXTRA+=(--skip-download)
[[ "$SEED_ONLY" == "1" ]] && EXTRA+=(--seed-only)
[[ "$RESET" == "1" ]] && EXTRA+=(--reset)
if [[ "$ALLOW_DOWNLOAD" == "1" ]]; then
  EXTRA+=(--allow-download)
else
  EXTRA+=(--no-allow-download)
fi

# Corpus stays under WORKING_DIR via config; work-dir is ckpt only. --reset clears ckpt.
set -o pipefail
"$PY" -m rwcite.cli.graph_pipeline \
  --config "$CONFIG" \
  --work-dir "$WORK_DIR" \
  --out-gexf "$OUT_GEXF" \
  "${EXTRA[@]}" \
  2>&1 | tee "$LOG_FILE"

# Graph summary JSON
SUMMARY="$WORK_DIR/graph_summary.json"
"$PY" - <<PY
import hashlib, json
from pathlib import Path
import networkx as nx

root = Path("$ROOT")
gexf = root / "$OUT_GEXF"
ret_path = root / "$RET_NODES"
g = nx.read_gexf(str(gexf))
if not g.is_directed():
    g = g.to_directed()
nodes = list(g.nodes())
edges = list(g.edges())
outdeg = dict(g.out_degree())
elig = [u for u in nodes if outdeg.get(u, 0) >= 10]
sent = 0
for _u, _v, d in g.edges(data=True):
    s = (d or {}).get("sentence") or (d or {}).get("cite_sentence") or ""
    if str(s).strip():
        sent += 1
ret = set()
if ret_path.exists():
    raw = json.loads(ret_path.read_text())
    ret = set(raw.keys() if isinstance(raw, dict) else raw)
h = hashlib.sha256()
with open(gexf, "rb") as f:
    for chunk in iter(lambda: f.read(1 << 20), b""):
        h.update(chunk)
summary = {
    "domain": "$DOMAIN",
    "gexf": "$OUT_GEXF",
    "gexf_sha256": h.hexdigest(),
    "nodes": len(nodes),
    "edges": len(edges),
    "elig10": len(elig),
    "sentence_rate": round(sent / max(1, len(edges)), 4),
    "elig10_in_ret": sum(1 for u in elig if u in ret),
    "elig10_in_ret_frac": round(sum(1 for u in elig if u in ret) / max(1, len(elig)), 4),
    "ret_in_graph_frac": round(sum(1 for u in ret if u in g) / max(1, len(ret)), 4) if ret else None,
    "n_zip_reuse_approx": int("$n_zip"),
    "n_papers_dirs_approx": int("$n_papers"),
    "seed_only": "$SEED_ONLY" == "1",
}
Path("$SUMMARY").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
PY

if [[ "$CUTOVER" == "1" ]]; then
  echo "== cutover domains.yaml gexf → $OUT_GEXF =="
  TS="$(date +%Y%m%d_%H%M%S)"
  if [[ -f "$GEXF_PROD" && "$GEXF_PROD" != "$OUT_GEXF" ]]; then
    BAK="${GEXF_PROD}.pre_unified_${TS}.bak"
    cp -a "$GEXF_PROD" "$BAK"
    echo "  bak old prod: $BAK"
  fi
  if [[ -f "$OUT_GEXF" && ! -f "${WORKING_DIR}description/test_graph.gexf.bak_pre_rr_${TS}" ]]; then
    if [[ -f "${WORKING_DIR}description/test_graph.gexf" ]]; then
      cp -a "${WORKING_DIR}description/test_graph.gexf" \
        "${WORKING_DIR}description/test_graph.gexf.pre_unified_${TS}.bak"
    fi
  fi
  "$PY" - <<PY
from pathlib import Path
import re
root = Path("$ROOT")
path = root / "configs/domains.yaml"
text = path.read_text()
dom = "$DOMAIN"
new_gexf = "$OUT_GEXF"
# Replace gexf line inside the domain block only (first gexf: under "  {dom}:")
pattern = rf"(  {re.escape(dom)}:\n(?:.*\n)*?    gexf: )([^\n]+)"
m = re.search(pattern, text)
if not m:
    raise SystemExit(f"cannot find gexf for domain {dom}")
text2 = text[: m.start(2)] + new_gexf + text[m.end(2) :]
path.write_text(text2)
print(f"  domains.yaml {dom}.gexf → {new_gexf}")
PY
fi

echo "== diag retrieval closure =="
DIAG_OUT="${WORKING_DIR}data/retrieval_diag.json"
mkdir -p "$(dirname "$DIAG_OUT")"
"$PY" -u scripts/diag_domain_retrieval.py --domains "$DOMAIN" --out "$DIAG_OUT" \
  2>&1 | tee -a "$LOG_FILE" || true

echo "== done domain=${DOMAIN} summary=$SUMMARY =="
