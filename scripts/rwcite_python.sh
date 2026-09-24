# Shared Python resolver for RW-Cite shells.
# Usage (after ROOT is set):
#   source "$(dirname "$0")/rwcite_python.sh"
#
# Prefer: RWCITE_PYTHON > .venv > python3.1x on PATH
# Requires Python >= 3.10. Bare names resolve even under minimal nohup PATH.
: "${ROOT:?ROOT must be set before sourcing rwcite_python.sh}"

_rwcite_py_ok() {
  local bin="$1"
  [[ -n "$bin" && -x "$bin" ]] || return 1
  "$bin" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null
}

_rwcite_which() {
  local token="$1" found root
  [[ -n "$token" ]] || return 1
  if [[ "$token" == /* ]]; then
    [[ -x "$token" ]] || return 1
    echo "$token"
    return 0
  fi
  found="$(command -v "$token" 2>/dev/null || true)"
  if [[ -n "$found" && -x "$found" ]]; then
    echo "$found"
    return 0
  fi
  for root in \
    "${ROOT}/.venv/bin"
  do
    if [[ -x "${root}/${token}" ]]; then
      echo "${root}/${token}"
      return 0
    fi
  done
  return 1
}

_rwcite_resolve_python() {
  local cand resolved
  local -a candidates=()

  if [[ -n "${RWCITE_PYTHON:-}" ]]; then
    if resolved="$(_rwcite_which "$RWCITE_PYTHON")" && _rwcite_py_ok "$resolved"; then
      echo "$resolved"
      return 0
    fi
    echo "WARN: RWCITE_PYTHON=${RWCITE_PYTHON} missing or <3.10; auto-detecting" >&2
  fi
  if [[ -n "${PY:-}" && "${PY}" != "python3" ]]; then
    candidates+=("$PY")
  fi
  candidates+=(
    "${ROOT}/.venv/bin/python"
    "python3.12"
    "python3.11"
    "python3.10"
    "python3"
  )

  for cand in "${candidates[@]}"; do
    if resolved="$(_rwcite_which "$cand")" && _rwcite_py_ok "$resolved"; then
      echo "$resolved"
      return 0
    fi
  done
  echo "ERROR: need Python >= 3.10 for RW-Cite." >&2
  echo "  Example:" >&2
  echo "  python3.11 -m venv .venv && source .venv/bin/activate" >&2
  return 1
}

PY="$(_rwcite_resolve_python)" || exit 1
export PY
echo "RW-Cite Python: $PY ($("$PY" -c 'import sys; print("%d.%d" % sys.version_info[:2])'))" >&2
