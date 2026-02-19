#!/usr/bin/env bash
# Launcher for local model audit mini app.

set -Eeuo pipefail

APP_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${APP_DIR}/.." && pwd)"
HOST="127.0.0.1"
PORT="8765"
SESSIONS_DIR="${HOME}/.codex/sessions"
REPORTS_DIR="${APP_DIR}"
MAX_RETENTION_DAYS="31"
MAX_CSV_ROWS="0"

usage() {
  cat <<USAGE
Usage: $(basename "$0") [-p PORT] [-h HOST] [-s SESSIONS_DIR] [-r REPORTS_DIR] [--max-retention-days DAYS] [--max-csv-rows N]

Options:
  -p PORT           Port to bind (default: ${PORT})
  -h HOST           Host to bind (default: ${HOST})
  -s SESSIONS_DIR   Codex sessions directory (default: ${SESSIONS_DIR})
  -r REPORTS_DIR    Reports directory (default: ${REPORTS_DIR})
  --max-retention-days DAYS  Keep only newest DAYS of events in typed CSV datasets (default: ${MAX_RETENTION_DAYS})
  --max-csv-rows N           Optional hard row cap after retention (default: ${MAX_CSV_ROWS}, disabled)
  --help            Show this help

Example:
  $(basename "$0") -p 6969
USAGE
}

while (($# > 0)); do
  case "$1" in
    -p)
      PORT="${2:-}"
      shift 2
      ;;
    -h)
      HOST="${2:-}"
      shift 2
      ;;
    -s)
      SESSIONS_DIR="${2:-}"
      shift 2
      ;;
    -r)
      REPORTS_DIR="${2:-}"
      shift 2
      ;;
    --max-retention-days)
      MAX_RETENTION_DAYS="${2:-}"
      shift 2
      ;;
    --max-csv-rows)
      MAX_CSV_ROWS="${2:-}"
      shift 2
      ;;
    --help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "${PORT}" || -z "${HOST}" || -z "${SESSIONS_DIR}" || -z "${REPORTS_DIR}" || -z "${MAX_RETENTION_DAYS}" || -z "${MAX_CSV_ROWS}" ]]; then
  echo "Missing required option value." >&2
  usage
  exit 1
fi

if [[ "${PORT}" =~ [^0-9] || "${MAX_RETENTION_DAYS}" =~ [^0-9] || "${MAX_CSV_ROWS}" =~ [^0-9] ]]; then
  echo "Invalid numeric option value." >&2
  exit 1
fi

PYTHON_BIN="python3"
if [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
  PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"
fi

exec "${PYTHON_BIN}" "${APP_DIR}/model_audit_local_app.py" \
  --host "${HOST}" \
  --port "${PORT}" \
  --sessions-dir "${SESSIONS_DIR}" \
  --reports-dir "${REPORTS_DIR}" \
  --max-retention-days "${MAX_RETENTION_DAYS}" \
  --max-csv-rows "${MAX_CSV_ROWS}"
