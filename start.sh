#!/usr/bin/env bash
# start.sh — Start the V-Valley API and Web UI for local development.
#
# Usage:
#   ./start.sh                                   Start both API and Web UI
#   ./start.sh --api                             Start only the API server
#   ./start.sh --web                             Start only the Web UI
#   ./start.sh --api-port 8090 --web-port 5173  Start with custom ports
#
# Prerequisites: Python 3.12+
# The script will create a virtualenv and install dependencies automatically.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$ROOT_DIR/.venv"
API_PORT="${VVALLEY_API_PORT:-8080}"
WEB_PORT="${VVALLEY_WEB_PORT:-3000}"

# --- Colors (disabled if not a terminal) ---
if [ -t 1 ]; then
  BOLD='\033[1m'
  GREEN='\033[0;32m'
  YELLOW='\033[0;33m'
  RED='\033[0;31m'
  CYAN='\033[0;36m'
  RESET='\033[0m'
else
  BOLD='' GREEN='' YELLOW='' RED='' CYAN='' RESET=''
fi

info()  { printf "${CYAN}[info]${RESET}  %s\n" "$*"; }
ok()    { printf "${GREEN}[ok]${RESET}    %s\n" "$*"; }
warn()  { printf "${YELLOW}[warn]${RESET}  %s\n" "$*"; }
die()   { printf "${RED}[error]${RESET} %s\n" "$*" >&2; exit 1; }

usage() {
  cat <<EOF
Usage: ./start.sh [options]

Options:
  --api              Start only the API server
  --web              Start only the Web UI
  --api-port PORT    API port (default: ${API_PORT})
  --web-port PORT    Web UI port (default: ${WEB_PORT})
  -h, --help         Show this help message

Environment:
  VVALLEY_API_PORT   Default API port override
  VVALLEY_WEB_PORT   Default Web UI port override
EOF
}

is_valid_port() {
  [[ "$1" =~ ^[0-9]+$ ]] && [ "$1" -ge 1 ] && [ "$1" -le 65535 ]
}

ensure_port_free() {
  local port="$1"
  local label="$2"
  local flag="$3"

  if ! command -v lsof >/dev/null 2>&1; then
    return 0
  fi

  if lsof -nP -iTCP:"$port" -sTCP:LISTEN >/dev/null 2>&1; then
    die "${label} port ${port} is already in use. Choose another with ${flag}."
  fi
}

# --- Parse flags ---
START_API=true
START_WEB=true
while [[ $# -gt 0 ]]; do
  case "$1" in
    --api)
      START_API=true
      START_WEB=false
      shift
      ;;
    --web)
      START_API=false
      START_WEB=true
      shift
      ;;
    --api-port)
      [ $# -ge 2 ] || die "--api-port requires a port value."
      API_PORT="$2"
      shift 2
      ;;
    --web-port)
      [ $# -ge 2 ] || die "--web-port requires a port value."
      WEB_PORT="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "Unknown option: $1 (use --help)."
      ;;
  esac
done

if $START_API; then
  is_valid_port "$API_PORT" || die "Invalid API port '${API_PORT}' (expected 1-65535)."
fi
if $START_WEB; then
  is_valid_port "$WEB_PORT" || die "Invalid Web UI port '${WEB_PORT}' (expected 1-65535)."
fi

if $START_API && $START_WEB && [ "$API_PORT" -eq "$WEB_PORT" ]; then
  die "API and Web UI cannot use the same port (${API_PORT})."
fi

if $START_API; then
  ensure_port_free "$API_PORT" "API" "--api-port"
fi
if $START_WEB; then
  ensure_port_free "$WEB_PORT" "Web UI" "--web-port"
fi

# --- Check Python ---
PYTHON=""
for candidate in python3.12 python3.13 python3.14 python3; do
  if command -v "$candidate" &>/dev/null; then
    major="$("$candidate" -c 'import sys; print(sys.version_info[0])')"
    minor="$("$candidate" -c 'import sys; print(sys.version_info[1])')"
    if [ "$major" -ge 3 ] && [ "$minor" -ge 12 ]; then
      PYTHON="$candidate"
      break
    fi
  fi
done
[ -n "$PYTHON" ] || die "Python 3.12+ is required but not found. Install it and try again."
info "Using $($PYTHON --version) ($PYTHON)"

# --- Virtual environment ---
if [ ! -d "$VENV_DIR" ]; then
  info "Creating virtual environment in .venv ..."
  "$PYTHON" -m venv "$VENV_DIR"
  ok "Virtual environment created."
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
info "Activated virtualenv: $VENV_DIR"

# --- Install dependencies ---
if [ "$VENV_DIR/bin/pip" -ot "$ROOT_DIR/requirements.txt" ] 2>/dev/null || \
   ! python -c "import fastapi, uvicorn" &>/dev/null; then
  info "Installing Python dependencies ..."
  pip install -q -r "$ROOT_DIR/requirements.txt"
  ok "Dependencies installed."
else
  info "Dependencies already installed."
fi

# --- Ensure .env exists ---
if [ ! -f "$ROOT_DIR/.env" ]; then
  cp "$ROOT_DIR/.env.example" "$ROOT_DIR/.env"
  ok "Created .env from .env.example (edit it to add LLM keys if desired)."
fi

# --- Ensure data directory exists ---
mkdir -p "$ROOT_DIR/data"

# --- Cleanup on exit ---
PIDS=()
cleanup() {
  printf "\n"
  info "Shutting down ..."
  for pid in "${PIDS[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
    fi
  done
  wait 2>/dev/null || true
  ok "Stopped."
}
trap cleanup EXIT INT TERM

# --- Start API ---
if $START_API; then
  info "Starting API server on http://127.0.0.1:${API_PORT} ..."
  uvicorn apps.api.vvalley_api.main:app \
    --reload \
    --port "$API_PORT" \
    --log-level info \
    &
  PIDS+=($!)
fi

# --- Wait for API to be ready before starting web (so the UI can reach it) ---
if $START_API && $START_WEB; then
  if command -v curl >/dev/null 2>&1; then
    info "Waiting for API to become ready ..."
    for i in $(seq 1 30); do
      if curl -sf "http://127.0.0.1:${API_PORT}/healthz" >/dev/null 2>&1; then
        ok "API is ready."
        break
      fi
      if [ "$i" -eq 30 ]; then
        warn "API did not respond within 30s — starting Web UI anyway."
      fi
      sleep 1
    done
  else
    warn "curl not found; skipping API readiness check."
  fi
fi

# --- Start Web UI ---
if $START_WEB; then
  info "Starting Web UI on http://127.0.0.1:${WEB_PORT} ..."
  python -m http.server "$WEB_PORT" --directory "$ROOT_DIR/apps/web" &
  PIDS+=($!)
fi

# --- Summary ---
printf "\n"
printf "${BOLD}V-Valley is running:${RESET}\n"
$START_API && printf "  ${GREEN}API${RESET}  → http://127.0.0.1:${API_PORT}  (Swagger: http://127.0.0.1:${API_PORT}/docs)\n"
$START_WEB && printf "  ${GREEN}Web${RESET}  → http://127.0.0.1:${WEB_PORT}\n"
if $START_WEB && [ "$API_PORT" -ne 8080 ]; then
  warn "Web UI defaults to API port 8080. In the page, set API Base to http://127.0.0.1:${API_PORT}."
fi
printf "\nPress ${BOLD}Ctrl+C${RESET} to stop.\n\n"

# --- Wait for background processes ---
wait
