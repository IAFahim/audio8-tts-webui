#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if [[ -f "$ROOT_DIR/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT_DIR/.env"
  set +a
fi

REMOTE_HOST="${AUDIO8_REMOTE_HOST:-}"
REMOTE_CONTROL="${AUDIO8_REMOTE_CONTROL:-$REMOTE_HOST}"
REMOTE_SERVICE="${AUDIO8_REMOTE_SERVICE:-audio8-tts.service}"
PORT="${AUDIO8_PORT:-8999}"

ACTION="start"
DETACH=0
NO_BROWSER=0
KEEP_RUNNING=0

usage() {
  cat <<'EOF'
Usage: ./remote.sh [start|stop|status] [options]

One-click launcher for a remote Audio8-TTS GPU server over SSH.
Configures via .env (or env vars):
  AUDIO8_REMOTE_HOST     SSH target for local port-forwarding (e.g. user@192.168.x.x)
  AUDIO8_REMOTE_CONTROL  SSH target with systemd permission (defaults to AUDIO8_REMOTE_HOST)
  AUDIO8_REMOTE_SERVICE  systemd unit name (default: audio8-tts.service)
  AUDIO8_PORT            Local & remote port (default: 8999)

Options:
  --detach        Start service + background SSH tunnel, open browser, and return
  --no-browser    Do not open the browser automatically
  --keep-running  On Ctrl+C in foreground mode, keep the remote systemd service running
  -h, --help      Show this help message
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    start|stop|status)
      ACTION="$1"
      shift
      ;;
    --detach)
      DETACH=1
      shift
      ;;
    --no-browser)
      NO_BROWSER=1
      shift
      ;;
    --keep-running)
      KEEP_RUNNING=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

PID_FILE="/tmp/audio8-tts-tunnel-${PORT}.pid"
APP_URL="http://127.0.0.1:${PORT}"

stop_tunnel() {
  if [[ -f "$PID_FILE" ]]; then
    local pid
    pid="$(cat "$PID_FILE" 2>/dev/null || true)"
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
    fi
    rm -f "$PID_FILE"
  fi
  if command -v pkill >/dev/null 2>&1; then
    pkill -f "ssh .*127\.0\.0\.1:${PORT}:127\.0\.0\.1:${PORT}" 2>/dev/null || true
  fi
}

if [[ "$ACTION" == "status" ]]; then
  if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE" 2>/dev/null || echo 0)" 2>/dev/null; then
    echo "SSH tunnel: active (PID $(cat "$PID_FILE"), $APP_URL)"
  else
    echo "SSH tunnel: inactive"
  fi
  if [[ -n "$REMOTE_CONTROL" ]]; then
    ssh -o ConnectTimeout=5 -o BatchMode=yes "$REMOTE_CONTROL" \
      "systemctl is-active '$REMOTE_SERVICE' || true" 2>/dev/null | sed 's/^/Remote service: /' || echo "Remote service: unreachable"
  else
    echo "Remote service: AUDIO8_REMOTE_HOST / AUDIO8_REMOTE_CONTROL not configured (.env)"
  fi
  exit 0
fi

if [[ -z "$REMOTE_HOST" ]]; then
  echo "Error: AUDIO8_REMOTE_HOST is not set. Copy .env.example to .env and configure it." >&2
  exit 1
fi

if [[ "$ACTION" == "stop" ]]; then
  echo "[remote.sh] Closing SSH tunnel on port ${PORT}..."
  stop_tunnel
  echo "[remote.sh] Stopping ${REMOTE_SERVICE} on ${REMOTE_CONTROL}..."
  ssh -o ConnectTimeout=10 "$REMOTE_CONTROL" "systemctl stop '$REMOTE_SERVICE'" || true
  echo "[remote.sh] Stopped."
  exit 0
fi

echo "[remote.sh] Starting ${REMOTE_SERVICE} on ${REMOTE_CONTROL}..."
ssh -o ConnectTimeout=10 "$REMOTE_CONTROL" "systemctl start '$REMOTE_SERVICE'"

stop_tunnel
echo "[remote.sh] Opening SSH tunnel ${APP_URL} -> ${REMOTE_HOST}:127.0.0.1:${PORT}..."
ssh -f -N \
  -o BatchMode=yes \
  -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=30 \
  -L "127.0.0.1:${PORT}:127.0.0.1:${PORT}" \
  "$REMOTE_HOST"

TUNNEL_PID="$(pgrep -fn "ssh .*127\.0\.0\.1:${PORT}:127\.0\.0\.1:${PORT}" || true)"
if [[ -n "$TUNNEL_PID" ]]; then
  printf '%s' "$TUNNEL_PID" > "$PID_FILE"
fi

echo "[remote.sh] Waiting for ${APP_URL}/api/health..."
READY=0
for _ in $(seq 1 60); do
  if curl -fsS "${APP_URL}/api/health" 2>/dev/null | grep -q '"ready":true'; then
    READY=1
    break
  fi
  sleep 1
done

if [[ "$READY" -ne 1 ]]; then
  echo "[remote.sh] Timed out waiting for remote server health check." >&2
  stop_tunnel
  exit 1
fi

echo "[remote.sh] Ready at ${APP_URL}"
if [[ "$NO_BROWSER" -eq 0 ]]; then
  if command -v xdg-open >/dev/null 2>&1; then
    xdg-open "$APP_URL" >/dev/null 2>&1 || true
  elif command -v open >/dev/null 2>&1; then
    open "$APP_URL" >/dev/null 2>&1 || true
  fi
fi

if [[ "$DETACH" -eq 1 ]]; then
  echo "[remote.sh] Running in detached mode. Use './remote.sh stop' when done."
  exit 0
fi

cleanup() {
  echo ""
  echo "[remote.sh] Cleaning up..."
  stop_tunnel
  if [[ "$KEEP_RUNNING" -eq 0 ]]; then
    echo "[remote.sh] Stopping ${REMOTE_SERVICE} on ${REMOTE_CONTROL} to free GPU VRAM..."
    ssh -o ConnectTimeout=10 "$REMOTE_CONTROL" "systemctl stop '$REMOTE_SERVICE'" >/dev/null 2>&1 || true
  fi
}
trap cleanup INT TERM EXIT

echo "[remote.sh] Press Ctrl+C to close the tunnel and stop the remote GPU service."
while [[ -n "${TUNNEL_PID:-}" ]] && kill -0 "$TUNNEL_PID" 2>/dev/null; do
  sleep 2
done
