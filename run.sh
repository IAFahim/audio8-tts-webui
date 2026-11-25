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

HOST="${AUDIO8_HOST:-127.0.0.1}"
PORT="${AUDIO8_PORT:-8999}"
MODEL_DIR="${AUDIO8_MODEL_DIR:-$ROOT_DIR/models/Audio8-TTS-Realistic-Air}"
DATA_DIR="${AUDIO8_DATA_DIR:-$ROOT_DIR/data}"
DEVICE="${AUDIO8_DEVICE:-auto}"

NO_BROWSER=0
FORCE_CPU=0
DOWNLOAD_ONLY=0
REINSTALL=0

usage() {
  cat <<'EOF'
Usage: ./run.sh [options]

One-click setup & launcher for Audio8-TTS Realistic-Air Web UI (Linux / macOS).

Options:
  --port PORT       Port to bind (default: 8999 or $AUDIO8_PORT)
  --host HOST       Host to bind (default: 127.0.0.1 or $AUDIO8_HOST)
  --cpu             Force CPU mode (float32) instead of CUDA auto-detection
  --no-browser      Do not automatically open the browser
  --download-only   Download weights and exit without starting the server
  --reinstall       Reinstall Python dependencies into .venv
  -h, --help        Show this help message
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --port)
      PORT="$2"
      shift 2
      ;;
    --host)
      HOST="$2"
      shift 2
      ;;
    --cpu)
      FORCE_CPU=1
      DEVICE="cpu"
      shift
      ;;
    --no-browser)
      NO_BROWSER=1
      shift
      ;;
    --download-only)
      DOWNLOAD_ONLY=1
      shift
      ;;
    --reinstall)
      REINSTALL=1
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

export AUDIO8_HOST="$HOST"
export AUDIO8_PORT="$PORT"
export AUDIO8_MODEL_DIR="$MODEL_DIR"
export AUDIO8_DATA_DIR="$DATA_DIR"
export AUDIO8_DEVICE="$DEVICE"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

pick_python() {
  for cand in python3.11 python3.12 python3 python; do
    if command -v "$cand" >/dev/null 2>&1; then
      if "$cand" -c 'import sys; raise SystemExit(0 if (3, 10) <= sys.version_info[:2] <= (3, 12) else 1)' 2>/dev/null; then
        echo "$cand"
        return 0
      fi
    fi
  done
  if command -v python3 >/dev/null 2>&1; then
    echo "python3"
    return 0
  fi
  return 1
}

VENV_DIR="$ROOT_DIR/.venv"
VENV_PY="$VENV_DIR/bin/python"

if [[ ! -x "$VENV_PY" ]]; then
  if command -v uv >/dev/null 2>&1; then
    echo "[run.sh] Creating .venv with uv (Python 3.11)..."
    uv venv --python 3.11 "$VENV_DIR" || uv venv "$VENV_DIR"
  else
    PY_BIN="$(pick_python || true)"
    if [[ -z "$PY_BIN" ]]; then
      echo "Error: Python 3.10-3.12 (or uv) is required." >&2
      exit 1
    fi
    echo "[run.sh] Creating .venv with $PY_BIN..."
    "$PY_BIN" -m venv "$VENV_DIR"
  fi
fi

USE_CUDA=0
if [[ "$FORCE_CPU" -eq 0 ]] && command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
  USE_CUDA=1
fi

STAMP_KEY="$([[ "$USE_CUDA" -eq 1 ]] && echo "cu124" || echo "cpu")-$(cksum "$ROOT_DIR/requirements.txt" | awk '{print $1}')"
STAMP_FILE="$VENV_DIR/.deps_installed"

if [[ "$REINSTALL" -eq 1 ]] || [[ ! -f "$STAMP_FILE" ]] || [[ "$(cat "$STAMP_FILE")" != "$STAMP_KEY" ]]; then
  if [[ "$USE_CUDA" -eq 1 ]]; then
    echo "[run.sh] Installing PyTorch 2.5.0+cu124 (NVIDIA GPU detected)..."
    TORCH_ARGS=(torch==2.5.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu124)
  else
    echo "[run.sh] Installing PyTorch 2.5.0 (CPU mode)..."
    TORCH_ARGS=(torch==2.5.0 torchaudio==2.5.0)
  fi

  if command -v uv >/dev/null 2>&1; then
    export UV_NO_CONFIG=1
    export UV_HTTP_TIMEOUT=300
    export UV_CONCURRENT_DOWNLOADS=2
    uv pip install --python "$VENV_PY" "${TORCH_ARGS[@]}"
    uv pip install --python "$VENV_PY" -r "$ROOT_DIR/requirements.txt"
  else
    "$VENV_PY" -m pip install --upgrade pip
    "$VENV_PY" -m pip install "${TORCH_ARGS[@]}"
    "$VENV_PY" -m pip install -r "$ROOT_DIR/requirements.txt"
  fi

  printf '%s' "$STAMP_KEY" > "$STAMP_FILE"
fi

# Inject the venv's NVIDIA wheel libs (libnvrtc, libcudnn, etc.) alongside nix-ld/OpenGL on Linux
if [[ "$(uname -s)" == "Linux" ]]; then
  shopt -s nullglob
  nv_libs=""
  for d in "$VENV_DIR"/lib/python*/site-packages/nvidia/*/lib; do
    nv_libs="${nv_libs:+${nv_libs}:}${d}"
  done
  if [[ -d /run/opengl-driver/lib ]]; then
    nv_libs="${nv_libs:+${nv_libs}:}/run/opengl-driver/lib"
  fi
  if [[ -d /run/current-system/sw/share/nix-ld/lib ]]; then
    nv_libs="${nv_libs:+${nv_libs}:}/run/current-system/sw/share/nix-ld/lib"
    export NIX_LD="${NIX_LD:-/run/current-system/sw/share/nix-ld/lib/ld.so}"
    export NIX_LD_LIBRARY_PATH="${NIX_LD_LIBRARY_PATH:-/run/current-system/sw/share/nix-ld/lib}"
  fi
  if [[ -n "$nv_libs" ]]; then
    export LD_LIBRARY_PATH="${nv_libs}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
  fi
  shopt -u nullglob
fi

if [[ ! -f "$MODEL_DIR/model.safetensors" ]] || [[ ! -f "$MODEL_DIR/codec.pth" ]]; then
  echo "[run.sh] Model weights not found at $MODEL_DIR — downloading..."
  "$VENV_PY" "$ROOT_DIR/download_model.py" --dir "$MODEL_DIR"
fi

if [[ "$DOWNLOAD_ONLY" -eq 1 ]]; then
  echo "[run.sh] Model download complete."
  exit 0
fi

BROWSER_HOST="$HOST"
if [[ "$BROWSER_HOST" == "0.0.0.0" || "$BROWSER_HOST" == "::" ]]; then
  BROWSER_HOST="127.0.0.1"
fi
APP_URL="http://${BROWSER_HOST}:${PORT}"

if [[ "$NO_BROWSER" -eq 0 ]]; then
  (
    for _ in $(seq 1 120); do
      if command -v curl >/dev/null 2>&1; then
        if curl -fsS "${APP_URL}/api/health" 2>/dev/null | grep -q '"ready":true'; then
          break
        fi
      fi
      sleep 1
    done
    if command -v xdg-open >/dev/null 2>&1; then
      xdg-open "$APP_URL" >/dev/null 2>&1 || true
    elif command -v open >/dev/null 2>&1; then
      open "$APP_URL" >/dev/null 2>&1 || true
    fi
  ) &
  OPENER_PID=$!
  trap 'kill "$OPENER_PID" 2>/dev/null || true' EXIT
fi

echo "[run.sh] Starting Audio8-TTS Web UI at ${APP_URL}..."
exec "$VENV_PY" -m uvicorn --app-dir "$ROOT_DIR" server:app --host "$HOST" --port "$PORT"
