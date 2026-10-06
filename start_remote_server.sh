#!/usr/bin/env bash
# Start the remote inference API without installing or updating dependencies.
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PORT="${PORT:-8765}"
# Loopback is the safe default for plain HTTP. Set BIND_HOST explicitly to make
# the service reachable from another computer (for example BIND_HOST=0.0.0.0).
BIND_HOST="${BIND_HOST:-127.0.0.1}"
VENV="$ROOT/.modalityconverter-server"
VPY="$VENV/bin/python"

if [[ ! -f "$ROOT/ModalityConverter/ModalityConverterLib/remote_server.py" ]]; then
  echo "Run this script from a SlicerModalityConverter repository checkout." >&2
  exit 1
fi
if [[ ! -x "$VPY" ]]; then
  echo "Remote server environment is not installed. Run ./install_remote_server.sh first." >&2
  exit 1
fi
if ! "$VPY" -c 'import fastapi, onnxruntime, torch' >/dev/null 2>&1; then
  echo "Remote server dependencies are incomplete. Run ./install_remote_server.sh to repair the environment." >&2
  exit 1
fi

if [[ "$BIND_HOST" == "127.0.0.1" || "$BIND_HOST" == "localhost" ]]; then
  CLIENT_ADDRESS="$BIND_HOST"
else
  CLIENT_ADDRESS="$(hostname -I 2>/dev/null | awk '{print $1}' || true)"
  if [[ -z "$CLIENT_ADDRESS" ]] && command -v ipconfig >/dev/null 2>&1; then
    CLIENT_ADDRESS="$(ipconfig getifaddr en0 2>/dev/null || true)"
  fi
  [[ -n "$CLIENT_ADDRESS" ]] || CLIENT_ADDRESS="$BIND_HOST"
fi
echo
echo "In Slicer > ModalityConverter > Advanced, use address http://${CLIENT_ADDRESS} and port ${PORT}."
if [[ "$BIND_HOST" != "127.0.0.1" && "$BIND_HOST" != "localhost" ]]; then
  echo "Plain HTTP is enabled. Restrict port ${PORT} to trusted clients with the server firewall/VPN."
else
  echo "Loopback binding only: set BIND_HOST to a server interface address to accept remote clients."
fi
echo "The bearer token is provided below. Keep it private. Stop this server with Ctrl+C."
exec "$VPY" "$ROOT/ModalityConverter/ModalityConverterLib/remote_server.py" \
  --host "$BIND_HOST" --port "$PORT"
