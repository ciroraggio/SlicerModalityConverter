#!/usr/bin/env bash
# Start the remote inference API without installing or updating dependencies.
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PORT="${PORT:-8765}"
USE_HTTPS="${USE_HTTPS:-0}"
if [[ "$USE_HTTPS" != "0" && "$USE_HTTPS" != "1" ]]; then
  echo "USE_HTTPS must be 0 or 1 (got: $USE_HTTPS)." >&2
  exit 2
fi
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
SERVER_ARGS=(--host "$BIND_HOST" --port "$PORT")
SCHEME="http"
if [[ "$USE_HTTPS" == "1" ]]; then
  command -v openssl >/dev/null || { echo "OpenSSL is required to generate the HTTPS certificate." >&2; exit 1; }
  TLS_DIR="$ROOT/.modalityconverter-server/tls"
  mkdir -p "$TLS_DIR"
  chmod 700 "$TLS_DIR"
  if [[ ! -f "$TLS_DIR/server.crt" || ! -f "$TLS_DIR/server.key" ]]; then
    openssl req -x509 -newkey rsa:3072 -sha256 -days 3650 -nodes \
      -keyout "$TLS_DIR/server.key" -out "$TLS_DIR/server.crt" \
      -subj "/CN=${CLIENT_ADDRESS}" -addext "subjectAltName=IP:${CLIENT_ADDRESS},DNS:${CLIENT_ADDRESS}" >/dev/null 2>&1
    chmod 600 "$TLS_DIR/server.key"
  fi
  SERVER_ARGS+=(--ssl-certfile "$TLS_DIR/server.crt" --ssl-keyfile "$TLS_DIR/server.key")
  SCHEME="https"
  echo "HTTPS certificate (copy this public file to Slicer): $TLS_DIR/server.crt"
  echo "Keep server.key private; do not copy or share it."
fi
echo "In Slicer > ModalityConverter > Advanced, use address ${SCHEME}://${CLIENT_ADDRESS} and port ${PORT}."
if [[ "$BIND_HOST" == "127.0.0.1" || "$BIND_HOST" == "localhost" ]]; then
  echo "Loopback binding only: set BIND_HOST to a server interface address to accept remote clients."
elif [[ "$USE_HTTPS" != "1" ]]; then
  echo "Plain HTTP is enabled. Restrict port ${PORT} to trusted clients with the server firewall/VPN."
fi
echo "The bearer token is provided below. Keep it private. Stop this server with Ctrl+C."
exec "$VPY" "$ROOT/ModalityConverter/ModalityConverterLib/remote_server.py" "${SERVER_ARGS[@]}"
