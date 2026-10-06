#!/usr/bin/env bash
# Provision or repair the isolated remote inference environment. Run explicitly;
# start_remote_server.sh never invokes pip.
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python3}"
VENV="$ROOT/.modalityconverter-server"
# auto follows nvidia-smi; set SERVER_GPU=1 to force CUDA provisioning when
# the container hides nvidia-smi, or SERVER_GPU=0 to force CPU packages.
SERVER_GPU="${SERVER_GPU:-auto}"

if [[ ! -f "$ROOT/ModalityConverter/ModalityConverterLib/remote_server.py" ]]; then
  echo "Run this script from a SlicerModalityConverter repository checkout." >&2
  exit 1
fi
command -v "$PYTHON" >/dev/null || { echo "Install Python 3.10+ first." >&2; exit 1; }
if [[ ! -x "$VENV/bin/python" ]]; then "$PYTHON" -m venv "$VENV"; fi
VPY="$VENV/bin/python"
"$VPY" -m pip install --upgrade pip

# Install the general stack first: MONAI may otherwise upgrade Torch after the
# CUDA-specific Torch/ONNX Runtime builds have been selected below.
"$VPY" -m pip install "fastapi" "uvicorn[standard]" "python-multipart" \
  psutil requests numpy "monai[itk]" onnx nibabel SimpleITK scipy

# Install the intended Torch build first so MONAI's dependencies keep it rather
# than installing a generic Torch wheel and replacing it afterward.
if [[ "$SERVER_GPU" == "auto" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then SERVER_GPU=1; else SERVER_GPU=0; fi
fi
if [[ "$SERVER_GPU" != "0" && "$SERVER_GPU" != "1" ]]; then
  echo "SERVER_GPU must be auto, 1, or 0 (got: $SERVER_GPU)." >&2
  exit 2
fi
echo "Provisioning server environment with SERVER_GPU=$SERVER_GPU using $VPY"

if [[ "$SERVER_GPU" == "1" ]]; then
  if ! "$VPY" -c 'import torch; assert torch.__version__.split("+")[0] == "2.5.1" and torch.version.cuda == "12.1"' >/dev/null 2>&1; then
    echo "Installing PyTorch 2.5.1 with CUDA 12.1..."
    "$VPY" -m pip install --index-url https://download.pytorch.org/whl/cu121 "torch==2.5.1"
  else
    echo "Reusing installed PyTorch 2.5.1 CUDA 12.1."
  fi
  if ! "$VPY" -c 'import onnxruntime as ort; assert ort.__version__ == "1.20.1" and "CUDAExecutionProvider" in ort.get_available_providers()' >/dev/null 2>&1; then
    echo "Installing ONNX Runtime GPU..."
    "$VPY" -m pip uninstall -y onnxruntime onnxruntime-gpu >/dev/null 2>&1 || true
    "$VPY" -m pip install "onnxruntime-gpu"
  else
    echo "Reusing installed ONNX Runtime GPU"
  fi
else
  if ! "$VPY" -c 'import torch; assert torch.__version__.split("+")[0] == "2.5.1" and torch.version.cuda is None' >/dev/null 2>&1; then
    echo "Installing CPU PyTorch 2.5.1..."
    "$VPY" -m pip install --index-url https://download.pytorch.org/whl/cpu "torch==2.5.1"
  else
    echo "Reusing installed CPU PyTorch 2.5.1."
  fi
  if ! "$VPY" -c 'import onnxruntime' >/dev/null 2>&1; then
    "$VPY" -m pip install "onnxruntime"
  else
    echo "Reusing installed ONNX Runtime."
  fi
fi


echo "Remote inference environment is ready at $VENV"

if [[ "$SERVER_GPU" == "1" ]]; then
  if "$VPY" -c 'import torch, onnxruntime as ort; getattr(ort, "preload_dlls", lambda: None)(); assert torch.cuda.is_available() and "CUDAExecutionProvider" in ort.get_available_providers()' >/dev/null 2>&1; then
    echo "Verified remote GPU inference (PyTorch CUDA + ONNX Runtime CUDA)."
  else
    echo "ERROR: GPU provisioning was requested, but remote GPU inference is unavailable in this environment." >&2
    "$VPY" -c 'import sys, torch, onnxruntime as ort; print("Python:", sys.executable); print("Torch:", torch.__version__, "CUDA:", torch.version.cuda, "available:", torch.cuda.is_available()); print("ONNX Runtime:", ort.__version__, "path:", ort.__file__, "providers:", ort.get_available_providers())' >&2 || true
    exit 1
  fi
fi
