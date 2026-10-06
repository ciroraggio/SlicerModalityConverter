"""Small authenticated HTTP service for running this extension's inference worker.

Use the repository-root ``start_remote_server.sh`` to create an isolated
runtime and launch this API. Bind to 127.0.0.1 by default; set BIND_HOST
explicitly to accept a remote client over a trusted network/VPN.
"""
import argparse
import hmac
import os
import secrets
import shutil
import subprocess
import sys
import tempfile
import threading
import uuid
import zipfile

from fastapi import FastAPI, Header, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
WORKER = os.path.join(os.path.dirname(__file__), "inference_worker.py")
JOBS = {}
LOCK = threading.Lock()
TOKEN = os.environ.get("MODALITY_CONVERTER_TOKEN", "")
GPU_CAPABILITIES = None
app = FastAPI(title="ModalityConverter inference server", docs_url=None, redoc_url=None)


def authorize(value):
    if not TOKEN or not hmac.compare_digest(value or "", TOKEN):
        raise HTTPException(status_code=401, detail="Invalid bearer token")


@app.get("/api/v1/health")
def health(authorization: str = Header(default="")):
    authorize(authorization[len("Bearer "):] if authorization.startswith("Bearer ") else authorization)
    return {"status": "ok", "models": _models()}


def _models():
    import json
    with open(os.path.join(ROOT, "Resources", "Models", "metadata.json"), encoding="utf-8") as f:
        return json.load(f)


def _gpu_capabilities():
    """Only advertise CUDA when both Torch tensors and ONNX inference can use it."""
    global GPU_CAPABILITIES
    if GPU_CAPABILITIES is not None:
        return GPU_CAPABILITIES
    try:
        # PyTorch's CUDA wheel bundles CUDA/cuDNN libraries in this environment.
        # Load them before ORT enumerates providers, or CUDA can appear missing
        # at server startup even though the inference worker can use it.
        import torch  # noqa: F401
        import onnxruntime
        preload = getattr(onnxruntime, "preload_dlls", None)
        if preload:
            preload()
        providers = onnxruntime.get_available_providers()
        if "CUDAExecutionProvider" not in providers:
            GPU_CAPABILITIES = {"cuda_available": False,
                                "cuda_reason": "ONNX Runtime {} at {} has providers {}".format(
                                    getattr(onnxruntime, "__version__", "unknown"),
                                    getattr(onnxruntime, "__file__", "unknown"), providers)}
            return GPU_CAPABILITIES
    except Exception as exc:
        GPU_CAPABILITIES = {"cuda_available": False,
                            "cuda_reason": "ONNX Runtime could not be checked: {}".format(exc)}
        return GPU_CAPABILITIES
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.init()
            GPU_CAPABILITIES = {"cuda_available": True, "cuda_reason": ""}
        else:
            try:
                torch.cuda.init()
            except Exception as exc:
                reason = str(exc)
            else:
                reason = "PyTorch reports that CUDA is unavailable"
            GPU_CAPABILITIES = {"cuda_available": False, "cuda_reason": reason}
    except Exception as exc:
        GPU_CAPABILITIES = {"cuda_available": False,
                            "cuda_reason": "PyTorch CUDA could not be initialized: {}".format(exc)}
    return GPU_CAPABILITIES


@app.get("/api/v1/resources")
def resources(authorization: str = Header(default="")):
    authorize(authorization[len("Bearer "):] if authorization.startswith("Bearer ") else authorization)
    import psutil
    gpuCapabilities = _gpu_capabilities()
    data = {"cpu_percent": psutil.cpu_percent(), "ram_used": psutil.virtual_memory().used,
            "ram_total": psutil.virtual_memory().total, "gpus": [], **gpuCapabilities}
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=index,name,utilization.gpu,memory.used,memory.total",
                                 "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=2)
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                i, name, util, used, total = [x.strip() for x in line.split(",", 4)]
                data["gpus"].append({"index": int(i), "name": name, "utilization": float(util),
                                     "memory_used_mb": float(used), "memory_total_mb": float(total)})
    except Exception:
        pass
    return data


@app.post("/api/v1/inference")
async def inference(model: str = Form(...), module: str = Form(...), device: str = Form(...),
                    show_previews: bool = Form(True), input_preprocessed: bool = Form(False),
                    input_file: UploadFile = File(...),
                    mask_file: UploadFile = File(None), authorization: str = Header(default="")):
    authorize(authorization[len("Bearer "):] if authorization.startswith("Bearer ") else authorization)
    metadata = _models()
    if model not in metadata or metadata[model].get("module_name") != module:
        raise HTTPException(400, "Unknown model")
    if device != "cpu" and (not device.startswith("cuda:") or not device[5:].isdigit()):
        raise HTTPException(400, "Unsupported device")
    job_id = str(uuid.uuid4())
    directory = tempfile.mkdtemp(prefix="modality-converter-")
    input_path, output_path = os.path.join(directory, "input.npy"), os.path.join(directory, "output.npy")
    try:
        for upload, path in ((input_file, input_path), (mask_file, os.path.join(directory, "mask.npy"))):
            if upload:
                size = 0
                with open(path, "wb") as dst:
                    while True:
                        chunk = await upload.read(1024 * 1024)
                        if not chunk: break
                        size += len(chunk)
                        if size > 2 * 1024**3: raise HTTPException(413, "Volume exceeds 2 GiB limit")
                        dst.write(chunk)
        mask_path = os.path.join(directory, "mask.npy")
        command = [sys.executable, WORKER, "--model", model, "--module", module,
                   "--input", input_path, "--output", output_path, "--device", device,
                   "--show-previews", "1" if show_previews else "0", "--extension-root", ROOT]
        if os.path.isfile(mask_path): command += ["--mask", mask_path]
        if input_preprocessed: command += ["--input-preprocessed"]
        env = os.environ.copy()
        env["MODALITY_CONVERTER_MODEL_DIR"] = os.path.join(os.path.expanduser("~"), ".SlicerModalityConverter", "Models")
        process = subprocess.Popen(command, cwd=directory, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, env=env, bufsize=1)
        with LOCK: JOBS[job_id] = {"directory": directory, "process": process, "progress": 0,
                                   "message": "Queued", "finished": False, "error": "", "log_tail": []}
        threading.Thread(target=_watch, args=(job_id,), daemon=True).start()
        return {"job_id": job_id}
    except Exception:
        shutil.rmtree(directory, ignore_errors=True)
        raise


def _watch(job_id):
    job = JOBS[job_id]
    proc = job["process"]
    for line in proc.stdout:
        line = line.rstrip()
        if line:
            job["log_tail"].append(line[-1000:])
            del job["log_tail"][:-40]
        if line.startswith("MODALITY_CONVERTER_PROGRESS:"):
            try:
                _, percent, message = line.split(":", 2)
                job["progress"], job["message"] = int(percent), message
            except ValueError:
                pass
    code = proc.wait()
    job["finished"] = True
    if code:
        details = "\n".join(job["log_tail"][-12:])
        job["error"] = "Inference worker exited with code {}. {}".format(code, details)


@app.get("/api/v1/inference/{job_id}")
def job_status(job_id: str, authorization: str = Header(default="")):
    authorize(authorization[len("Bearer "):] if authorization.startswith("Bearer ") else authorization)
    job = JOBS.get(job_id)
    if not job: raise HTTPException(404, "Unknown job")
    return {k: job[k] for k in ("progress", "message", "finished", "error", "log_tail")}


@app.get("/api/v1/inference/{job_id}/result")
def result(job_id: str, authorization: str = Header(default="")):
    authorize(authorization[len("Bearer "):] if authorization.startswith("Bearer ") else authorization)
    job = JOBS.get(job_id)
    if not job or not job["finished"] or job["error"]: raise HTTPException(409, "Result not ready")
    return FileResponse(os.path.join(job["directory"], "output.npy"), filename="output.npy")


@app.get("/api/v1/inference/{job_id}/previews")
def previews(job_id: str, authorization: str = Header(default="")):
    authorize(authorization[len("Bearer "):] if authorization.startswith("Bearer ") else authorization)
    job = JOBS.get(job_id)
    if not job or not job["finished"] or job["error"]:
        raise HTTPException(409, "Preview volumes are not ready")
    preview_dir = os.path.join(job["directory"], "previews")
    if not os.path.isdir(preview_dir):
        raise HTTPException(404, "No preview volumes were produced")
    archive_path = os.path.join(job["directory"], "previews.zip")
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_STORED) as archive:
        for filename in sorted(os.listdir(preview_dir)):
            path = os.path.join(preview_dir, filename)
            if filename.endswith(".npy") and os.path.isfile(path):
                archive.write(path, arcname=filename)
    return FileResponse(archive_path, filename="previews.zip", media_type="application/zip")


def main():
    global TOKEN
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (default: localhost)")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--token", default="", help="Bearer token; generated if omitted")
    args = parser.parse_args()
    TOKEN = args.token or secrets.token_urlsafe(32)
    print("ModalityConverter bearer token: {}".format(TOKEN), flush=True)
    capabilities = _gpu_capabilities()
    if capabilities["cuda_available"]:
        print("Remote CUDA is ready for PyTorch and ONNX Runtime.", flush=True)
    else:
        print("Remote GPU inference unavailable; CPU remains available: {}".format(
            capabilities["cuda_reason"]), flush=True)
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__": main()
