import os
from ModalityConverterLib.UI.utils import PRINT_MODULE_SUFFIX

def import_onnx_model(modelPath, device): 
    try:
        import onnxruntime as ort
    except Exception as e:
        message = str(e)
        if "libcudart" in message:
            raise RuntimeError(
                "The installed ONNX Runtime GPU package cannot find its CUDA runtime ({}). "
                "Install or repair the CPU ONNX Runtime package, or install a GPU runtime "
                "that matches this ONNX Runtime build.".format(message)
            ) from e
        raise RuntimeError("ONNX Runtime could not be imported: {}".format(message)) from e
    
    """Load the model using ONNX Runtime."""
    if not os.path.exists(modelPath):
        raise FileNotFoundError(f"Model file not found at {modelPath}")
    
    available_providers = ort.get_available_providers()
    if device == "coreml":
        if "CoreMLExecutionProvider" not in available_providers:
            raise RuntimeError("CoreMLExecutionProvider is unavailable in this ONNX Runtime environment")
        providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
    elif device.startswith("cuda"):
        if "CUDAExecutionProvider" not in available_providers:
            raise RuntimeError("CUDAExecutionProvider is unavailable in this ONNX Runtime environment")
        try:
            device_id = int(device.split(":", 1)[1])
        except (IndexError, ValueError):
            device_id = 0
        providers = [("CUDAExecutionProvider", {"device_id": device_id}), "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]

    print(f"{PRINT_MODULE_SUFFIX} Loading ONNX model with providers: {providers}")    
    model = ort.InferenceSession(modelPath, providers=providers)
    return model