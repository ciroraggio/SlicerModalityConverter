import os
import slicer
from ModalityConverterLib.UI.utils import PRINT_MODULE_SUFFIX

def import_onnx_model(modelPath, device): 
    import onnxruntime as ort
    
    """Load the model using ONNX Runtime."""
    if not os.path.exists(modelPath):
        raise FileNotFoundError(f"Model file not found at {modelPath}")
    
    if device != "cpu" and ort.get_device() != "GPU":
    # if the user wants to use the GPU but onnxruntime is somehow not built with GPU support
        slicer.util.errorDisplay("A GPU device is selected, but ONNX Runtime is not built with GPU support. Installation of ONNX Runtime GPU is required. Slicer will restart after installation.")
        slicer.util.pip_install("onnxruntime-gpu")
        slicer.app.restart()
        
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device.startswith("cuda") else ["CPUExecutionProvider"]
    
    print(f"{PRINT_MODULE_SUFFIX} Loading ONNX model with providers: {providers}")
    
    model = ort.InferenceSession(modelPath, providers=providers)
    return model