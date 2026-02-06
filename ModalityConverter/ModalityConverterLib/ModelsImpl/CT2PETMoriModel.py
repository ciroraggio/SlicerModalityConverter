import os
import slicer
from slicer import vtkMRMLScalarVolumeNode
from ModalityConverterLib.ModelBase import BaseModel, register_model
from ModalityConverterLib.UI.utils import PRINT_MODULE_SUFFIX
import numpy as np
import torch

@register_model("ct2pet_mori")
class CT2PETMoriModel(BaseModel):
    """Model class for CT to PET inference"""
    
    def __init__(self, modelKey: str, device: str = "cpu"):
        """
        Initialize the model class
        
        Parameters:
            modelKey (str): Key/identifier for the model
            device (str): Device to run inference on ('cpu' or 'cuda')
        """
        super().__init__(modelKey, device)
        self.use_onnx = False
        self.onnx_session = None
        self.model = None
    
    def _loadModelFromPath(self, modelPath):
        self.modelPath = modelPath
        if not os.path.exists(modelPath):
            raise FileNotFoundError(f"Model file not found: {modelPath}")

        # Check if it's an ONNX model
        if modelPath.endswith('.onnx'):
            try:
                import onnxruntime as ort
                if self.device != "cpu" and ort.get_device() != "GPU":
                    slicer.util.errorDisplay("A GPU device is selected, but ONNX Runtime is not built with GPU support. Installation of ONNX Runtime GPU is required. Slicer will restart after installation.")
                    slicer.util.pip_install("onnxruntime-gpu")
                    slicer.app.restart()
                
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if self.device.startswith('cuda') else ['CPUExecutionProvider']
                onnx_session = ort.InferenceSession(modelPath, providers=providers)
                self.use_onnx = True
                self.onnx_session = onnx_session  # Keep reference for compatibility
                print(f"{PRINT_MODULE_SUFFIX} Loaded ONNX model: {modelPath}")
                print(f"{PRINT_MODULE_SUFFIX} Using provider: {onnx_session.get_providers()[0]}")
                return onnx_session  # This gets stored in self.model by BaseModel
            except ImportError:
                raise ImportError("onnxruntime is required for ONNX models. Install with: pip install onnxruntime")
        else:
            try:
                self.model = torch.load(modelPath, map_location=self.device)
                if isinstance(self.model, torch.nn.Module):
                    self.model = self.model.to(self.device)
                    self.model.eval()
                else:
                    raise ValueError("Model file appears to be a state dict. Model architecture loading not implemented.")
                self.use_onnx = False
                print(f"{PRINT_MODULE_SUFFIX} Loaded PyTorch model: {modelPath}")
                return self.model
            except Exception as e:
                raise RuntimeError(f"Failed to load PyTorch model: {e}")
    
    def _preprocessCT(self, im, minn=-900.0, maxx=200.0):
        """Preprocess CT image"""
        img = np.clip(np.array(im), minn, maxx)
        return (img - minn) / (maxx - minn)
    
    def _edge_zero(self, img):
        """Zero out edges of the image"""
        img[:, [0, -1], :] = 0
        img[:, :, [0, -1]] = 0
        return img
    
    def _post_gamma_PET(self, img, gamma=1/2, maxx=7.0):
        """Post-process PET image with gamma correction"""
        return np.power(np.clip(img, 0.0, 1.0), 1/gamma) * maxx

    def runInference(
        self,
        inputVolume: vtkMRMLScalarVolumeNode,
        outputVolume: vtkMRMLScalarVolumeNode,
        inputMask: vtkMRMLScalarVolumeNode = None,
        showAllFiles: bool = True,
    ):
        if not isinstance(inputVolume, slicer.vtkMRMLScalarVolumeNode):
            raise TypeError("Input must be a vtkMRMLScalarVolumeNode")

        if self.model is None:
            raise RuntimeError("Model not loaded. Call loadModel() first.")

        print(f"{PRINT_MODULE_SUFFIX} Running inference...")
        slicer.app.processEvents()

        # Get input volume as numpy array: im shape [Z, H, W]
        im = slicer.util.arrayFromVolume(inputVolume)
        n_slide = int(im.shape[0])

        if n_slide == 0:
            raise ValueError("Input volume has 0 slices. Cannot run inference.")

        # Pad input to enable inference for first/last 3 slices by repeating edges
        first = im[0:1, :, :]   # [1, H, W]
        last  = im[-1:, :, :]  # [1, H, W]
        im_pad = np.concatenate(
            [np.repeat(first, 3, axis=0), im, np.repeat(last, 3, axis=0)],
            axis=0
        )  # [Z+6, H, W] -> guarantees k:k+7 always valid for k in [0..Z-1]

        # Initialize output PET volume (same Z as input)
        PET = np.zeros((im.shape[1], im.shape[2], n_slide), dtype=np.float32)

        # Progress milestones
        total_slices = n_slide
        milestones = [1, 10] + [i for i in range(100, total_slices + 1, 100)]
        if showAllFiles:
            print(f"{PRINT_MODULE_SUFFIX} Progress will be shown at slices: {', '.join(map(str, milestones))}")

        # Process ALL slices (including first/last 3 via padding)
        for count, k in enumerate(range(0, n_slide)):
            window7 = im_pad[k:k+7, :, :]  # [7, H, W] always

            CT_processed = self._edge_zero(self._preprocessCT(window7))
            CT_tensor = torch.from_numpy(CT_processed).float().unsqueeze(0)  # [1, 7, H, W]

            with torch.no_grad():
                if self.use_onnx:
                    inp_name = self.model.get_inputs()[0].name
                    pred_array = self.model.run(None, {inp_name: CT_tensor.numpy().astype(np.float32)})[0]
                    pred_tensor = torch.from_numpy(pred_array)
                else:
                    pred_tensor = self.model(CT_tensor.to(self.device)).cpu()

            # Expecting output like [1, C, H, W]; take channel 1 (middle)
            if pred_tensor.ndim != 4 or pred_tensor.shape[1] < 2:
                raise RuntimeError(f"Unexpected model output shape: {tuple(pred_tensor.shape)} (need [1, >=2, H, W])")

            pred_2d = pred_tensor[0, 1, :, :].numpy()
            PET[:, :, k] = pred_2d

            if showAllFiles and (count + 1) in milestones:
                print(f"{PRINT_MODULE_SUFFIX} Processed {count + 1} slices...")
                slicer.app.processEvents()
            slicer.app.processEvents()

        # Post-process PET image
        PET = self._post_gamma_PET(PET, maxx=7.0)
        PET = np.transpose(PET, (2, 0, 1))  # -> [Z, H, W]

        # Update output volume
        slicer.util.updateVolumeFromArray(outputVolume, PET)
        outputVolume.CopyOrientation(inputVolume)
        slicer.util.resetSliceViews()

        if showAllFiles:
            print(f"{PRINT_MODULE_SUFFIX} Inference completed.")

        return PET
