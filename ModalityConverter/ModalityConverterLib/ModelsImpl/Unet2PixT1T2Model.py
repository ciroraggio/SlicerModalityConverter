import os
import slicer
from ModalityConverterLib.ModelBase import BaseModel, register_model
from ModalityConverterLib.UI.utils import PRINT_MODULE_SUFFIX

@register_model("unet2pix_t1_t2")
class Unet2PixT1T2Model(BaseModel):
    def __init__(self, modelKey: str, device: str = "cpu"):
        super().__init__(modelKey, device)
        self.ort_session = None

    def _loadModelFromPath(self, modelPath):
        try:
            import onnxruntime as ort

            """Load the model using ONNX Runtime."""
            if not os.path.exists(modelPath):
                raise FileNotFoundError(f"Model file not found at {modelPath}")
            
            if self.device != "cpu" and ort.get_device() != "GPU":
            # if the user wants to use the GPU but onnxruntime is somehow not built with GPU support
                slicer.util.errorDisplay("A GPU device is selected, but ONNX Runtime is not built with GPU support. Installation of ONNX Runtime GPU is required. Slicer will restart after installation.")
                slicer.util.pip_install("onnxruntime-gpu")
                slicer.app.restart()
                
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if self.device.startswith("cuda") else ["CPUExecutionProvider"]
            
            print(f"{PRINT_MODULE_SUFFIX} Loading ONNX model with providers: {providers}")
            
            self.ort_session = ort.InferenceSession(modelPath, providers=providers)
            return self.ort_session
        
        except Exception as e:
            raise RuntimeError(f"Failed to load ONNX model from {modelPath}: {str(e)}")

    def _percnorm(self, arr, lperc=5, uperc=99.5):
        from numpy import percentile, clip
        lowerbound = percentile(arr, lperc)
        upperbound = percentile(arr, uperc)
        clip(arr, lowerbound, upperbound, out=arr)
        return arr

    def _normalize(self, img):
        img_min = img.min()
        img_max = img.max()
        if (img_max - img_min) > 1e-8:
            return (img - img_min) / (img_max - img_min)
        return img - img_min

    def _resize_slice(self, slice_2d, target_h, target_w):
        from scipy.ndimage import zoom

        """
        Resizes slice to target dimensions using interpolation.
        Required because the model expects fixed input size (224x192).
        """
        h, w = slice_2d.shape
        zoom_factors = (target_h / h, target_w / w)
        # Order 1 = bilinear interpolation
        return zoom(slice_2d, zoom_factors, order=1)

    def runInference(self, inputVolume, outputVolume, inputMask=None, showAllFiles=True):
        from numpy import float32, newaxis, zeros_like, zeros

        if not isinstance(inputVolume, slicer.vtkMRMLScalarVolumeNode):
            raise TypeError("Input must be a vtkMRMLScalarVolumeNode")
        
        print(f"{PRINT_MODULE_SUFFIX} Preprocessing input volume...")

        inputNp = slicer.util.arrayFromVolume(inputVolume)
        d_orig, h_orig, w_orig = inputNp.shape
        outputNp = zeros_like(inputNp, dtype=float32)

        for z in range(d_orig):
            # This model was trained on axial slice, thus extract axial slice
            slice_orig = inputNp[z, :, :].copy().astype(float32)

            # Preprocessing
            slice_proc = self._percnorm(slice_orig)
            slice_proc = self._normalize(slice_proc)
            target_h, target_w = 224, 192
            slice_input_model = self._resize_slice(slice_proc, target_h, target_w)

            # transform slice in batch shape
            model_input_tensor = slice_input_model[newaxis, newaxis, :, :]

            input_name = self.ort_session.get_inputs()[0].name
            timestep_name = self.ort_session.get_inputs()[1].name
            timestep_input = zeros((1,), dtype=float32)

            ort_inputs = {input_name: model_input_tensor, timestep_name: timestep_input}
            predicted_slice = self.ort_session.run(None, ort_inputs)[0].squeeze()  # [224, 192]
 
            # resize to original dimension
            generated_slice_final = self._resize_slice(predicted_slice, h_orig, w_orig)

            generated_slice_final = (generated_slice_final - generated_slice_final.min())
            generated_slice_final /= (generated_slice_final.max() + 1e-8)
            generated_slice_final *= inputNp.max()

            outputNp[z, :, :] = generated_slice_final

        slicer.util.updateVolumeFromArray(outputVolume, outputNp)
        outputVolume.CopyOrientation(inputVolume)
        slicer.util.resetSliceViews()
        return outputNp
    