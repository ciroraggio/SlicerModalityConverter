import numpy as np
import onnxruntime as ort
import scipy.ndimage
from ModalityConverterLib.ModelBase import BaseModel, register_model


@register_model("unet2pix_t1_t2")
class Unet2PixT1T2Model(BaseModel):
    def __init__(self, modelKey: str, device: str = "cpu"):
        super().__init__(modelKey, device)
        self.ort_session = None

    def _loadModelFromPath(self, modelPath):
        providers = ['CPUExecutionProvider']
        if self.device == 'cuda':
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']

        self.ort_session = ort.InferenceSession(modelPath, providers=providers)
        return self.ort_session

    def _percnorm(self, arr, lperc=5, uperc=99.5):
        lowerbound = np.percentile(arr, lperc)
        upperbound = np.percentile(arr, uperc)
        np.clip(arr, lowerbound, upperbound, out=arr)
        return arr

    def _normalize(self, img):
        img_min = img.min()
        img_max = img.max()
        if (img_max - img_min) > 1e-8:
            return (img - img_min) / (img_max - img_min)
        return img - img_min

    def _resize_slice(self, slice_2d, target_h, target_w):
        """
        Resizes slice to target dimensions using interpolation.
        Required because the model expects fixed input size (224x192).
        """
        h, w = slice_2d.shape
        zoom_factors = (target_h / h, target_w / w)
        # Order 1 = bilinear interpolation
        return scipy.ndimage.zoom(slice_2d, zoom_factors, order=1)

    def runInference(self, inputVolume, outputVolume, inputMask=None, showAllFiles=True):
        """
        Logic:
        1. Copy full input (T1) to output.
        2. Extract ONLY the central axial slice.
        3. Resize to 224x192 -> Inference (Generate T2) -> Resize back.
        4. Replace only the central slice in the output volume with the synthetic T2.
        """

        outputVolume[:] = inputVolume[:]

        # Original dimensions: [Depth, Height, Width]
        d_orig, h_orig, w_orig = inputVolume.shape
        center_idx = d_orig // 2

        # Extract slice (copy to ensure float32 processing)
        slice_orig = inputVolume[center_idx, :, :].copy().astype(np.float32)

        # Preprocessing
        slice_proc = self._percnorm(slice_orig)
        slice_proc = self._normalize(slice_proc)

        # Resize for model (Target: H=224, W=192)
        target_h, target_w = 224, 192
        slice_input_model = self._resize_slice(slice_proc, target_h, target_w)

        # Prepare ONNX tensor [1, 1, 224, 192]
        input_tensor = slice_input_model[np.newaxis, np.newaxis, :, :]

        # Get ONNX input names
        input_name = self.ort_session.get_inputs()[0].name
        timestep_name = self.ort_session.get_inputs()[1].name
        timestep_input = np.zeros((1,), dtype=np.float32)

        # Inference
        ort_inputs = {
            input_name: input_tensor,
            timestep_name: timestep_input
        }
        result = self.ort_session.run(None, ort_inputs)

        # Model output is [1, 1, 224, 192] -> squeeze to [224, 192]
        generated_slice_small = result[0].squeeze()

        # Post-processing: Resize back to patient original dimensions
        generated_slice_final = self._resize_slice(generated_slice_small, h_orig, w_orig)

        # Insert generated T2 slice back into volume
        outputVolume[center_idx, :, :] = generated_slice_final
