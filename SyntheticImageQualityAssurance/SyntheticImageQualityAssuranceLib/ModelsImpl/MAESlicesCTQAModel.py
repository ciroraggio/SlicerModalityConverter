import slicer
from slicer import vtkMRMLScalarVolumeNode
from ModalityConverterLib.ModelBase import BaseModel, register_model
from ModalityConverterLib.Utils.modelLoadUtils import import_onnx_model
from ModalityConverterLib.UI.utils import PRINT_MODULE_SUFFIX


@register_model("MAE_slice_sCT_QA")
class MAESlicesCTQAModel(BaseModel):
    """Model class for CT to PET inference"""

    def __init__(self, modelKey: str, device: str = "cpu"):
        """
        Initialize the model class

        Parameters:
            modelKey (str): Key/identifier for the model
            device (str): Device to run inference on ('cpu' or 'cuda')
        """
        super().__init__(modelKey, device)
        print("Hello!")


    def _loadModelFromPath(self, modelPath):
        # Model instance will be stored by the BaseModel class in self.model 
        if (slicer.app.majorVersion, slicer.app.minorVersion) < (5, 10):
            raise RuntimeError(
                "This model is supported only for 3D Slicer >= 5.10. "
                "Please update your 3D Slicer version to use this model."
            )

        try:
            return import_onnx_model(modelPath=modelPath, device=self.device)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load ONNX model from {modelPath}: {e}"
            ) from e

    def _preprocessCT(self, im, minn=-900.0, maxx=200.0):
        from numpy import clip, array
        """Preprocess CT image"""
        img = clip(array(im), minn, maxx)
        return (img - minn) / (maxx - minn)

    def _edge_zero(self, img):
        """Zero out edges of the image"""
        img[:, [0, -1], :] = 0
        img[:, :, [0, -1]] = 0
        return img

    def _post_gamma_PET(self, img, gamma=1/2, maxx=7.0):
        from numpy import power, clip
        """Post-process PET image with gamma correction"""
        return power(clip(img, 0.0, 1.0), 1/gamma) * maxx

    def runInference(
        self,
        inputVolume: vtkMRMLScalarVolumeNode,
        outputVolume: vtkMRMLScalarVolumeNode,
        inputMask: vtkMRMLScalarVolumeNode = None,
        showAllFiles: bool = True,
    ):
        from numpy import concatenate, repeat, zeros, transpose, float32
        print("HELLO!!!!")

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
        first = im[0:1, :, :]  # [1, H, W]
        last  = im[-1:, :, :]  # [1, H, W]
        # [Z+6, H, W] -> guarantees k:k+7 always valid for k in [0..Z-1]
        im_pad = concatenate([repeat(first, 3, axis=0), im, repeat(last, 3, axis=0)], axis=0)

        # Initialize output PET volume (same Z as input)
        PET = zeros((im.shape[1], im.shape[2], n_slide), dtype=float32)

        # Progress milestones
        percent_milestones = [25, 50, 75, 100]
        milestones = [max(1, min(n_slide, round(n_slide * p / 100))) for p in percent_milestones]
        #print(f"{PRINT_MODULE_SUFFIX} Progress will be shown at slices: {', '.join(map(str, milestones))}")

        for count, k in enumerate(range(n_slide)):
            window7 = im_pad[k:k+7, :, :]
            CT_processed = self._edge_zero(self._preprocessCT(window7))
            CT_array = CT_processed.astype(float32)[None, ...]  # [1, 7, H, W]

            inp_name = self.model.get_inputs()[0].name
            pred_array = self.model.run(None, {inp_name: CT_array})[0]

            # Expecting output like [1, C, H, W]; take channel 1 (middle)
            if pred_array.ndim != 4 or pred_array.shape[1] < 2:
                raise RuntimeError(f"Unexpected model output shape: {tuple(pred_array.shape)} (need [1, >=2, H, W])")

            pred_2d = pred_array[0, 1, :, :]
            PET[:, :, k] = pred_2d

            if (count + 1) in milestones:
                pct = round((count + 1) / n_slide * 100)
                print(f"{PRINT_MODULE_SUFFIX} Processed {count + 1}/{n_slide} slices ({pct}%)")
                slicer.app.processEvents()

            slicer.app.processEvents()

        # Post-process PET
        PET = self._post_gamma_PET(PET, maxx=7.0)
        PET = transpose(PET, (2, 0, 1))  # -> [Z, H, W]

        # Update output volume
        slicer.util.updateVolumeFromArray(outputVolume, PET)
        outputVolume.CopyOrientation(inputVolume)
        slicer.util.setSliceViewerLayers(background=outputVolume, foreground=None)
        slicer.util.resetSliceViews()

        print(f"{PRINT_MODULE_SUFFIX} Inference completed.")

        return PET
