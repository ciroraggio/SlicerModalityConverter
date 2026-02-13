import slicer
from slicer import vtkMRMLScalarVolumeNode
from SyntheticImageQualityAssuranceLib.ModelBase import BaseModel, register_model
from SyntheticImageQualityAssuranceLib.Utils.modelLoadUtils import import_onnx_model
from SyntheticImageQualityAssuranceLib.UI.utils import PRINT_MODULE_SUFFIX
import numpy as np

@register_model("mae_slice_sct_qa")
class MAESlicesCTQAModel(BaseModel):
    """Model class for sCT 2D MAE prediction"""

    def __init__(self, modelKey: str, device: str = "cpu"):
        """
        Initialize the model class

        Parameters:
            modelKey (str): Key/identifier for the model
            device (str): Device to run inference on ('cpu' or 'cuda')
        """
        super().__init__(modelKey, device)

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

    def runInference(
        self,
        inputVolume: vtkMRMLScalarVolumeNode,
        outputVolume: vtkMRMLScalarVolumeNode,
        inputMask: vtkMRMLScalarVolumeNode = None,
        showAllFiles: bool = False,
    ):

        if not isinstance(inputVolume, slicer.vtkMRMLScalarVolumeNode):
            raise TypeError("Input must be a vtkMRMLScalarVolumeNode")

        if self.model is None:
            raise RuntimeError("Model not loaded. Call loadModel() first.")

        print(f"{PRINT_MODULE_SUFFIX} Running inference...")
        slicer.app.processEvents()

        # Get sct e mask array
        sct = slicer.util.arrayFromVolume(inputVolume)
        mask = slicer.util.arrayFromVolume(inputMask)
        mae_prediction = np.zeros_like(sct)

        n_slide = int(sct.shape[1])

        if n_slide == 0:
            raise ValueError("Input volume has 0 slices. Cannot run inference.")

        # Predict MAE for each axial slice
        for i in range(n_slide):
            if np.sum(mask[:,i,:]) > 0:
                # The model accepts 2D images 256x256
                reshaped_slice = np.resize(sct[:,i,:], (256,256))
                reshaped_slice = np.reshape(reshaped_slice, (1,1,256,256)).astype(np.float16)

                # Run inference
                inp_name = self.model.get_inputs()[0].name
                mae_slice = int(self.model.run(None, {inp_name: reshaped_slice})[0][0][0])

                # MAE clips at 135
                if mae_slice > 135:
                    mae_slice = 135

                mae_prediction[:,i,:] = mae_slice

        # Mask mae prediction map
        mae_prediction[mask==0]=0

        # Update output volume
        slicer.util.updateVolumeFromArray(outputVolume, mae_prediction)
        outputVolume.CopyOrientation(inputVolume)
        displayNode = outputVolume.GetDisplayNode()
        if not displayNode:
            outputVolume.CreateDefaultDisplayNodes()
            displayNode = outputVolume.GetDisplayNode()
        displayNode.SetAndObserveColorNodeID('vtkMRMLColorTableNodeRainbow')
        slicer.util.setSliceViewerLayers(background=outputVolume, foreground=None)
        slicer.util.resetSliceViews()

        print(f"{PRINT_MODULE_SUFFIX} Inference completed.")

        return mae_prediction
