
import slicer
from slicer import vtkMRMLScalarVolumeNode
from ModalityConverterLib.ModelBase import BaseModel
from ModalityConverterLib.ModelBase import register_model
from ModalityConverterLib.UI.utils import PRINT_MODULE_SUFFIX
from ModalityConverterLib.Utils.modelLoadUtils import import_onnx_model
@register_model("fedsynthct_pix2pix_cbct_headneck")
class FedSynthCBCTPix2PixModel(BaseModel):
    def __init__(self, modelKey: str, device: str = "cpu"):
        super().__init__(modelKey, device)
        self.hu_min_clipping = -800
        self.hu_max_clipping = 2000
        
    def _loadModelFromPath(self, modelPath):
        try:
            return import_onnx_model(modelPath=modelPath, device=self.device)
            
        except Exception as e:
            raise RuntimeError(f"Failed to load ONNX model from {modelPath}: {str(e)}")
    
    def getPreprocessingTransform(self, maxSize, backgroundValue, isCbct=False):
        from monai.transforms import EnsureChannelFirst, Compose, Orientation, CenterSpatialCrop, Spacing
        from ModalityConverterLib.Utils.customTransformsUtils import PadToCube, CustomResize, LUTClipping        
        
        steps = [
                    EnsureChannelFirst(channel_dim="no_channel"),
                    Orientation(as_closest_canonical=True),
                    CenterSpatialCrop((328, 256, 328)),
                    CustomResize(maxSize=maxSize),
                    Spacing(pixdim=(1,1,1), mode="bilinear"),
                    PadToCube(backgroundValue=backgroundValue, size=maxSize),   
                ]
            
        
        if isCbct:
            steps += [LUTClipping(hu_min=self.hu_min_clipping, hu_max=self.hu_max_clipping)]

        transform = Compose(steps)

        return transform

        
    def invertNormalization(self, image_norm, hu_min=-710, hu_max=1640):
        return image_norm * (hu_max - hu_min) + hu_min

        
    def preprocess(self, inputVolume, inputMask=None, showAllFiles=True):
        """Preprocess the input volume for inference."""
        from torch import from_numpy
        if not isinstance(inputVolume, slicer.vtkMRMLScalarVolumeNode):
            raise TypeError("Input must be a vtkMRMLScalarVolumeNode")
        
        print(f"{PRINT_MODULE_SUFFIX} Preprocessing input volume...")

        if not isinstance(inputMask, slicer.vtkMRMLScalarVolumeNode):
                raise TypeError("Input mask must be a vtkMRMLScalarVolumeNode")
            
        inputNp = slicer.util.arrayFromVolume(inputVolume)
        maskNp = slicer.util.arrayFromVolume(inputMask)

        if maskNp.min() < 0 or maskNp.max() > 1:
            raise ValueError("Input mask must be binary.")
         
        input_tensor = from_numpy(inputNp).float()  # (1, 1, D, H, W)
        mask_tensor = from_numpy(maskNp)  # (1, 1, D, H, W)
        
        print(f"{PRINT_MODULE_SUFFIX} Applying transforms...")
        preprocessedInputVolume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", "PreprocessedInputVolume")
        preprocessedMaskVolume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", "PreprocessedMaskVolume")
        
        cbctTransform = self.getPreprocessingTransform(maxSize=256, backgroundValue=-1000, isCbct=True)
        maskTransform = self.getPreprocessingTransform(maxSize=256, backgroundValue=0)
        preprocessedInput = cbctTransform(input_tensor)
        preprocessedMask = maskTransform(mask_tensor)
                    
        if showAllFiles:
            slicer.util.updateVolumeFromArray(preprocessedInputVolume, preprocessedInput.squeeze().cpu().numpy())
            displayNode = preprocessedInputVolume.GetDisplayNode()
            #if displayNode:
            #    displayNode.SetWindow(1.0)
            #    displayNode.SetLevel(0)
            
            slicer.util.updateVolumeFromArray(preprocessedMaskVolume, preprocessedMask.squeeze().cpu().numpy())    
            
            preprocessedInputVolume.CopyOrientation(inputVolume)
            preprocessedMaskVolume.CopyOrientation(inputVolume)
        else:
            displayNode = preprocessedMaskVolume.GetDisplayNode()
            if displayNode:
                displayNode.RemoveAllViewIDs()
                displayNode.SetVisibility(False)

            slicer.mrmlScene.RemoveNode(preprocessedMaskVolume)
            
        print(f"{PRINT_MODULE_SUFFIX} Preprocessing completed.")
        slicer.util.resetSliceViews()
        slicer.app.processEvents()
        
        return {"input": preprocessedInput, "mask": preprocessedMask}
    
    def runInference(self, inputVolume: vtkMRMLScalarVolumeNode, outputVolume: vtkMRMLScalarVolumeNode, inputMask: vtkMRMLScalarVolumeNode = None, showAllFiles: bool = True):
        from torch import zeros, from_numpy, tensor, median, stack
        preprocessedData = self.preprocess(inputVolume, inputMask, showAllFiles)

        print(f"{PRINT_MODULE_SUFFIX} Running inference...")
        slicer.app.processEvents()
        
        preprocessedInput = preprocessedData["input"].cpu().numpy()
        preprocessedMask = preprocessedData["mask"].cpu().numpy()

        # Prepare containers
        sCT = {
            view: zeros(preprocessedInput.shape, device=self.device)
            for view in ["first_plane", "second_plane", "third_plane"]
        }
        cbctVol = zeros(
            (1, 1, preprocessedInput.shape[2], preprocessedInput.shape[3]), device=self.device
        )

        for view in sCT:
            for sliceIndex in range(preprocessedInput.shape[1]):
                if view == "first_plane":
                    inputSlice = preprocessedInput[0, sliceIndex, :, :]
                    maskSlice = preprocessedMask[0, sliceIndex, :, :]
                elif view == "second_plane":
                    inputSlice = preprocessedInput[0, :, sliceIndex, :]
                    maskSlice = preprocessedMask[0, :, sliceIndex, :]
                else: # third_plane
                    inputSlice = preprocessedInput[0, :, :, sliceIndex]
                    maskSlice = preprocessedMask[0, :, :, sliceIndex]

                # Convert NumPy slice to torch and copy into cbctVol
                cbctVol[0, 0, :, :] = from_numpy(inputSlice).to(self.device).type(cbctVol.dtype)

                if 1 in maskSlice:
                    # Prepare input for ONNX Runtime
                    # Add batch and channel dimensions: (1, 1, H, W)
                    ortInputs = {
                        self.model.get_inputs()[0].name: cbctVol[0, 0].cpu().numpy()[None, None, :, :]
                    }
                    ortOuts = self.model.run(None, ortInputs)

                    sCT_slice = tensor(ortOuts[0]).to(self.device).type(sCT[view].dtype)

                    # Assign slice back to correct axis
                    if view == "first_plane":
                        sCT[view][:, sliceIndex, :, :] = sCT_slice
                    elif view == "second_plane":
                        sCT[view][:, :, sliceIndex, :] = sCT_slice
                    elif view == "third_plane":
                        sCT[view][:, :, :, sliceIndex] = sCT_slice

        # Median voting across views
        votedSCT, _ = median(stack([sCT["first_plane"], sCT["second_plane"], sCT["third_plane"]], dim=0), dim=0)   
        
        votedSCT[from_numpy(preprocessedMask == 0).to(self.device)] = -1000

        slicer.util.updateVolumeFromArray(outputVolume, votedSCT.squeeze().cpu().numpy())
        outputVolume.CopyOrientation(inputVolume)
        
        if showAllFiles:
            slicer.util.setSliceViewerLayers(background=outputVolume, foreground=slicer.util.getNode("PreprocessedInputVolume"))
        else:
            slicer.util.setSliceViewerLayers(background=outputVolume, foreground=None)
            
        slicer.util.resetSliceViews()
        
        print(f"{PRINT_MODULE_SUFFIX} Inference completed, sCT volume is ready.")

        return votedSCT
        



