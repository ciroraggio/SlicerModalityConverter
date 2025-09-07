
from ModalityConverterLib.ModelBase import BaseModel
import slicer
from slicer import vtkMRMLScalarVolumeNode
import os
from ModalityConverterLib.UI.utils import PRINT_MODULE_SUFFIX

"""Base class for FedSynthCT-Brain models in the ModalityConverter library."""
class FedSynthBrainBaseModel(BaseModel):
    def __init__(self, modelKey: str, device: str = "cpu"):
        super().__init__(modelKey, device)
        
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
            
            model = ort.InferenceSession(modelPath, providers=providers)
            return model
        
        except Exception as e:
            raise RuntimeError(f"Failed to load ONNX model from {modelPath}: {str(e)}")
    
    def getPreprocessingTransform(self, maxSize, backgroundValue, isMri=False):
        from monai.transforms import EnsureChannelFirst, Compose, CenterSpatialCrop
        from ModalityConverterLib.ModelsImpl.FedSynthBrainModelsUtils import CustomResize, PadToCube, MRINormalize
        
        steps = [
                EnsureChannelFirst(channel_dim="no_channel"),
                CenterSpatialCrop((328, 256, 328)),
                CustomResize(maxSize=maxSize),
                PadToCube(backgroundValue=backgroundValue, size=maxSize)
            ]
        
        if isMri:
            steps += [MRINormalize(type="minmax")]

        transform = Compose(steps)

        return transform
        
    def preprocess(self, inputVolume, inputMask=None, showAllFiles=True):
        """Preprocess the input volume for inference."""
        from torch import from_numpy
        if not isinstance(inputVolume, slicer.vtkMRMLScalarVolumeNode):
            raise TypeError("Input must be a vtkMRMLScalarVolumeNode")
        
        print(f"{PRINT_MODULE_SUFFIX} Preprocessing input volume...")

        if not inputMask:
            print(f"{PRINT_MODULE_SUFFIX} No mask provided, generating mask from input volume...")
            slicer.app.processEvents()
            
            maskVolume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", "MaskedInputVolume")

            cliMaskingParams = {
                "inputVolume": inputVolume.GetID(),
                "outputROIMaskVolume": maskVolume.GetID(),
                "fillValue": 0,  # Value assigned to the background,
                "numberOfThreads": 4
            }
            
            cliMaskingNode = slicer.cli.run(slicer.modules.brainsroiauto, None, cliMaskingParams, wait_for_completion=True)
            slicer.mrmlScene.RemoveNode(cliMaskingNode)
        else:
            if not isinstance(inputMask, slicer.vtkMRMLScalarVolumeNode):
                raise TypeError("Input mask must be a vtkMRMLScalarVolumeNode")
            
            maskVolume = inputMask
            
        print(f"{PRINT_MODULE_SUFFIX} Applying N4ITK Bias Field Correction...")
        slicer.app.processEvents()
        correctedInputVolume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", "PreprocessedInputVolume")

        n4Params = {
            "inputImageName": inputVolume.GetID(),
            "maskImageName": maskVolume.GetID(),
            "outputImageName": correctedInputVolume.GetID(),
            "shrinkFactor": 2,         # default is 4, lower value = slower but better
            "numberOfIterations": [50, 40, 30],  # Multi-resolution levels
            "convergenceThreshold": 0.00001,
            "bsplineOrder": 3,
        }

        cliN4Node = slicer.cli.run(slicer.modules.n4itkbiasfieldcorrection, None, n4Params, wait_for_completion=True)
        slicer.mrmlScene.RemoveNode(cliN4Node)

        inputNp = slicer.util.arrayFromVolume(correctedInputVolume)
        maskNp = slicer.util.arrayFromVolume(maskVolume)
        
        if maskNp.min() < 0 or maskNp.max() > 1:
            raise ValueError("Input mask must be binary.")
         
        input_tensor = from_numpy(inputNp).float()  # (1, 1, D, H, W)
        mask_tensor = from_numpy(maskNp)  # (1, 1, D, H, W)
        
        print(f"{PRINT_MODULE_SUFFIX} Applying transforms...")
        mriTransform = self.getPreprocessingTransform(256, backgroundValue=0.0, isMri=True)
        maskTransform = self.getPreprocessingTransform(256, backgroundValue=0)
        preprocessedInput = mriTransform(input_tensor)
        preprocessedMask = maskTransform(mask_tensor)
        
        if showAllFiles:
            slicer.util.updateVolumeFromArray(correctedInputVolume, preprocessedInput.squeeze().cpu().numpy())
            displayNode = correctedInputVolume.GetDisplayNode()
            if displayNode:
                displayNode.SetWindow(1.0)
                displayNode.SetLevel(0)
                
            slicer.util.updateVolumeFromArray(maskVolume, preprocessedMask.squeeze().cpu().numpy())
            
            correctedInputVolume.CopyOrientation(inputVolume)
            maskVolume.CopyOrientation(inputVolume)
        else:
            displayNode = maskVolume.GetDisplayNode()
            if displayNode:
                displayNode.RemoveAllViewIDs()
                displayNode.SetVisibility(False)

            slicer.mrmlScene.RemoveNode(maskVolume)
            
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
        mrVol = zeros(
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

                # Convert NumPy slice to torch and copy into mrVol
                mrVol[0, 0, :, :] = from_numpy(inputSlice).to(self.device).type(mrVol.dtype)

                if 1 in maskSlice:
                    # Prepare input for ONNX Runtime
                    # Add batch and channel dimensions: (1, 1, H, W)
                    ortInputs = {
                        self.model.get_inputs()[0].name: mrVol[0, 0].cpu().numpy()[None, None, :, :]
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
        votedSCT[from_numpy(preprocessedMask == 0).to(self.device)] = -1024

        slicer.util.updateVolumeFromArray(outputVolume, votedSCT.squeeze().cpu().numpy())
        outputVolume.CopyOrientation(inputVolume)
        
        if showAllFiles:
            slicer.util.setSliceViewerLayers(background=outputVolume, foreground=slicer.util.getNode("PreprocessedInputVolume"))
        else:
            slicer.util.setSliceViewerLayers(background=outputVolume, foreground=None)
            
        slicer.util.resetSliceViews()
        
        print(f"{PRINT_MODULE_SUFFIX} Inference completed, sCT volume is ready.")

        return votedSCT
        



