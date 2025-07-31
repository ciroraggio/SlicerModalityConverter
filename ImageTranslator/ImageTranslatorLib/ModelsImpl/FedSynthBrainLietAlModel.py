import torch
from ImageTranslatorLib.ModelBase import BaseModel, register_model
import slicer
from slicer import vtkMRMLScalarVolumeNode
import os

@register_model("fedsynthct_lietal_t1w_brain")
class FedSynthBrainLietAlModel(BaseModel):
    def __init__(self, modelKey: str, device: str = "cpu"):
        super().__init__(modelKey, device)
        
    def _loadModelFromPath(self, modelPath):
        import onnxruntime as ort
        """Load the model using ONNX Runtime."""
        if not os.path.exists(modelPath):
            raise FileNotFoundError(f"Model file not found at {modelPath}")
    
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if self.device.startswith("cuda") else ["CPUExecutionProvider"]
            
    
        try:
            print(f"Loading ONNX model with providers: {providers}")
            model = ort.InferenceSession(modelPath, providers=providers)
            return model
        except Exception as e:
            raise RuntimeError(f"Failed to load ONNX model from {modelPath}: {str(e)}")
    
    def getPreprocessingTransform(self, maxSize, backgroundValue, isMri=False):
        from monai.transforms import EnsureChannelFirst, Compose, CenterSpatialCrop
        from ImageTranslatorLib.ModelsImpl.FedSynthBrainLietAlModelUtils import CustomResize, PadToCube, MRINormalize
        
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
        if not isinstance(inputVolume, slicer.vtkMRMLScalarVolumeNode):
            raise TypeError("Input must be a vtkMRMLScalarVolumeNode")
        
        print("Preprocessing input volume...")

        if not inputMask:
            print("No mask provided, generating mask from input volume...")
            
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
            
        print("Applying N4ITK Bias Field Correction...")
        correctedInputVolume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", "PreprocessedMRIVolume")

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
         
        input_tensor = torch.from_numpy(inputNp).float()  # (1, 1, D, H, W)
        mask_tensor = torch.from_numpy(maskNp)  # (1, 1, D, H, W)
        
        print("Applying transforms...")
        mriTransform = self.getPreprocessingTransform(256, backgroundValue=0.0, isMri=True)
        maskTransform = self.getPreprocessingTransform(256, backgroundValue=0)
        preprocessedInput = mriTransform(input_tensor)
        preprocessedMask = maskTransform(mask_tensor)
        
        if showAllFiles:
            slicer.util.updateVolumeFromArray(correctedInputVolume, preprocessedInput.squeeze().cpu().numpy())
            slicer.util.updateVolumeFromArray(maskVolume, preprocessedMask.squeeze().cpu().numpy())
            
            correctedInputVolume.CopyOrientation(inputVolume)
            maskVolume.CopyOrientation(inputVolume)
            
        print("Preprocessing completed.")
        
        return {"input": preprocessedInput, "mask": preprocessedMask}
    
    def runInference(self, inputVolume: vtkMRMLScalarVolumeNode, outputVolume: vtkMRMLScalarVolumeNode, inputMask: vtkMRMLScalarVolumeNode = None, showAllFiles: bool = True):
        preprocessedData = self.preprocess(inputVolume, inputMask, showAllFiles)

        preprocessedInput = preprocessedData["input"].cpu().numpy()  # Convert to NumPy array
        preprocessedMask = preprocessedData["mask"].cpu().numpy()

        # Prepare containers
        sCT = {
            view: torch.zeros(preprocessedInput.shape, device=self.device)
            for view in ["first_plane", "second_plane", "third_plane"]
        }
        MRSlice = torch.zeros(
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

                # Convert NumPy slice to torch and copy into mr_slice
                MRSlice[0, 0, :, :] = torch.from_numpy(inputSlice).to(self.device).type(MRSlice.dtype)

                if 1 in maskSlice:
                    # Prepare input for ONNX Runtime
                    # Add batch and channel dimensions: (1, 1, H, W)
                    ortInputs = {
                        self.model.get_inputs()[0].name: MRSlice[0, 0].cpu().numpy()[None, None, :, :]
                    }
                    ortOuts = self.model.run(None, ortInputs)

                    sCT_slice = torch.tensor(ortOuts[0]).to(self.device).type(sCT[view].dtype)

                    # Assign slice back to correct axis
                    if view == "first_plane":
                        sCT[view][:, sliceIndex, :, :] = sCT_slice
                    elif view == "second_plane":
                        sCT[view][:, :, sliceIndex, :] = sCT_slice
                    elif view == "third_plane":
                        sCT[view][:, :, :, sliceIndex] = sCT_slice

        # Median voting across views
        votedSCT, _ = torch.median(
            torch.stack([sCT["first_plane"], sCT["second_plane"], sCT["third_plane"]], dim=0), dim=0
        )
        votedSCT[torch.from_numpy(preprocessedMask == 0).to(self.device)] = -1024

        # Export to Slicer volume
        slicer.util.updateVolumeFromArray(outputVolume, votedSCT.squeeze().cpu().numpy())

        # Copy geometry and show
        outputVolume.CopyOrientation(inputVolume)
        slicer.util.setSliceViewerLayers(background=outputVolume, foreground=slicer.util.getNode("PreprocessedMRIVolume"))
        slicer.util.resetSliceViews()
        
        print("Inference completed. sCT volume is ready.")

        return votedSCT




