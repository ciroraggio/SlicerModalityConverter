from abc import ABC, abstractmethod
import os
from slicer import vtkMRMLScalarVolumeNode
from ImageTranslatorLib.UI.utils import PRINT_MODULE_SUFFIX

# Global registry for model classes
MODEL_REGISTRY = {}

def register_model(key):
    """Decorator to register a model class with a specific key in the metadata.json."""
    def decorator(cls):
        print(f"{PRINT_MODULE_SUFFIX} Registering model '{key}' with class {cls.__name__}")
        MODEL_REGISTRY[key] = cls
        return cls
    return decorator

"""Base class for all models in the ImageTranslator library."""
class BaseModel(ABC):
    def __init__(self, modelKey: str, device: str = "cpu"):
        self.modelKey = modelKey
        self.model = None
        self.baseModelsDir = os.path.join(os.path.dirname(__file__), '../Resources/Models')
        self.modelsDir = os.path.abspath(self.baseModelsDir)
        self.device = device.lower()
        
    def loadModel(self):
        import json
        import requests
        
        modelMetadataPath = os.path.join(self.modelsDir, "model_metadata.json")

        # Ensure the models directory exists
        if not os.path.exists(self.modelsDir):
            os.makedirs(self.modelsDir)

        # Search the key file by unique key
        for file in os.listdir(self.modelsDir):
            if file.startswith(self.modelKey):
                self.modelPath = os.path.join(self.modelsDir, file)
                break
        else:
            self.modelPath = None  # No files found with the specified key

        # Download model if not present locally
        if not os.path.exists(self.modelPath):
            
            # Load model metadata
            if not os.path.exists(modelMetadataPath):
                raise FileNotFoundError(f"Model metadata file not found at {modelMetadataPath}")

            with open(modelMetadataPath, "r") as f:
                modelMetadata = json.load(f)

            if self.modelKey not in modelMetadata:
                raise ValueError(f"Model key '{self.modelKey}' not found in metadata file.")
            
            print(f"{PRINT_MODULE_SUFFIX} Model '{self.modelKey}' not found locally. Downloading...")
            url = modelMetadata[self.modelKey]["url"]
            response = requests.get(url, stream=True)
            response.raise_for_status()

            with open(self.modelPath, "wb") as modelFile:
                for chunk in response.iter_content(chunk_size=8192): # Download in chunks of 8KB
                    modelFile.write(chunk)

        # Load the model with custom subclasses method
        self.model = self._loadModelFromPath(self.modelPath)

    @abstractmethod
    def _loadModelFromPath(self, modelPath: str):
        """Load the model from the given path. Will be called in the load_data method. To be implemented in subclasses for loading customization."""
        pass

    @abstractmethod
    def runInference(self, inputVolume: vtkMRMLScalarVolumeNode, outputVolume: vtkMRMLScalarVolumeNode, inputMask: vtkMRMLScalarVolumeNode=None, showAllFiles: bool=True):
        pass
