import logging
import os
from typing import Optional

import vtk
import slicer
from slicer.i18n import tr as _
from slicer.i18n import translate
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
from slicer.parameterNodeWrapper import (
    parameterNodeWrapper,
)

from slicer import vtkMRMLScalarVolumeNode

#
# ImageTranslator
#


class ImageTranslator(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        from ImageTranslatorLib.UI.utils import HELP_TEXT, CONTRIBUTORS
        ScriptedLoadableModule.__init__(self, parent)

        self.parent.title = _("ImageTranslator")
        self.parent.categories = [translate("qSlicerAbstractCoreModule", "I2IHub")]
        self.parent.dependencies = []
        self.parent.contributors = CONTRIBUTORS
        self.parent.helpText = _(HELP_TEXT)
        self.parent.acknowledgementText = _("")

        # Additional initialization step after application startup is complete
        #slicer.app.connect("startupCompleted()", registerSampleData)


#
# ImageTranslatorParameterNode
#


@parameterNodeWrapper
class ImageTranslatorParameterNode:
    inputVolume: vtkMRMLScalarVolumeNode
    maskVolume: vtkMRMLScalarVolumeNode
    outputVolume: vtkMRMLScalarVolumeNode  


#
# ImageTranslatorWidget
#


class ImageTranslatorWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    """Uses ScriptedLoadableModuleWidget base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent=None) -> None:
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.__init__(self, parent)
        # needed for parameter node observation
        VTKObservationMixin.__init__(self)
        self.logic = None
        self._parameterNode = None
        self._parameterNodeGuiTag = None
        self.selectedModelKey = None
        self.selectedModelModuleName = None
        self.selectedDeviceKey = None
        self.requiredDeps = ["monai", "onnx", "onnxruntime", "torch"]
        self.dependenciesInstalled = False

    def checkDependencies(self):
        from importlib.util import find_spec
        allPresent = all(find_spec(mod) is not None for mod in self.requiredDeps)

        self.ui.installRequirementsButton.setVisible(not allPresent)
        self.ui.infoLabel.setVisible(not allPresent)
        self.ui.applyButton.setVisible(allPresent)
        self.ui.sampleDataButton.setVisible(allPresent)
        self.dependenciesInstalled = allPresent 

    def setMainButtonsState(self, state: bool = True):
        self.ui.applyButton.setEnabled(state)
        self.ui.sampleDataButton.setEnabled(state)
                
    def onHelpButtonClicked(self):
        from ImageTranslatorLib.UI.HelpDialog import HelpDialog
        dialog = HelpDialog(slicer.util.mainWindow())
        dialog.exec_()

    def populateModelDropdown(self):
        import json

        """Populate the model dropdown dynamically based on metadata.json."""
        modelsDir = os.path.join(os.path.dirname(__file__), "Resources/Models")
        modelsMetadataPath = os.path.join(modelsDir, "metadata.json")

        if not os.path.exists(modelsMetadataPath):
            slicer.util.errorDisplay("Model metadata file not found.")
            return

        with open(modelsMetadataPath, "r") as f:
            modelMetadata = json.load(f)
            self.models_metadata = modelMetadata

        self.ui.modelSelector.clear()

        for modelKey, model_info in self.models_metadata.items():
            displayName = model_info.get("display_name", modelKey)
            description = model_info.get(
                "description", "No description available.")
            moduleName = model_info.get("module_name", None)

            if not modelKey or not moduleName:
                slicer.util.errorDisplay(f"Model key '{modelKey}' or module_name '{moduleName}' is not defined in metadata.json.")
                raise ValueError(f"Model key '{modelKey}' or module_name '{moduleName}' is not defined in metadata.json.")

            self.ui.modelSelector.addItem(displayName, {
                                          "key": modelKey, "description": description, "module_name": moduleName
                                          })

        self.ui.modelSelector.currentIndexChanged.connect(self.onModelSelected)

        if self.ui.modelSelector.count > 0:
            self.ui.modelSelector.setCurrentIndex(0)
            # Force trigger selection for the first item
            self.onModelSelected(0)
        else:
            raise ValueError("No models available.")

    def initDeviceDropdown(self):
        self.ui.deviceList.addItem("cpu [slow]", {"key": "cpu"})
        self.ui.deviceList.currentIndexChanged.connect(self.onDeviceSelected)
        self.ui.deviceList.setCurrentIndex(0)
        # Force trigger selection for the first item
        self.onDeviceSelected(0)
    
    def populateDeviceDropdown(self):
        if self.dependenciesInstalled:
            from torch.cuda import is_available as cuda_available, device_count, get_device_name
            if cuda_available():
                for i in range(device_count()):
                    deviceName = get_device_name(i)
                    self.ui.deviceList.addItem(f"gpu {i} - {deviceName}", {"key": f"cuda:{i}"})

    def onDeviceSelected(self, index):
        """Handle device selection."""
        selected_data = self.ui.deviceList.itemData(index)
        if selected_data:
            self.selectedDeviceKey = selected_data.get("key")

    def onModelSelected(self, index):
        """Handle model selection and display its description."""
        selected_data = self.ui.modelSelector.itemData(index)
        if selected_data:
            self.selectedModelKey = selected_data.get("key")
            self.selectedModelDescription = selected_data.get("description", "No description available.")
            self.selectedModelModuleName = selected_data.get("module_name")
            self.ui.modelDescriptionLabel.setWordWrap(True)
            self.ui.modelDescriptionLabel.setText(f"<b>Description</b>:<br>{self.selectedModelDescription}")

    def onInstallRequirements(self):
        from ImageTranslatorLib.UI.utils import PRINT_MODULE_SUFFIX
        
        if not slicer.util.confirmOkCancelDisplay(
            "The dependencies needed for the extension will be installed, the operation may take a few minutes. A Slicer restart will be necessary.",
            "Press OK to install and restart."
        ):
            raise ValueError("Missing dependencies.")

        self.ui.installRequirementsButton.setEnabled(False)
        slicer.util.setPythonConsoleVisible(True)
        self.ui.infoLabel.setText("Installing missing dependencies, please wait...")
        print(f"{PRINT_MODULE_SUFFIX} Installing missing dependencies, please wait...")

        try:
            for dep in self.requiredDeps:
                print(f"{PRINT_MODULE_SUFFIX} Installing {dep}...")
                slicer.util.pip_install(dep) if dep != "monai" else slicer.util.pip_install("monai[itk]")
            
            slicer.util.pip_install("onnxruntime-gpu")
            
            print(f"{PRINT_MODULE_SUFFIX} All dependencies installed successfully.")
            slicer.app.restart()
        except Exception as e:
            slicer.util.errorDisplay(f"Failed to install requirements: {e}")
            self.ui.installRequirementsButton.setEnabled(True)

    def setup(self) -> None:
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.setup(self)

        # Load widget from .ui file (created by Qt Designer).
        # Additional widgets can be instantiated manually and added to self.layout.
        from qt import QIcon, QSize, QTimer
        
        uiWidget = slicer.util.loadUI(self.resourcePath("UI/ImageTranslator.ui"))
        self.layout.addWidget(uiWidget)

        self.ui = slicer.util.childWidgetVariables(uiWidget)

        # Set scene in MRML widgets. Make sure that in Qt designer the top-level qMRMLWidget's
        # "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
        # "setMRMLScene(vtkMRMLScene*)" slot.
        uiWidget.setMRMLScene(slicer.mrmlScene)


        # Create logic class. Logic implements all computations that should be possible to run
        # in batch mode, without a graphical user interface.
        self.logic = ImageTranslatorLogic()

        # Connections

        # These connections ensure that we update parameter node when scene is closed
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

        self.ui.helpButton.setText("Help  ")
        iconPath = os.path.join(os.path.dirname(__file__), 'Resources', 'Icons', 'question.png')
        self.ui.helpButton.setIcon(QIcon(iconPath))
        self.ui.helpButton.setIconSize(QSize(16, 16))
        self.ui.helpButton.connect("clicked(bool)", self.onHelpButtonClicked)
        
        self.ui.applyButton.connect("clicked(bool)", self.onApplyButton)
        self.ui.sampleDataButton.connect('clicked(bool)', self.onSampleDataButtonClicked)
        self.ui.installRequirementsButton.connect("clicked(bool)", self.onInstallRequirements)
        self.ui.installRequirementsButton.setVisible(False)

        self.initializeParameterNode()
        
        self.checkDependencies()
        
        self.initDeviceDropdown()
        
        # importing torch functions to check gpu availability block and delay the UI initialization. 
        # This timer ensures the dropdown is populated with gpu options right after the UI is loaded.
        QTimer.singleShot(0, self.populateDeviceDropdown) 
        
    def cleanup(self) -> None:
        """Called when the application closes and the module widget is destroyed."""
        self.removeObservers()

    def enter(self) -> None:
        """Called each time the user opens this module."""
        # Make sure parameter node exists and observed
        self.initializeParameterNode()
        self.populateModelDropdown()
        


    def exit(self) -> None:
        """Called each time the user opens a different module."""
        # Do not react to parameter node changes (GUI will be updated when the user enters into the module)
        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            self._parameterNodeGuiTag = None
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)

    def onSceneStartClose(self, caller, event) -> None:
        """Called just before the scene is closed."""
        # Parameter node will be reset, do not use it anymore
        self.setParameterNode(None)

    def onSceneEndClose(self, caller, event) -> None:
        """Called just after the scene is closed."""
        # If this module is shown while the scene is closed then recreate a new parameter node immediately
        if self.parent.isEntered:
            self.initializeParameterNode()

    def initializeParameterNode(self) -> None:
        """Ensure parameter node exists and observed."""
        # Parameter node stores all user choices in parameter values, node selections, etc.
        # so that when the scene is saved and reloaded, these settings are restored.

        self.setParameterNode(self.logic.getParameterNode())

        # Select default input nodes if nothing is selected yet to save a few clicks for the user
        if not self._parameterNode.inputVolume:
            firstVolumeNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
            if firstVolumeNode:
                self._parameterNode.inputVolume = firstVolumeNode
            
    def setParameterNode(self, inputParameterNode: Optional[ImageTranslatorParameterNode]) -> None:
        """
        Set and observe parameter node.
        Observation is needed because when the parameter node is changed then the GUI must be updated immediately.
        """
        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)
        self._parameterNode = inputParameterNode
        if self._parameterNode:
            # Note: in the .ui file, a Qt dynamic property called "SlicerParameterName" is set on each
            # ui element that needs connection.
            self._parameterNodeGuiTag = self._parameterNode.connectGui(self.ui)
            self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)
            self._checkCanApply()

    def _checkCanApply(self, caller=None, event=None) -> None:
        if self._parameterNode and self._parameterNode.inputVolume and self._parameterNode.outputVolume:
            self.ui.applyButton.toolTip = _("Compute output volume")
            self.ui.applyButton.enabled = True
        else:
            self.ui.applyButton.toolTip = _("Select input and output volume nodes")
            self.ui.applyButton.enabled = False

    def onSampleDataButtonClicked(self):
        """Open "Sample Data" module when user clicks "Download sample" button."""
        slicer.util.selectModule('SampleData')
        
    def updateInfoLabel(self, text: str) -> None:
        """Update the info label with the provided text."""
        self.ui.infoLabel.setVisible(True)
        self.ui.infoLabel.setText(text)
        slicer.app.processEvents()

    def onApplyButton(self) -> None:
        """Run processing when user clicks "Apply" button."""
        with slicer.util.tryWithErrorDisplay(_("Failed to compute results."), waitCursor=True):
            self.setMainButtonsState(False)
            slicer.util.setPythonConsoleVisible(True)
            slicer.util.resetSliceViews()
            self.updateInfoLabel("Processing, please wait...")

            self.logic.process(inputVolume=self.ui.inputSelector.currentNode(),
                               outputVolume=self.ui.outputSelector.currentNode(),
                               maskVolume=self.ui.maskSelector.currentNode(),
                               showAllFiles=self.ui.showAllFilesCheckBox.isChecked(),
                               selectedModelKey=self.selectedModelKey,
                               selectedModelModuleName=self.selectedModelModuleName,
                               device=self.selectedDeviceKey)
            
            self.updateInfoLabel("Processing completed successfully.")
            self.setMainButtonsState(True)
#
# ImageTranslatorLogic
#

class ImageTranslatorLogic(ScriptedLoadableModuleLogic):
    def __init__(self) -> None:
        """Called when the logic class is instantiated. Can be used for initializing member variables."""
        ScriptedLoadableModuleLogic.__init__(self)
        self.model = None

    def getParameterNode(self):
        return ImageTranslatorParameterNode(super().getParameterNode())

    def getModelInstance(self, selectedModelModuleName, selectedModelKey, basePackage="ImageTranslatorLib.ModelsImpl", device="cpu"):
        """
        Dynamically loads and returns an instance of a model class based on the provided module name and model key.

        Parameters:
        - selectedModelModuleName (str): The name of the module containing the model class.
        - selectedModelKey (str): The key identifying the model in the model registry.
        - basePackage (str): The base package path where the model modules are located. Defaults to "ImageTranslatorLib.ModelsImpl".

        Returns:
        - object: An instance of the model class corresponding to the provided key.

        Raises:
        - ImportError: If the specified module cannot be imported.
        - ValueError: If the model key is not found in the model registry.
        """

        import importlib
        from ImageTranslatorLib.ModelBase import MODEL_REGISTRY

        # Importing the model module dynamically will trigger the registration of the model class in the model registry
        fullModulePath = f"{basePackage}.{selectedModelModuleName}"
        try:
            logging.info(f"Attempting to import module: {fullModulePath}")
            importlib.import_module(fullModulePath)
        except ImportError as e:
            raise ImportError(f"Could not import module '{fullModulePath}' for model '{selectedModelKey}': {str(e)}")

        if selectedModelKey not in MODEL_REGISTRY:
            raise ValueError(f"Model key '{selectedModelKey}' not found in the model registry. Please check the metadata.json file and ensure the key is correctly defined.")

        model_class = MODEL_REGISTRY[selectedModelKey]

        return model_class(selectedModelKey, device)

    def process(self,
                inputVolume: vtkMRMLScalarVolumeNode,  # volume to be translated
                outputVolume: vtkMRMLScalarVolumeNode,  # result
                maskVolume: vtkMRMLScalarVolumeNode,  # mask volume [optional]
                selectedModelKey: str = "",
                selectedModelModuleName: str = None,
                showAllFiles: bool = True,
                device: str = "cpu") -> None:  # model key for selection
                
        if not inputVolume or not outputVolume:
            raise ValueError("Input or output volume is invalid")
        
        modelInstance = self.getModelInstance(selectedModelModuleName, selectedModelKey, device=device)
        modelInstance.loadModel()

        # Run inference using the loaded model
        modelInstance.runInference(inputVolume=inputVolume, inputMask=maskVolume, outputVolume=outputVolume, showAllFiles=showAllFiles)