import logging
import os
import sys
import uuid
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
# ModalityConverter
#


class ModalityConverter(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        from ModalityConverterLib.UI.utils import HELP_TEXT, CONTRIBUTORS
        ScriptedLoadableModule.__init__(self, parent)

        self.parent.title = _("ModalityConverter")
        self.parent.categories = [translate("qSlicerAbstractCoreModule", "Image Synthesis")]
        self.parent.dependencies = []
        self.parent.contributors = CONTRIBUTORS
        self.parent.helpText = _(HELP_TEXT)
        self.parent.acknowledgementText = _("")

        # Additional initialization step after application startup is complete
        #slicer.app.connect("startupCompleted()", registerSampleData)


#
# ModalityConverterParameterNode
#


@parameterNodeWrapper
class ModalityConverterParameterNode:
    inputVolume: vtkMRMLScalarVolumeNode
    maskVolume: vtkMRMLScalarVolumeNode
    outputVolume: vtkMRMLScalarVolumeNode  

#
# ModalityConverterWidget
#


class ModalityConverterWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
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
        self.maskRequiredForSelectedModel = False
        self.selectedDeviceKey = None
        self.requiredDeps = ["monai", "onnx", "onnxruntime", "torch", "nibabel", "SimpleITK", "requests_toolbelt"]
        self.dependenciesInstalled = False
        self.inferenceProcessManager = None
        self.currentRunToken = None
        self.runContext = None
        self.activeRunId = None
        self.pendingCliNode = None
        self.brainPreprocessing = None
        self.brainCancellationRequested = False
        self._runGuiBusy = False
        self._resourceTimer = None
        self._cpuSample = None
        self.remoteServerUrl = None
        self.remoteToken = None
        self.remoteResources = None
        self.remoteSites = self._loadRemoteSiteProfiles()
        self.activeRemoteSite = None
        
        self.POLLING_UPLOADING_TIMER_SHORT = 1000  # milliseconds
        self.POLLING_UPLOADING_TIMER_LONG = 5000  # milliseconds
        

    def checkDependencies(self):
        from importlib.util import find_spec
        allPresent = all(find_spec(mod) is not None for mod in self.requiredDeps)
        onnxRuntimeError = None
        if allPresent:
            try:
                import onnxruntime
                onnxruntime.get_available_providers()
            except Exception as e:
                allPresent = False
                onnxRuntimeError = str(e)

        self.ui.installRequirementsButton.setVisible(not allPresent)
        self.ui.infoLabel.setVisible(not allPresent)
       
        if onnxRuntimeError:
            self.ui.infoLabel.setText(
                "ONNX Runtime cannot be loaded ({}). Use Install dependencies to repair the CPU runtime."
                .format(onnxRuntimeError))
        
        self.ui.applyButton.setVisible(allPresent)
        self.ui.sampleDataButton.setVisible(allPresent)
        self.dependenciesInstalled = allPresent 

    def setMainButtonsState(self, state: bool = True):
        self.ui.applyButton.setEnabled(state)
        self.ui.sampleDataButton.setEnabled(state)
                
    def onHelpButtonClicked(self):
        from ModalityConverterLib.UI.HelpDialog import HelpDialog
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
            description = model_info.get("description", "No description available.")
            moduleName = model_info.get("module_name", None)
            maskRequired = model_info.get("mask_required", False)
            modelDeprecated = model_info.get("deprecated", None)

            if not modelKey or not moduleName:
                slicer.util.errorDisplay(f"Model key '{modelKey}' or module_name '{moduleName}' is not defined in metadata.json.")
                raise ValueError(f"Model key '{modelKey}' or module_name '{moduleName}' is not defined in metadata.json.")
            
            if not modelDeprecated:
                self.ui.modelSelector.addItem(displayName, {
                                          "key": modelKey, "description": description, "module_name": moduleName, "mask_required": maskRequired
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
        if not getattr(self, "_deviceSignalConnected", False):
            self.ui.deviceList.currentIndexChanged.connect(self.onDeviceSelected)
            self._deviceSignalConnected = True
        self.ui.deviceList.setCurrentIndex(0)
        # Force trigger selection for the first item
        self.onDeviceSelected(0)
    
    def populateDeviceDropdown(self):
        if self.dependenciesInstalled:
            if not sys.platform.startswith('darwin'):
                from torch.cuda import is_available as cuda_available, device_count, get_device_name
                from onnxruntime import get_available_providers
                hasCudaProvider = "CUDAExecutionProvider" in get_available_providers()
                if cuda_available() and hasCudaProvider:
                    for i in range(device_count()):
                        deviceName = get_device_name(i)
                        self.ui.deviceList.addItem(f"gpu {i} - {deviceName}", {"key": f"cuda:{i}"})
            else:
                from onnxruntime import get_available_providers
                available_providers = get_available_providers()
                
                if "CoreMLExecutionProvider" in available_providers:
                    self.ui.deviceList.addItem("coreml (Apple GPU/NPU)", {"key": "coreml"})
                    
    def onDeviceSelected(self, index):
        """Handle device selection."""
        selected_data = self.ui.deviceList.itemData(index)
        if selected_data:
            self.selectedDeviceKey = selected_data.get("key")
            if hasattr(self, "resourceUsageWidget"):
                self.updateResourceUsage()

    def onModelSelected(self, index):
        """Handle model selection and display its description."""
        selected_data = self.ui.modelSelector.itemData(index)
        if selected_data:
            self.selectedModelKey = selected_data.get("key")
            self.selectedModelDescription = selected_data.get("description", "No description available.")
            self.selectedModelModuleName = selected_data.get("module_name")
            self.maskRequiredForSelectedModel = selected_data.get("mask_required")
            self.ui.modelDescriptionLabel.setWordWrap(True)
            self.ui.modelDescriptionLabel.setText(f"<b>Description</b>:<br>{self.selectedModelDescription}")
            self.ui.labelInputMask.setText("ROI Mask" if selected_data.get("mask_required") else "ROI Mask (Optional)")
            self._checkCanApply()

    def onInstallRequirements(self):
        from ModalityConverterLib.UI.utils import PRINT_MODULE_SUFFIX
        
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
                if dep == "monai":
                    slicer.util.pip_install("monai[itk]")
                elif dep == "onnxruntime":
                    # Keep the CPU distribution usable by default. GPU-enabled
                    # wheels require a matching CUDA runtime on the host.
                    slicer.util.pip_install("--force-reinstall onnxruntime")
                else:
                    slicer.util.pip_install(dep)
            
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
        from qt import QIcon, QSize, QTimer, QProgressBar, Qt
        
        uiWidget = slicer.util.loadUI(self.resourcePath("UI/ModalityConverter.ui"))
        self.layout.addWidget(uiWidget)

        self.ui = slicer.util.childWidgetVariables(uiWidget)

        # Set scene in MRML widgets. Make sure that in Qt designer the top-level qMRMLWidget's
        # "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
        # "setMRMLScene(vtkMRMLScene*)" slot.
        uiWidget.setMRMLScene(slicer.mrmlScene)

        self.logic = ModalityConverterLogic()
        from ModalityConverterLib.InferenceProcessManager import InferenceProcessManager

        self.inferenceProcessManager = InferenceProcessManager()
        self.inferenceProcessManager.progressChanged.connect(self.onInferenceProgress)
        self.inferenceProcessManager.finished.connect(self.onInferenceFinished)
        self.inferenceProcessManager.failed.connect(self.onInferenceFailed)
        self.inferenceProcessManager.cancelled.connect(self.onInferenceCancelled)
        self.ui.applyButton.connect("clicked(bool)", self.onRunButtonClicked)
        self.ui.applyButton.setStyleSheet("background-color: rgb(52, 206, 165);")
        self.progressBar = QProgressBar(uiWidget)
        self.progressBar.setRange(0, 100)
        self.progressBar.setValue(0)
        self.progressBar.setVisible(False)
        self.layout.addWidget(self.progressBar)

        advancedLayout = self.ui.advancedCollapsibleButton.layout()
        from ModalityConverterLib.UI.RemoteConnectionWidget import RemoteConnectionWidget
        from ModalityConverterLib.UI.ResourceUsageWidget import ResourceUsageWidget
        self.remoteUI = RemoteConnectionWidget(self.ui.advancedCollapsibleButton)
        self.remoteSiteSelector = self.remoteUI.siteSelector
        self.remoteManageButton = self.remoteUI.manageButton
        self.remoteConnectButton = self.remoteUI.connectButton
        self.remoteStatusLabel = self.remoteUI.statusLabel
        for site in self.remoteSites:
            self.remoteSiteSelector.addItem(site["name"], site)
        savedSiteName = str(slicer.app.settings().value("ModalityConverter/selectedRemoteSite", ""))
        savedSiteIndex = self.remoteSiteSelector.findText(savedSiteName)
        if savedSiteIndex > 0:
            self.remoteSiteSelector.setCurrentIndex(savedSiteIndex)
        self.remoteServerUrl = None
        self.remoteToken = None
        self.remoteResources = None
        advancedLayout.addRow(self.remoteUI)
        self.remoteConnectButton.clicked.connect(self.onRemoteConnectClicked)
        self.remoteManageButton.clicked.connect(self.onManageRemoteSites)
        self.remoteSiteSelector.currentIndexChanged.connect(self.onRemoteSiteSelected)
        self.resourceUsageWidget = ResourceUsageWidget(
            self.ui.advancedCollapsibleButton, advancedLayout)
        self._resourceTimer = QTimer(uiWidget)
        self._resourceTimer.timeout.connect(self.updateResourceUsage)
        self._resourceTimer.start(self.POLLING_UPLOADING_TIMER_SHORT)
        self.updateResourceUsage()

        # Connections

        # These connections ensure that we update parameter node when scene is closed
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

        self.ui.helpButton.setText("  Guide")
        iconPath = os.path.join(os.path.dirname(__file__), 'Resources', 'Icons', 'book.png')
        self.ui.helpButton.setIcon(QIcon(iconPath))
        self.ui.helpButton.setIconSize(QSize(16, 16))
        self.ui.helpButton.setLayoutDirection(Qt.LeftToRight)
        self.ui.helpButton.connect("clicked(bool)", self.onHelpButtonClicked)

        iconDir = os.path.join(os.path.dirname(__file__), 'Resources', 'Icons')
        self.ui.sampleDataButton.setIcon(QIcon(os.path.join(iconDir, 'download.png')))
        self.ui.sampleDataButton.setIconSize(QSize(16, 16))
        self.ui.sampleDataButton.setLayoutDirection(Qt.LeftToRight)
        self.runIcon = QIcon(os.path.join(iconDir, 'play-button.png'))
        self.stopIcon = QIcon(os.path.join(iconDir, 'pause.png'))
        self.ui.applyButton.setIcon(self.runIcon)
        self.ui.applyButton.setIconSize(QSize(16, 16))
        self.ui.applyButton.setLayoutDirection(Qt.LeftToRight)

        self.ui.sampleDataButton.connect('clicked(bool)', self.onSampleDataButtonClicked)
        self.ui.installRequirementsButton.connect("clicked(bool)", self.onInstallRequirements)
        self.ui.installRequirementsButton.setVisible(False)

        self.initializeParameterNode()
        
        self.checkDependencies()
        
        self.populateModelDropdown()
        
        self.initDeviceDropdown()
        
        # importing torch functions to check gpu availability block and delay the UI initialization. 
        # This timer ensures the dropdown is populated with gpu options right after the UI is loaded.
        QTimer.singleShot(0, self.populateDeviceDropdown) 
        
    def cleanup(self) -> None:
        """Called when the application closes and the module widget is destroyed."""
        if self.pendingCliNode:
            slicer.cli.cancel(self.pendingCliNode)
        if self.inferenceProcessManager:
            self.inferenceProcessManager.cleanup()
        if self._resourceTimer:
            self._resourceTimer.stop()
        self._cleanupBrainPreprocessing()
        self.removeObservers()

    def enter(self) -> None:
        """Called each time the user opens this module."""
        try:
            dataProbe = slicer.util.findChild(slicer.util.mainWindow(), "DataProbeCollapsibleWidget")
            if dataProbe:
                dataProbe.collapsed = True
        except Exception:
            logging.debug("Could not collapse Data Probe", exc_info=True)
        # Make sure parameter node exists and observed
        self.initializeParameterNode()


    def exit(self) -> None:
        """Called each time the user opens a different module."""
        # Do not react to parameter node changes (GUI will be updated when the user enters into the module)
        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            self._parameterNodeGuiTag = None
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)

    def onSceneStartClose(self, caller, event) -> None:
        """Called just before the scene is closed."""
        if self.pendingCliNode:
            self.brainCancellationRequested = True
            slicer.cli.cancel(self.pendingCliNode)
        if self.inferenceProcessManager and self.inferenceProcessManager.isRunning:
            self.currentRunToken = None
            self.runContext = None
            self.inferenceProcessManager.cleanup()
            self._cleanupBrainPreprocessing()
            self._restoreRunGui()
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
            
    def setParameterNode(self, inputParameterNode: Optional[ModalityConverterParameterNode]) -> None:
        """
        Set and observe parameter node.
        Observation is needed because when the parameter node is changed then the GUI must be updated immediately.
        """
        self._parameterNode = inputParameterNode
        if self._parameterNode:
            # Note: in the .ui file, a Qt dynamic property called "SlicerParameterName" is set on each
            # ui element that needs connection.
            self._parameterNodeGuiTag = self._parameterNode.connectGui(self.ui)
            self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)
            self._checkCanApply()

    def _checkCanApply(self, caller=None, event=None) -> None:
        hasIO = (
            self._parameterNode and
            self._parameterNode.inputVolume and
            self._parameterNode.outputVolume
        )

        hasMaskIfNeeded = (
            not self.maskRequiredForSelectedModel or
            self._parameterNode.maskVolume
        )

        canApply = hasIO and hasMaskIfNeeded

        self.ui.applyButton.enabled = True if self._runGuiBusy else bool(canApply)
        self.ui.applyButton.toolTip = (
            _("Compute output volume")
            if canApply
            else _(
                    "Input mask is required for this model but it was not provided. "
                    "The automatic extraction for this model is not yet supported.\n"
                    "Please, provide a binary mask volume and retry!"
                )
            if self.maskRequiredForSelectedModel
            else _("Select input and output volume nodes")
        )


    def onSampleDataButtonClicked(self):
        """Open "Sample Data" module when user clicks "Download sample" button."""
        slicer.util.selectModule('SampleData')
        
    def updateInfoLabel(self, text: str) -> None:
        """Update the info label with the provided text."""
        self.ui.infoLabel.setVisible(True)
        self.ui.infoLabel.setText(text)

    def onApplyButton(self) -> None:
        try:
            runSpec = {
                "inputVolume": self.ui.inputSelector.currentNode(),
                "outputVolume": self.ui.outputSelector.currentNode(),
                "maskVolume": self.ui.maskSelector.currentNode(),
                "showAllFiles": self.ui.showAllFilesCheckBox.isChecked(),
                "modelKey": self.selectedModelKey,
                "moduleName": self.selectedModelModuleName,
                "device": self.selectedDeviceKey,
            }
            self.activeRunId = str(uuid.uuid4())
            self.setRunGuiBusy(True)
            if (runSpec["moduleName"] or "").startswith("FedSynthBrain"):
                # These models call Slicer's brainsroiauto and N4 CLIs. Run those
                # Slicer-only preprocessing steps on the client, then send the
                # corrected array and mask to the remote inference worker.
                self.brainCancellationRequested = False
                self.brainPreprocessing = {"runSpec": runSpec, "generatedMask": None,
                                           "correctedInput": None, "cliNodes": [],
                                           "handledCliIds": set(), "observers": {}}
                self.updateInfoLabel("Preparing brain mask and N4 correction...")
                self._startBrainPreprocessing(self.activeRunId)
            elif self.remoteServerUrl:
                context = self.logic.prepareInference(
                    inputVolume=runSpec["inputVolume"], outputVolume=runSpec["outputVolume"],
                    maskVolume=runSpec["maskVolume"], showAllFiles=runSpec["showAllFiles"],
                    selectedModelKey=runSpec["modelKey"], selectedModelModuleName=runSpec["moduleName"],
                    device=runSpec["device"])
                self._launchRemoteInference(context)
            else:
                context = self.logic.prepareInference(**{
                    "inputVolume": runSpec["inputVolume"], "outputVolume": runSpec["outputVolume"],
                    "maskVolume": runSpec["maskVolume"], "showAllFiles": runSpec["showAllFiles"],
                    "selectedModelKey": runSpec["modelKey"], "selectedModelModuleName": runSpec["moduleName"],
                    "device": runSpec["device"]})
                self._launchInference(context)
        except Exception as e:
            self._cleanupBrainPreprocessing()
            self.activeRunId = None
            self.runContext = None
            self._restoreRunGui()
            slicer.util.errorDisplay("Could not start inference: {}".format(e))
            logging.exception("Could not start inference")

    def onRunButtonClicked(self, checked=False):
        if self._runGuiBusy:
            self.onStopInference()
        else:
            self.onApplyButton()

    def _loadRemoteSiteProfiles(self):
        import json
        profilesPath = os.path.join(os.path.dirname(__file__), "Resources", "RemoteSites", "sites.json")
        try:
            with open(profilesPath, "r", encoding="utf-8") as profilesFile:
                profiles = json.load(profilesFile)
            if not isinstance(profiles, list):
                raise ValueError("Remote sites file must contain a JSON array")
            # Migrate profiles saved by older builds in QSettings when the new
            # packaged JSON is still empty.
            if not profiles:
                legacyRaw = slicer.app.settings().value("ModalityConverter/remoteSites", "[]")
                legacyProfiles = json.loads(str(legacyRaw))
                if isinstance(legacyProfiles, list) and legacyProfiles:
                    profiles = legacyProfiles
            return [{"name": item.get("name", ""), "host": item.get("host", ""),
                     "port": str(item.get("port", "8765")), "token": "",
                     "scheme": str(item.get("scheme", "http")).lower(),
                     "certificate": str(item.get("certificate", ""))}
                    for item in profiles if item.get("name") and item.get("host")]
        except Exception:
            logging.exception("Could not load saved remote site profiles from %s", profilesPath)
            return []

    def _saveRemoteSiteProfiles(self, sites):
        import json
        # Bearer tokens are session-only; persist only non-secret connection details.
        profiles = [{"name": site["name"], "host": site["host"], "port": site["port"],
                    "scheme": site.get("scheme", "http"), "certificate": site.get("certificate", "")}
                    for site in sites]
        profilesPath = os.path.join(os.path.dirname(__file__), "Resources", "RemoteSites", "sites.json")
        with open(profilesPath, "w", encoding="utf-8") as profilesFile:
            json.dump(profiles, profilesFile, indent=2, ensure_ascii=False)
            profilesFile.write("\n")

    def _disconnectRemoteSite(self, reason=None):
        """Reset all remote UI/session state after the server stops responding."""
        self.remoteServerUrl = self.remoteToken = self.remoteResources = None
        self.activeRemoteSite = None
        self.remoteUI.setConnected(False)
        self.ui.deviceList.clear()
        self.initDeviceDropdown()
        self.populateDeviceDropdown()
        message = "Remote site disconnected" if not reason else "Remote site disconnected · {}".format(reason)
        self.remoteStatusLabel.setText("● Offline · {}".format(message))
        self.remoteStatusLabel.setStyleSheet("color: #b36b00; font-weight: 600;")
        self.updateResourceUsage()

    def onRemoteConnectClicked(self):
        if self.remoteServerUrl:
            self._disconnectRemoteSite()
            return
        try:
            import requests
            site = self.remoteSiteSelector.currentData
            if not site:
                raise ValueError("Add and select a remote site first")
            host = site["host"].strip().rstrip("/")
            if not host:
                raise ValueError("Enter the server address")
            scheme = site.get("scheme", "http").lower()
            if scheme not in ("http", "https"):
                raise ValueError("Choose HTTP or HTTPS")
            host = host.replace("http://", "").replace("https://", "").strip("/")
            port = int(site["port"])
            if not 1 <= port <= 65535:
                raise ValueError("Port must be between 1 and 65535")
            url = "{}://{}:{}".format(scheme, host, port)
            certificate = site.get("certificate", "").strip() if scheme == "https" else True
            if scheme == "https" and (not certificate or not os.path.isfile(certificate)):
                raise ValueError("Select the server public certificate file for HTTPS")
            token = site.get("token", "").strip()
            if not token:
                raise ValueError("Update the bearer token for this site in Manage sites before connecting")
            headers = {"Authorization": "Bearer " + token}
            response = requests.get(url + "/api/v1/health", headers=headers, timeout=4, verify=certificate)
            response.raise_for_status()
            health = response.json()
            if not isinstance(health.get("models"), dict):
                raise ValueError("Remote server returned an invalid model catalog")
            resourcesResponse = requests.get(url + "/api/v1/resources", headers=headers, timeout=5, verify=certificate)
            resourcesResponse.raise_for_status()
            self.remoteServerUrl, self.remoteToken = url, token
            site["certificate"] = certificate if scheme == "https" else ""
            self.activeRemoteSite = site
            self.ui.deviceList.clear()
            self.ui.deviceList.addItem("Remote CPU", {"key": "cpu"})
            resources = resourcesResponse.json()
            self.remoteResources = resources
            remoteGpuAvailable = bool(resources.get("cuda_available") and resources.get("gpus"))
            for gpu in resources.get("gpus", []) if resources.get("cuda_available") else []:
                cudaIndex = gpu.get("cuda_index", gpu["index"])
                self.ui.deviceList.addItem("Remote GPU {} - {}".format(gpu["index"], gpu["name"]),
                                           {"key": "cuda:{}".format(cudaIndex)})
            self.ui.deviceList.setCurrentIndex(1 if remoteGpuAvailable else 0)
            if resources.get("gpus") and not resources.get("cuda_available"):
                reason = resources.get("cuda_reason") or "CUDA runtime unavailable"
                self.remoteStatusLabel.setText("● Online · {}; GPU unavailable, using CPU ({})".format(site["name"], reason))
            else:
                self.remoteStatusLabel.setText("● Online · {} · {}".format(site["name"], url))
            self.remoteUI.setConnected(True)
            self.remoteStatusLabel.setStyleSheet("color: #16803c; font-weight: 600;")
            self.updateResourceUsage()
        except Exception as exc:
            self._disconnectRemoteSite(str(exc))
            slicer.util.errorDisplay("Could not connect to remote server: {}".format(exc))

    def onRemoteSiteSelected(self, index):
        settings = slicer.app.settings()
        selectedName = self.remoteSiteSelector.currentText if index >= 0 else ""
        settings.setValue("ModalityConverter/selectedRemoteSite", "" if selectedName == "Local" else selectedName)
        if self.remoteServerUrl:
            self.remoteServerUrl = self.remoteToken = self.remoteResources = None
            self.activeRemoteSite = None
            self.remoteUI.setConnected(False)
            self.remoteStatusLabel.setText("● Offline · Select a site and test the connection")
            self.remoteStatusLabel.setStyleSheet("color: #b36b00; font-weight: 600;")
            self.ui.deviceList.clear()
            self.initDeviceDropdown()
            self.populateDeviceDropdown()
            self.updateResourceUsage()

    def onManageRemoteSites(self):
        from ModalityConverterLib.UI.RemoteSitesDialog import RemoteSitesDialog
        dialog = RemoteSitesDialog(self.remoteSites, slicer.util.mainWindow())
        if not dialog.exec_():
            return
        self.remoteSites = dialog.sites
        self._saveRemoteSiteProfiles(self.remoteSites)
        selectedName = self.remoteSiteSelector.currentText
        self.remoteSiteSelector.blockSignals(True)
        self.remoteSiteSelector.clear()
        self.remoteSiteSelector.addItem("Local", None)
        for site in self.remoteSites:
            self.remoteSiteSelector.addItem(site["name"], site)
        match = self.remoteSiteSelector.findText(selectedName)
        if match <= 0 and self.remoteSites:
            match = 1
        self.remoteSiteSelector.setCurrentIndex(max(0, match))
        self.remoteSiteSelector.blockSignals(False)
        selectedProfile = self.remoteSiteSelector.currentText
        slicer.app.settings().setValue(
            "ModalityConverter/selectedRemoteSite", "" if selectedProfile == "Local" else selectedProfile)
        if self.remoteServerUrl:
            self.remoteServerUrl = self.remoteToken = self.remoteResources = None
            self.activeRemoteSite = None
            self.remoteUI.setConnected(False)
        self.remoteStatusLabel.setText("● Offline · Select a site and connect")
        self.remoteStatusLabel.setStyleSheet("color: #b36b00; font-weight: 600;")
        self.ui.deviceList.clear()
        self.initDeviceDropdown()
        self.populateDeviceDropdown()

    def _launchRemoteInference(self, context, inputPreprocessed=False):
        if self._resourceTimer:
            self._resourceTimer.setInterval(self.POLLING_UPLOADING_TIMER_SHORT)
        context["workerPath"] = os.path.join(os.path.dirname(__file__), "ModalityConverterLib", "remote_client_worker.py")
        self.runContext = context
        self.updateInfoLabel("Sending volume to remote server using {}...".format(context["device"]))
        self.currentRunToken = self.inferenceProcessManager.startRemote(
            self.logic.pythonExecutable(), context["workerPath"], context,
            self.remoteServerUrl, self.remoteToken, inputPreprocessed=inputPreprocessed,
            certificate=(self.activeRemoteSite.get("certificate", "") if self.activeRemoteSite else ""))

    def updateResourceUsage(self):
        if self.remoteServerUrl and self.remoteToken:
            try:
                import requests
                response = requests.get(self.remoteServerUrl + "/api/v1/resources",
                    headers={"Authorization": "Bearer " + self.remoteToken}, timeout=10,
                    verify=(self.activeRemoteSite.get("certificate") or True) if self.activeRemoteSite else True)
                response.raise_for_status()
                self.remoteResources = response.json()
            except Exception as exc:
                self._disconnectRemoteSite(str(exc))
                return
            else:
                if self.activeRemoteSite:
                    self.remoteStatusLabel.setText("● Online · {}".format(self.activeRemoteSite["name"]))
                    self.remoteStatusLabel.setStyleSheet("color: #16803c; font-weight: 600;")
        if self.remoteServerUrl:
            data = self.remoteResources or {}
            activeJobs = data.get("active_jobs", 0)
            cpu = data.get("worker_cpu_percent") if activeJobs else data.get("cpu_percent")
            cpuLabel = ("Remote inference CPU: {:.0f}% ({} job{})".format(
                cpu, activeJobs, "s" if activeJobs != 1 else "") if activeJobs else
                "Remote server CPU used: {:.0f}%".format(cpu) if cpu is not None else "Remote CPU used: N/D")
            self._setResourceBar("cpu", cpu, cpuLabel)
            total = data.get("ram_total")
            used = data.get("worker_ram_used") if activeJobs else data.get("ram_used")
            ramLabel = ("Remote inference RAM: {:.1f} GB".format(used / 1024**3) if activeJobs else
                "Remote RAM used: {:.1f} / {:.1f} GB".format(used / 1024**3, total / 1024**3)
                if total else "Remote RAM used: N/D")
            self._setResourceBar("ram", 100.0 * used / total if total and used is not None else None, ramLabel)
            gpu = next((g for g in data.get("gpus", [])
                        if "cuda:{}".format(g.get("cuda_index", g["index"])) == (self.selectedDeviceKey or "")), None)
            self._setResourceBar("gpu", gpu.get("utilization") if gpu else None,
                "Remote GPU {} ({}): {:.1f} / {:.1f} GB".format(
                    gpu["index"], gpu["name"], gpu["memory_used_mb"] / 1024,
                    gpu["memory_total_mb"] / 1024) if gpu else "Remote GPU used: N/D")
            return
        cpu = None
        ramUsed = None
        ramTotal = None
        try:
            import psutil
            cpu = psutil.cpu_percent()
            ram = psutil.virtual_memory()
            ramUsed = ram.used
            ramTotal = ram.total
        except Exception:
            if sys.platform.startswith("linux"):
                try:
                    with open("/proc/stat", "r") as statFile:
                        values = [int(value) for value in statFile.readline().split()[1:8]]
                    idle = values[3] + values[4]
                    total = sum(values)
                    if self._cpuSample:
                        oldIdle, oldTotal = self._cpuSample
                        delta = total - oldTotal
                        if delta > 0:
                            cpu = 100.0 * (delta - (idle - oldIdle)) / delta
                    self._cpuSample = (idle, total)
                    memInfo = {}
                    with open("/proc/meminfo", "r") as memFile:
                        for line in memFile:
                            name, value = line.split(":", 1)
                            memInfo[name] = int(value.strip().split()[0]) * 1024
                    ramTotal = memInfo.get("MemTotal")
                    available = memInfo.get("MemAvailable")
                    if ramTotal is not None and available is not None:
                        ramUsed = ramTotal - available
                except Exception:
                    pass

        self._setResourceBar("cpu", cpu)
        if ramUsed is not None and ramTotal:
            ramPercent = 100.0 * ramUsed / ramTotal
            self._setResourceBar("ram", ramPercent,
                                 "RAM used: {:.1f} GB / {:.1f} GB".format(
                                     ramUsed / (1024 ** 3), ramTotal / (1024 ** 3)))
        else:
            self._setResourceBar("ram", None, "RAM used: N/D")

        gpu = gpuUsed = gpuTotal = None
        device = self.selectedDeviceKey or "cpu"
        if device.startswith("cuda:"):
            try:
                import subprocess
                index = device.split(":", 1)[1]
                result = subprocess.run(["nvidia-smi", "--id=" + index,
                                         "--query-gpu=utilization.gpu,memory.used,memory.total",
                                         "--format=csv,noheader,nounits"],
                                        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                        universal_newlines=True, timeout=1)
                if result.returncode == 0:
                    values = [float(value.strip()) for value in result.stdout.strip().split(",")]
                    if len(values) == 3:
                        gpu, gpuUsed, gpuTotal = values
            except Exception:
                pass
        if gpu is not None and gpuTotal:
            self._setResourceBar("gpu", gpu,
                                 "GPU used: {:.1f} GB / {:.1f} GB".format(
                                     gpuUsed / 1024.0, gpuTotal / 1024.0))
        else:
            self._setResourceBar("gpu", None, "GPU used: N/D")

    def _setResourceBar(self, key, value, labelText=None):
        self.resourceUsageWidget.setResource(key, value, labelText)

    def setRunGuiBusy(self, busy):
        self._runGuiBusy = bool(busy)
        self.ui.applyButton.setText("  Stop" if busy else "  Run")
        self.ui.applyButton.setIcon(self.stopIcon if busy else self.runIcon)
        self.ui.applyButton.setStyleSheet(
            "background-color: rgba(210, 55, 55, 185);" if busy else
            "background-color: rgb(52, 206, 165);")
        self.ui.applyButton.setToolTip("Stop the current operation" if busy else "Run the algorithm.")
        self.ui.applyButton.setEnabled(bool(busy) or bool(self._parameterNode and self._parameterNode.inputVolume and self._parameterNode.outputVolume and (not self.maskRequiredForSelectedModel or self._parameterNode.maskVolume)))
        self.ui.sampleDataButton.setEnabled(not busy)
        if busy:
            self.progressBar.setValue(0)
            self.progressBar.setVisible(True)
            for name in ("inputSelector", "outputSelector", "maskSelector", "modelSelector", "deviceList"):
                getattr(self.ui, name).enabled = False
            if hasattr(self, "remoteConnectButton"):
                self.remoteConnectButton.setEnabled(False)

    def _startBrainPreprocessing(self, runId):
        """Run Slicer's native brain mask and N4 CLIs asynchronously before inference."""
        state = self.brainPreprocessing
        spec = state["runSpec"]
        inputVolume = spec["inputVolume"]
        if not inputVolume or not inputVolume.GetImageData():
            raise ValueError("Select a valid input volume")
        maskVolume = spec["maskVolume"]
        if maskVolume is None:
            maskVolume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", "MaskedInputVolume")
            state["generatedMask"] = maskVolume
            parameters = {"inputVolume": inputVolume.GetID(),
                          "outputROIMaskVolume": maskVolume.GetID(),
                          "fillValue": 0, "numberOfThreads": 4}
            self._runBrainCli(runId, "mask", slicer.modules.brainsroiauto, parameters)
        else:
            self._startN4Cli(runId, maskVolume)

    def _startN4Cli(self, runId, maskVolume):
        state = self.brainPreprocessing
        spec = state["runSpec"]
        correctedVolume = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLScalarVolumeNode", "ModalityConverterN4Input")
        state["correctedInput"] = correctedVolume
        parameters = {
            "inputImageName": spec["inputVolume"].GetID(),
            "maskImageName": maskVolume.GetID(),
            "outputImageName": correctedVolume.GetID(),
            "shrinkFactor": 2,
            "numberOfIterations": [50, 40, 30],
            "convergenceThreshold": 0.00001,
            "bsplineOrder": 3,
        }
        self._runBrainCli(runId, "n4", slicer.modules.n4itkbiasfieldcorrection, parameters)

    def _runBrainCli(self, runId, stage, module, parameters):
        cliNode = slicer.cli.run(module, None, parameters, wait_for_completion=False,
                                 delete_temporary_files=True, update_display=False)
        state = self.brainPreprocessing
        state["cliNodes"].append(cliNode)
        self.pendingCliNode = cliNode
        observerTag = cliNode.AddObserver(vtk.vtkCommand.ModifiedEvent,
            lambda caller, event, token=runId, step=stage: self._onBrainCliModified(token, step, caller))
        state["observers"][cliNode.GetID()] = observerTag

    def _onBrainCliModified(self, runId, stage, cliNode):
        state = self.brainPreprocessing
        if runId != self.activeRunId or not state:
            return
        cliId = cliNode.GetID()
        if cliId in state["handledCliIds"]:
            return
        statusText = cliNode.GetStatusString().lower()
        if "cancelling" in statusText:
            return
        status = cliNode.GetStatus()
        isCancelled = "cancel" in statusText
        isFailed = bool(status & cliNode.ErrorsMask)
        isCompleted = bool(status & cliNode.Completed)
        if not (isCancelled or isFailed or isCompleted):
            return
        state["handledCliIds"].add(cliId)
        observerTag = state["observers"].pop(cliId, None)
        if observerTag is not None:
            cliNode.RemoveObserver(observerTag)
        if self.pendingCliNode is cliNode:
            self.pendingCliNode = None
        if isCancelled or self.brainCancellationRequested:
            self.brainCancellationRequested = False
            self._cleanupBrainPreprocessing()
            self.activeRunId = None
            self.updateInfoLabel("Brain preprocessing cancelled.")
            self._restoreRunGui()
            return
        if isFailed:
            self._failBrainPreprocessing(runId, cliNode.GetErrorText() or cliNode.GetStatusString())
            return
        try:
            if stage == "mask":
                self.updateInfoLabel("Applying N4 bias field correction...")
                self._startN4Cli(runId, state["generatedMask"])
                return
            spec = state["runSpec"]
            maskVolume = spec["maskVolume"]
            if maskVolume is None:
                maskVolume = state["generatedMask"]
            context = self.logic.prepareInference(
                inputVolume=state["correctedInput"], outputVolume=spec["outputVolume"],
                maskVolume=maskVolume, showAllFiles=spec["showAllFiles"],
                selectedModelKey=spec["modelKey"],
                selectedModelModuleName=spec["moduleName"], device=spec["device"])
            context["inputVolume"] = spec["inputVolume"]  # preserve original output geometry
            if self.remoteServerUrl:
                self._launchRemoteInference(context, inputPreprocessed=True)
            else:
                self._launchInference(context, inputPreprocessed=True)
        except Exception as e:
            self._failBrainPreprocessing(runId, str(e))

    def _failBrainPreprocessing(self, runId, message):
        if runId != self.activeRunId:
            return
        if self.runContext:
            import shutil
            shutil.rmtree(self.runContext.get("workDir", ""), ignore_errors=True)
        self.runContext = None
        self._cleanupBrainPreprocessing()
        self.activeRunId = None
        self.updateInfoLabel("Brain preprocessing failed.")
        self._restoreRunGui()
        slicer.util.errorDisplay("Brain preprocessing failed: {}".format(message))

    def _cleanupBrainPreprocessing(self, preserveVolumes=False):
        state = self.brainPreprocessing
        self.pendingCliNode = None
        if state:
            for cliNode in state.get("cliNodes", []):
                observerTag = state.get("observers", {}).get(cliNode.GetID())
                if observerTag is not None:
                    cliNode.RemoveObserver(observerTag)
            if not preserveVolumes:
                for node in (state.get("generatedMask"), state.get("correctedInput")):
                    if node and slicer.mrmlScene.IsNodePresent(node):
                        slicer.mrmlScene.RemoveNode(node)
            for cliNode in state.get("cliNodes", []):
                if cliNode and slicer.mrmlScene.IsNodePresent(cliNode):
                    slicer.mrmlScene.RemoveNode(cliNode)
        self.brainPreprocessing = None

    def _launchInference(self, context, inputPreprocessed=False):
        self.runContext = context
        self.updateInfoLabel("Starting inference worker...")
        extraArguments = ["--input-preprocessed"] if inputPreprocessed else None
        self.currentRunToken = self.inferenceProcessManager.start(
            pythonExecutable=self.logic.pythonExecutable(),
            workerPath=context["workerPath"], inputPath=context["inputPath"],
            outputPath=context["outputPath"], modelKey=context["modelKey"],
            moduleName=context["moduleName"], device=context["device"],
            extensionRoot=context["extensionRoot"], modelCacheDir=context["modelCacheDir"],
            maskPath=context["maskPath"], showAllFiles=context["showAllFiles"],
            extraArguments=extraArguments)

    def onStopInference(self):
        if self.pendingCliNode:
            self.brainCancellationRequested = True
            self.ui.applyButton.enabled = False
            self.updateInfoLabel("Stopping brain preprocessing...")
            slicer.cli.cancel(self.pendingCliNode)
            return
        if self.inferenceProcessManager and self.inferenceProcessManager.isRunning:
            self.ui.applyButton.enabled = False
            self.updateInfoLabel("Stopping inference...")
            self.inferenceProcessManager.cancel()

    def onInferenceProgress(self, percent, message):
        if not self.currentRunToken:
            return
        
        if self._resourceTimer and self.remoteServerUrl:
            uploadingVolume = "uploading volume" in (message or "").lower()
            self._resourceTimer.setInterval(self.POLLING_UPLOADING_TIMER_SHORT if uploadingVolume else self.POLLING_UPLOADING_TIMER_LONG)
            
        self.progressBar.setValue(percent)
        self.updateInfoLabel(message or "Inference running ({:d}%)".format(percent))

    def onInferenceFinished(self, token, exitCode, stdout):
        if token != self.currentRunToken or not self.runContext:
            return
        try:
            import numpy as np

            output = np.load(self.runContext["outputPath"], allow_pickle=False)

            if output.ndim != 3 or output.size == 0:
                raise ValueError("Worker output is not a non-empty 3D volume")

            outputVolume = self.runContext["outputVolume"]
            slicer.util.updateVolumeFromArray(outputVolume, output)
            outputVolume.CopyOrientation(self.runContext["inputVolume"])

            if self.runContext.get("moduleName") == "CT2PETMoriModel":
                displayNode = outputVolume.GetDisplayNode()
                if displayNode:
                    colorNodeId = "vtkMRMLColorTableNodePET-Rainbow2"
                    colorNode = slicer.mrmlScene.GetNodeByID(colorNodeId)
                    if colorNode:
                        # Mori's post-processing returns SUV values on a 0–7 scale.
                        # Keep the PET palette mapped to that scale instead of
                        # allowing automatic window/level to reset it to grayscale.
                        lookupTable = colorNode.GetLookupTable()
                        if lookupTable:
                            lookupTable.SetRange(0.0, 7.0)
                            lookupTable.Build()
                            lookupTable.Modified()
                        displayNode.SetAndObserveColorNodeID(colorNodeId)
                        displayNode.AutoWindowLevelOff()
                        displayNode.SetWindow(7.0)
                        displayNode.SetLevel(3.5)

            previewNodes = {}
            if self.runContext["showAllFiles"]:
                previewDir = os.path.join(self.runContext["workDir"], "previews")
                if os.path.isdir(previewDir):
                    for filename in os.listdir(previewDir):
                        if not filename.endswith(".npy"):
                            continue
                        array = np.load(os.path.join(previewDir, filename), allow_pickle=False)
                        name = os.path.splitext(filename)[0]
                        previewNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", name)
                        slicer.util.updateVolumeFromArray(previewNode, array)
                        previewNode.CopyOrientation(self.runContext["inputVolume"])
                        previewNodes[name] = previewNode

            foreground = previewNodes.get("PreprocessedInputVolume")
            slicer.util.setSliceViewerLayers(background=outputVolume, foreground=foreground)
            slicer.util.resetSliceViews()
            self.updateInfoLabel("Inference completed.")
        except Exception as e:
            slicer.util.errorDisplay("Inference finished, but its output could not be loaded: {}".format(e))
            logging.exception("Failed to load inference output")
        finally:
            preservePreviews = bool(exitCode == 0 and self.runContext and self.runContext.get("showAllFiles"))
            self._finishRun(token, preservePreviews=preservePreviews)

    def onInferenceFailed(self, token, message):
        if token != self.currentRunToken:
            return
        
        slicer.util.errorDisplay("Inference failed:\n{}".format(message))
        logging.error("Inference worker failed: %s", message)
        self.updateInfoLabel("Inference failed.")
        self._finishRun(token)

    def onInferenceCancelled(self, token):
        if token != self.currentRunToken:
            return
        
        self.updateInfoLabel("Inference cancelled.")
        self._finishRun(token)

    def _finishRun(self, token, preservePreviews=False):
        if token != self.currentRunToken:
            return
        
        self._restoreRunGui()
        self.currentRunToken = None
        self.runContext = None
        self.inferenceProcessManager.cleanup()
        self._cleanupBrainPreprocessing(preserveVolumes=preservePreviews)
        self.activeRunId = None

    def _restoreRunGui(self):
        self._runGuiBusy = False
        if self._resourceTimer:
            self._resourceTimer.setInterval(self.POLLING_UPLOADING_TIMER_SHORT)
        self.ui.applyButton.setText("  Run")
        self.ui.applyButton.setIcon(self.runIcon)
        self.ui.applyButton.setStyleSheet("background-color: rgb(52, 206, 165);")
        self.ui.applyButton.setToolTip("Run the algorithm.")
        self.setMainButtonsState(True)
        
        if hasattr(self, "progressBar"):
            self.progressBar.setVisible(False)
            
        for name in ("inputSelector", "outputSelector", "maskSelector", "modelSelector", "deviceList"):
            widget = getattr(getattr(self, "ui", None), name, None)
            if widget:
                widget.enabled = True
                
        if hasattr(self, "remoteConnectButton"):
            self.remoteConnectButton.setEnabled(True)
            
        if hasattr(self, "ui"):
            self._checkCanApply()

#
# ModalityConverterLogic
#

class ModalityConverterLogic(ScriptedLoadableModuleLogic):
    def __init__(self) -> None:
        """Called when the logic class is instantiated. Can be used for initializing member variables."""
        ScriptedLoadableModuleLogic.__init__(self)
        self.model = None

    def getParameterNode(self):
        return ModalityConverterParameterNode(super().getParameterNode())

    def getModelInstance(self, selectedModelModuleName, selectedModelKey, basePackage="ModalityConverterLib.ModelsImpl", device="cpu"):
        """
        Dynamically loads and returns an instance of a model class based on the provided module name and model key.

        Parameters:
        - selectedModelModuleName (str): The name of the module containing the model class.
        - selectedModelKey (str): The key identifying the model in the model registry.
        - basePackage (str): The base package path where the model modules are located. Defaults to "ModalityConverterLib.ModelsImpl".

        Returns:
        - object: An instance of the model class corresponding to the provided key.

        Raises:
        - ImportError: If the specified module cannot be imported.
        - ValueError: If the model key is not found in the model registry.
        """

        import importlib
        from ModalityConverterLib.ModelBase import MODEL_REGISTRY

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

    def pythonExecutable(self):
        """Return Slicer's bundled standalone Python interpreter when available."""
        suffix = ".exe" if sys.platform.startswith("win") else ""
        candidates = []
        
        try:
            binaryDir = slicer.app.applicationDirPath
            if callable(binaryDir):
                binaryDir = binaryDir()
            candidates.append(os.path.join(str(binaryDir), "PythonSlicer" + suffix))
        except Exception:
            pass
        
        try:
            slicerHome = slicer.app.slicerHome
            if callable(slicerHome):
                slicerHome = slicerHome()
            candidates.append(os.path.join(str(slicerHome), "bin", "PythonSlicer" + suffix))
        except Exception:
            pass
        
        if os.path.basename(sys.executable).lower().startswith("pythonslicer"):
            candidates.append(sys.executable)
        for candidate in candidates:
            if candidate and os.path.isfile(candidate):
                return candidate
        raise FileNotFoundError("Could not locate Slicer's standalone PythonSlicer interpreter")

    def prepareInference(self, inputVolume, outputVolume, maskVolume, selectedModelKey,
                         selectedModelModuleName, showAllFiles=True, device="cpu"):
        import json
        import tempfile
        import numpy as np
        
        if not inputVolume or not outputVolume:
            raise ValueError("Select valid input and output scalar volumes")
        
        if not selectedModelKey or not selectedModelModuleName:
            raise ValueError("Select a valid inference model")
        
        if device not in ("cpu", "coreml") and not (device or "").startswith("cuda:"):
            raise ValueError("Select a supported inference device")
        
        if not inputVolume.GetImageData() or inputVolume.GetImageData().GetNumberOfPoints() == 0:
            raise ValueError("Input volume is empty")
        
        if not os.path.isfile(os.path.join(os.path.dirname(__file__), "ModalityConverterLib", "inference_worker.py")):
            raise FileNotFoundError("Inference worker script is missing")
        
        modelsDir = os.path.join(os.path.dirname(__file__), "Resources", "Models")
        with open(os.path.join(modelsDir, "metadata.json"), "r") as metadataFile:
            metadata = json.load(metadataFile)
            
        if selectedModelKey not in metadata or metadata[selectedModelKey].get("module_name") != selectedModelModuleName:
            raise ValueError("Selected model does not match the installed model metadata")
        
        if metadata[selectedModelKey].get("mask_required") and not maskVolume:
            raise ValueError("This model requires an ROI mask")
        
        if maskVolume:
            maskArray = slicer.util.arrayFromVolume(maskVolume)
            if maskArray.shape != slicer.util.arrayFromVolume(inputVolume).shape:
                raise ValueError("Input and mask volumes must have matching dimensions")
            
        workDir = tempfile.mkdtemp(prefix="modality-converter-")
        inputPath = os.path.join(workDir, "input.npy")
        outputPath = os.path.join(workDir, "output.npy")
        
        # NPY preserves the source scalar dtype and values without a compression pass.
        np.save(inputPath, np.asarray(slicer.util.arrayFromVolume(inputVolume)), allow_pickle=False)
        maskPath = None
        
        if maskVolume:
            maskPath = os.path.join(workDir, "mask.npy")
            np.save(maskPath, np.asarray(slicer.util.arrayFromVolume(maskVolume)), allow_pickle=False)
        return {
            "workDir": workDir, "inputPath": inputPath, "outputPath": outputPath,
            "maskPath": maskPath, "outputVolume": outputVolume, "inputVolume": inputVolume,
            "modelKey": selectedModelKey, "moduleName": selectedModelModuleName,
            "device": device, "showAllFiles": bool(showAllFiles),
            "workerPath": os.path.join(os.path.dirname(__file__), "ModalityConverterLib", "inference_worker.py"),
            "extensionRoot": os.path.dirname(os.path.abspath(__file__)),
            "modelCacheDir": os.path.join(os.path.expanduser("~"), ".SlicerModalityConverter", "Models"),
        }
