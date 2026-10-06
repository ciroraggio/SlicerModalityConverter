import argparse
import os
import sys
import traceback
import types


def report(percent, message):
    print("MODALITY_CONVERTER_PROGRESS:{}:{}".format(int(percent), message), flush=True)


class Volume:
    def __init__(self, name="Volume"):
        self.name = name
        self.array = None
        self.displayNode = _DisplayNode()

    def GetID(self):
        return self.name

    def GetDisplayNode(self):
        return self.displayNode

    def CopyOrientation(self, other):
        pass


class _DisplayNode:
    def SetWindow(self, value): pass
    def SetLevel(self, value): pass
    def RemoveAllViewIDs(self): pass
    def SetVisibility(self, value): pass


class _Scene:
    def __init__(self):
        self.nodes = {}

    def AddNewNodeByClass(self, className, name):
        node = Volume(name)
        self.nodes[name] = node
        return node

    def RemoveNode(self, node):
        self.nodes.pop(node.name, None)


def install_slicer_compat(inputArray, maskArray):
    """Provide only the narrow, array-oriented surface used by model classes."""
    fake = types.ModuleType("slicer")
    fake.vtkMRMLScalarVolumeNode = Volume
    fake.mrmlScene = _Scene()
    fake.app = types.SimpleNamespace(processEvents=lambda: None, majorVersion=5, minorVersion=12)
    util = types.SimpleNamespace()
    util.arrayFromVolume = lambda node: node.array
    util.updateVolumeFromArray = lambda node, array: setattr(node, "array", array)
    util.setSliceViewerLayers = lambda **kwargs: None
    util.resetSliceViews = lambda: None
    util.getNode = lambda name: fake.mrmlScene.nodes[name]
    util.errorDisplay = lambda message: (_ for _ in ()).throw(RuntimeError(message))
    util.warningDisplay = lambda message: print("WARNING: " + message, file=sys.stderr)
    fake.util = util
    fake.cli = types.SimpleNamespace(run=lambda *args, **kwargs: (_ for _ in ()).throw(
        RuntimeError("Slicer CLI operations are unavailable in the inference worker")))
    fake.modules = types.SimpleNamespace()
    sys.modules["slicer"] = fake
    try:
        import vtk
        vtk.vtkMRMLScalarVolumeNode = Volume
    except ImportError:
        vtk = types.ModuleType("vtk")
        vtk.vtkMRMLScalarVolumeNode = Volume
        sys.modules["vtk"] = vtk

    inputNode = Volume("InputVolume")
    inputNode.array = inputArray
    maskNode = None
    if maskArray is not None:
        maskNode = Volume("MaskedInputVolume")
        maskNode.array = maskArray
        fake.mrmlScene.nodes[maskNode.name] = maskNode
    return fake, inputNode, maskNode


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run SlicerModalityConverter inference")
    parser.add_argument("--model", required=True, help="Registered model key")
    parser.add_argument("--module", required=True, help="Registered model Python module")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--mask")
    parser.add_argument("--show-previews", choices=("0", "1"), default="1")
    parser.add_argument("--extension-root", required=True)
    parser.add_argument("--input-preprocessed", action="store_true",
                        help="Input has already received the Slicer N4 preprocessing step")
    args = parser.parse_args(argv)
    try:
        import numpy as np
        if args.device != "cpu" and not (args.device.startswith("cuda:") or args.device == "coreml"):
            raise ValueError("Unsupported inference device: " + args.device)
        inputArray = np.load(args.input, allow_pickle=False)
        maskArray = None
        if args.mask:
            maskArray = np.load(args.mask, allow_pickle=False)
        if inputArray.ndim != 3 or inputArray.size == 0:
            raise ValueError("Input must be a non-empty 3D scalar volume")

        sys.path.insert(0, args.extension_root)
        slicer, inputNode, maskNode = install_slicer_compat(inputArray, maskArray)
        inputNode.modalityConverterBiasCorrected = args.input_preprocessed
        outputNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", "OutputVolume")

        report(2, "Loading model on {}".format(args.device))
        import importlib
        from ModalityConverterLib.ModelBase import MODEL_REGISTRY
        importlib.import_module("ModalityConverterLib.ModelsImpl." + args.module)
        if args.model not in MODEL_REGISTRY:
            raise ValueError("Model key '{}' is not registered by {}".format(args.model, args.module))
        model = MODEL_REGISTRY[args.model](args.model, args.device)
        model.progressCallback = lambda percent, message: report(15 + int(percent * 0.8), message)
        model.loadModel()
        report(15, "Model loaded; preprocessing and inference started")
        model.runInference(inputVolume=inputNode, inputMask=maskNode,
                           outputVolume=outputNode,
                           showAllFiles=args.show_previews == "1")
        if outputNode.array is None:
            raise RuntimeError("Model completed without producing an output volume")

        outputPayload = {"array": np.asarray(outputNode.array)}
        previewDir = os.path.join(os.path.dirname(args.output), "previews")
        if args.show_previews == "1":
            os.makedirs(previewDir, exist_ok=True)
            for name, node in slicer.mrmlScene.nodes.items():
                if node is not outputNode and node.array is not None:
                    safeName = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name)
                    previewPath = os.path.join(previewDir, safeName + ".npy")
                    np.save(previewPath, np.asarray(node.array), allow_pickle=False)
        np.save(args.output, outputPayload["array"], allow_pickle=False)
        report(100, "Inference completed")
        return 0
    except Exception as exc:
        print("MODALITY_CONVERTER_ERROR:" + str(exc), file=sys.stderr, flush=True)
        traceback.print_exc(file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
