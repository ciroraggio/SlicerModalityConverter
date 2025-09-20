# SlicerModalityConverter - Slicer 5.8.1 Compatible

[![3D Slicer](https://img.shields.io/badge/3D%20Slicer-5.8.1%20Compatible-blue)](https://www.slicer.org/)
[![Python](https://img.shields.io/badge/Python-3.8%2B-green)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Medical Imaging](https://img.shields.io/badge/Medical%20Imaging-AI%20Translation-purple)](https://github.com/soumen02/SlicerModalityConverter)

> **🔧 Refactored for Slicer 5.8.1 Compatibility** - This fork has been specifically updated to work with 3D Slicer 5.8.1, addressing compatibility issues with the original version that required Slicer 5.9.0+.

- [SlicerModalityConverter - Slicer 5.8.1 Compatible](#slicermodalityconverter---slicer-581-compatible)
  - [Installation](#installation)
  - [ModalityConverter](#modalityconverter)
    - [Key Features](#key-features)
    - [How to Use](#how-to-use)
    - [Example (with video)](#example-with-video)
    - [How to Integrate a Custom Model](#how-to-integrate-a-custom-model)
    - [Summary of Requirements](#summary-of-requirements)
  - [Slicer 5.8.1 Compatibility](#slicer-581-compatibility)
  - [How to Contribute](#how-to-contribute)
  - [How to Cite](#how-to-cite)

**SlicerModalityConverter** is an open-source 3D Slicer extension designed for medical image-to-image (I2I) translation. This fork has been specifically refactored to ensure compatibility with **3D Slicer 5.8.1**, making it accessible to users who haven't upgraded to the latest Slicer version.

The ModalityConverter module integrates multiple deep learning models trained for different kinds of I2I translation (MRI-to-CT, CBCT-to-CT), providing a user-friendly interface for medical image synthesis and AI-powered modality conversion.

## Installation

### 🎯 **For Slicer 5.8.1 Users:**

This **SlicerModalityConverter fork** is specifically designed for **3D Slicer 5.8.1** compatibility. The original version requires Slicer 5.9.0+, but this fork works seamlessly with the older version.

### 📦 **Installation Methods:**

1. **Direct Installation (Recommended):**
   - Download this fork: `https://github.com/soumen02/SlicerModalityConverter`
   - Install via **Extension Manager** in 3D Slicer 5.8.1
   - Or build from source using CMake

2. **From Source:**
   ```bash
   git clone https://github.com/soumen02/SlicerModalityConverter.git
   cd SlicerModalityConverter
   # Follow Slicer extension building instructions
   ```

### 🔍 **Search Keywords:**
- `3D Slicer 5.8.1 ModalityConverter`
- `SlicerModalityConverter backward compatibility`
- `Medical image translation Slicer 5.8.1`
- `MRI to CT conversion Slicer extension`
- `Image synthesis Slicer 5.8.1`

Here is a short [video tutorial](https://youtu.be/CBkKGilpO1w?feature=shared) on how to install extensions.

## ModalityConverter

### Key Features

- Support for multiple pre-trained deep learning models
- GPU acceleration support for faster processing
- Easy custom models integration for advanced users

### How to Use

- Select an input image
- Choose a pre-trained model from the dropdown menu. Selecting each model will display detailed information on the translation modality, specific processing and inference output
- Optionally provide a binary mask to focus the translation on specific regions
- Click "Run" to generate the synthetic image

This extension is intended for research purposes only. If a model is applied to an input image of the wrong type (i.e. using a CT or CBCT instead of an MRI for an MRI-to-sCT model), the output will be wrong or unpredictable.

### Example (with video)

1. Click **Download sample** to open the *Sample Data* module.  
2. Download the **MRHead** volume.  
3. In the *ModalityConverter* module, select the MRHead volume as **Input volume**.  
4. From the **Model** list, choose the desired model (e.g., `[Brain] FedSynthCT MRI-T1w Li Model`).  
5. (Optional) Check **Preview volumes** if you want to visualize intermediate volumes generated during processing.  
6. Choose the **Output volume**.  
7. In the **Advanced** section, select the device from the **Device** list. A GPU device provides faster inference.  

At this point, the interface should look like this:  
<center>
<img src="https://raw.githubusercontent.com/ciroraggio/SlicerModalityConverter/main/ModalityConverter/assets/ScreenshotUI.png" />
</center>

8. Click **Run** to start the inference. The resulting volume will appear once the process is complete:  
<center>
<img src="https://raw.githubusercontent.com/ciroraggio/SlicerModalityConverter/main/ModalityConverter/assets/ScreenshotResultExample.png" />
</center>

**Full example**


https://github.com/user-attachments/assets/c4a71794-36d1-40d6-91c9-47f56983d56c



### How to Integrate a Custom Model

To add your own model to the **ModalityConverter** module of the ModalityConverter Slicer extension, follow these 3 steps. The integration is designed to be modular and automatic once the proper structure is respected.

🧠 **Step 1 — Implement a New Model Class**

Create a Python class in the directory:

```
.../ModalityConverter/ModalityConverter/ModalityConverterLib/ModelsImpl/
```

Your class must inherit from the `BaseModel` abstract class and implement the required methods. Here is the base structure:

```python
from ModalityConverterLib.ModelBase import BaseModel, register_model

@register_model("your_unique_model_key")
class YourModelClass(BaseModel):
    def __init__(self, modelKey: str, device: str = "cpu"):
        super().__init__(modelKey, device)

    def _loadModelFromPath(self, modelPath):
        # Load the model using your framework (e.g. torch, onnx, etc.)
        pass

    def runInference(self, inputVolume, outputVolume, inputMask=None, showAllFiles=True):
        # Define the inference procedure
        pass
```

- The `@register_model("your_unique_model_key")` decorator is **mandatory** and must match the key used in `model_metadata.json`.
- Implement `_loadModelFromPath(modelPath)` to define how your model is loaded (e.g. `torch.load`, `onnxruntime`, etc.).
- Implement `runInference(...)` to define how the model performs inference and writes results to the `outputVolume`.

You can also optionally implement `preprocess(...)` and helper methods for preprocessing input volumes, applying transforms, etc.

For a full example, see the classes `FedSynthBrainBaseModel` and `FedSynthBrainLiModel` provided in the source tree.

---

🏷️ **Step 2 — Add Your Model Entry to the Metadata File**

Open the file:

```
.../ModalityConverter/ModalityConverter/Resources/Models/model_metadata.json
```

Add a new entry in the following format:

```json
"your_unique_model_key": {
  "url": "https://example.com/path/to/your_model_file.onnx",
  "display_name": "Your Model Display Name",
  "description": "A detailed HTML-formatted description of your model. <b>Include citations, inputs, and outputs.</b>",
  "module_name": "YourModelClass" 
}
```

- `your_unique_model_key`: must match the string used in the `@register_model(...)` decorator and the model file name.
- `url`: direct link to download the model file (e.g., `.pth`, `.onnx`, etc.).
- `module_name`: must match the name of the class and the module.py you've defined (e.g. module: `YourModelClass.py`, class name: `YourModelClass`).
- `description`: supports HTML tags to format citations, inputs, and outputs. E.g., use `<b>`, `<cite>`, `<br/>`. The following template can be used:

```html
[MODEL_DESCRIPTION]<br><b>Input</b>: [MODALITY]<br><b>Preprocess</b>: [PRE_PROCESSING_DESCRIPTION]<br><b>Output</b>: [SYNTHETIC_OUTPUT_MODALITY] [OUTPUT_DIMENSION].<br><b>How to cite:</b><br>If you use this model, please cite:<br><cite>[CITATION]</cite>
```

---

📦 **Step 3 — Name and Upload Your Model File**

Save your model file (e.g. `your_unique_model_key.onnx`) and host it at the `url` specified in the JSON. The model filename **must** start with the same key used in the decorator and the metadata (e.g., `your_unique_model_key.onnx` or `your_unique_model_key.zip`).

Place the model in:

```
.../ModalityConverter/ModalityConverter/Resources/Models/
```

> If the model is not already present locally, the system will **automatically download** it from the provided URL when selected in the GUI.

---

**Full Example**

Here is a basic example to get started:

1. Implement a new model class

   a. Go to  `.../ModalityConverter/ModalityConverter/ModalityConverterLib/ModelsImpl`

   b. Create a python module: `ExampleModel.py`

   c. Create a new custom model class:

    ```python
    from ModalityConverterLib.ModelBase import BaseModel, register_model
            
    @register_model("a_model_unique_key")
    class ExampleModel(BaseModel):
        def _loadModelFromPath(self, modelPath):
            # Upload the model from the OS as you prefer
            model = ...
            return model
                
        def runInference(self, inputVolume, outputVolume, inputMask=None, showAllFiles=True):
            # Customize your preprocess, run inference and show the output
            ...
    ```

2. Update Metadata File at `.../ModalityConverter/ModalityConverter/Resources/Models/metadata.json`

    ```json
        {
            "...": {
                "..."
            },
            "a_model_unique_key": {
                "url": "https://example.com/examplemodel.ext",
                "display_name": "ExampleModel",
                "description": "ExampleModel description",
                "module_name": "ExampleModel"
            }
        }
3. Export the pretrained model file (`a_model_unique_key.extension`) and host it at the `url` specified in the JSON. The model filename **must** start with the same key used in the decorator and the metadata (`a_model_unique_key`).

4. Reload the module and enjoy your model! Once your model class is implemented and the metadata updated:

- It will appear automatically in the dropdown menu of the module UI.
- It will be downloaded and loaded dynamically as needed.
- Your inference and preprocessing logic will run when selected.

<center>
    <img src="https://raw.githubusercontent.com/ciroraggio/SlicerModalityConverter/main/ModalityConverter/assets/ExampleModelIntegration.png" />
</center>

---

### Summary of Requirements

| Requirement         | Description                                                                      |
| ------------------- | -------------------------------------------------------------------------------- |
| Class location      | `ModalityConverterLib/ModelsImpl/`                                                 |
| Required methods    | `_loadModelFromPath(...)`, `runInference(...)`                                   |
| Model key decorator | `@register_model("your_model_key")`                                              |
| Metadata file       | Add an entry to `model_metadata.json`                                            |
| Model file naming   | Must start with the same `your_model_key` used in the decorator and JSON         |
| Download support    | Model file is auto-downloaded from the provided URL if it is not present locally |

---

## Slicer 5.8.1 Compatibility

This fork has been specifically refactored to ensure compatibility with **3D Slicer 5.8.1**. The following changes were made to address compatibility issues:

### 🔧 **Key Refactoring Changes:**

- **Replaced `parameterNodeWrapper` decorator** with manual parameter node implementation
- **Fixed observer pattern issues** that caused warnings in Slicer 5.8.1
- **Updated dependency installation** to handle missing GPU packages gracefully
- **Improved UI element connections** for better parameter synchronization
- **Enhanced error handling** for cross-version compatibility

### 🎯 **Why This Fork?**

- **Original version requires Slicer 5.9.0+** - Many users still use Slicer 5.8.1
- **Backward compatibility** - Works seamlessly with older Slicer installations
- **Same functionality** - All features preserved from the original
- **Better error handling** - More robust dependency management

### 📋 **System Requirements:**

- **3D Slicer 5.8.1** (tested and verified)
- **Python 3.8+** (included with Slicer)
- **Optional GPU support** (CUDA-compatible GPU recommended for faster processing)

### 🚀 **Installation for Slicer 5.8.1:**

1. **Clone this fork** instead of the original repository
2. **Install via Extension Manager** in Slicer 5.8.1
3. **Or build from source** using the provided CMakeLists.txt

---

## How to Contribute

Integrating new models for different modalities is encouraged!

Once you have integrated and tested your custom model locally, simply create a pull request in the [original repository](https://github.com/ciroraggio/SlicerModalityConverter/) to request integration of your model into the 3D Slicer extension.

If you require any further information or have any queries, please send an email to: <email>ciro.raggio@kit.edu</email>.

## How to Cite

Please cite the relevant publication when using models integrated in this module. Each model's description includes its corresponding citation information.

The ModalityConverter 3D Slicer module should be cited as follows:

<cite>
Raggio C.B., Zaffino P., Spadea M.F., SlicerModalityConverter: An Open-Source 3D Slicer Extension for Medical Image-to-Image Translation, 2025, https://github.com/ciroraggio/SlicerModalityConverter .
</cite>

---

## 🔍 **SEO Keywords & Search Terms**

This repository is optimized for the following search terms to help users find the Slicer 5.8.1 compatible version:

- **3D Slicer 5.8.1 ModalityConverter**
- **SlicerModalityConverter backward compatibility**
- **Medical image translation Slicer 5.8.1**
- **MRI to CT conversion Slicer extension**
- **Image synthesis Slicer 5.8.1**
- **Deep learning medical imaging Slicer**
- **AI image translation 3D Slicer**
- **Modality conversion Slicer 5.8.1**
- **Slicer extension compatibility fix**
- **Medical AI Slicer 5.8.1**

---

## 📊 **Repository Stats**

- **Original Repository:** [ciroraggio/SlicerModalityConverter](https://github.com/ciroraggio/SlicerModalityConverter)
- **This Fork:** [soumen02/SlicerModalityConverter](https://github.com/soumen02/SlicerModalityConverter)
- **Slicer Version:** 5.8.1 Compatible
- **Python Version:** 3.8+
- **License:** MIT
- **Category:** Medical Imaging, AI, Deep Learning
