import os

CONTRIBUTORS = ["Ciro Benito Raggio (Karlsruhe Institute of Technology, Germany)", "Paolo Zaffino (Magna Graecia University of Catanzaro, Italy)", "Maria Francesca Spadea (Karlsruhe Institute of Technology, Germany)"]
MODULE_ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
LOGO_PATH = os.path.join(MODULE_ROOT_DIR, 'ImageToImageHub.png')
HELP_TEXT = f"""
<center>
    <img src="file://{LOGO_PATH}" width=320 height=180>
</center>
<br/>
<b>Description</b>
<br/>
ImageToImageHub is an open-source 3D Slicer module designed for medical image-to-image (I2I) translation. The ImageTranslator module integrates multiple deep learning models trained for different kind of I2I translation (MRI-to-CT, CBCT-to-CT), providing a user-friendly interface.
<br/>
<br/>
<b>Key Features</b>
<ul>
    <li>Support for multiple pre-trained deep learning models</li>
    <li>GPU acceleration support for faster processing</li>
    <li>Easy custom models integration for advanced users</li>
</ul>
<br/>
<b>How to Use</b>
<ul>
    <li>Select an input image</li>
    <li>Choose a pre-trained model from the dropdown menu. Selecting each model will display detailed information on the translation modality, specific processing and inference output</li>
    <li>Optionally provide a binary mask to focus the translation on specific regions</li>
    <li>Click "Run" to generate the synthetic image</li>
</ul>
<br/>
<b>More info</b>
<ul>
    <li>View the source code on GitHub: <a href="https://github.com/ciroraggio/SlicerImageToImageHub">https://github.com/ciroraggio/SlicerImageToImageHub</a></li>
    <li><a href="https://github.com/ciroraggio/SlicerImageToImageHub/blob/main/README.md#-how-to-integrate-a-custom-model-into-the-imagetranslator-module">How to integrate your models in ImageToImageHub</a></li>
</ul>
<br/>
<b>How to cite</b>
<br/> 
Please cite the relevant publication when using models integrated in this module. Each model's description includes its corresponding citation information.
<br/>
The ImageToImageHub 3D Slicer module should be cited as follows:<br/>
<cite>
Raggio C.B., Zaffino P., Spadea M.F., Slicer-ImageToImageHub: An Open-Source  Extension for Medical Image-to-Image Translation, 2025, https://github.com/ciroraggio/SlicerImageToImageHub .
</cite>
"""

PRINT_MODULE_SUFFIX = "ImageTranslator -"

def updateButtonStyle(button, style):
    button.setStyleSheet(style)
    
