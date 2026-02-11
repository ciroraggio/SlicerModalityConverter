import os

CONTRIBUTORS = ["Ciro Benito Raggio (Karlsruhe Institute of Technology, Germany)", "Paolo Zaffino (Magna Graecia University of Catanzaro, Italy)", "Maria Francesca Spadea (Karlsruhe Institute of Technology, Germany)"]
MODULE_ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
LOGO_PATH = os.path.join(MODULE_ROOT_DIR, 'ModalityConverter.png')
HELP_TEXT = f"""
<center>
    <img src="file://{LOGO_PATH}" width=320 height=180>
</center>
<br/>
<b>Description</b>
<br/>
SlicerModalityConverter is an open-source 3D Slicer extension designed for medical image-to-image (I2I) translation. The ModalityConverter module integrates multiple deep learning models trained for different kind of I2I translation (MRI-to-CT, CBCT-to-CT), providing a user-friendly interface.
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
    <li>View the source code on GitHub: <a href="https://github.com/ciroraggio/SlicerModalityConverter">https://github.com/ciroraggio/SlicerModalityConverter</a></li>
    <li><a href="https://github.com/ciroraggio/SlicerModalityConverter/blob/main/README.md#how-to-integrate-a-custom-model">How to integrate your models in ModalityConverter</a></li>
</ul>
<br/>
<b>How to cite</b>
<br/> 
Please cite the relevant publication when using models integrated in this module. Each model's description includes its corresponding citation information.
<br/>
The SlicerModalityConverter extension should be cited as follows:<br/>
<cite>
Raggio C.B., Zaffino P., Spadea M.F., SlicerModalityConverter: An Open-Source 3D Slicer Extension for Medical Image-to-Image Translation, 2025, https://github.com/ciroraggio/SlicerModalityConverter .
</cite>
"""

PRINT_MODULE_SUFFIX = "ModalityConverter -"

def updateButtonStyle(button, style):
    button.setStyleSheet(style)
    
