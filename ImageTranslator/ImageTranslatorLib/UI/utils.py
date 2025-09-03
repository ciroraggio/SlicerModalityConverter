CONTRIBUTORS = ["Ciro Benito Raggio (Karlsruhe Institute of Technology, Germany)", "Paolo Zaffino (Magna Graecia University of Catanzaro, Italy)", "Maria Francesca Spadea (Karlsruhe Institute of Technology, Germany)"]

HELP_TEXT = """
<b>Description</b>
<br/>
I2IHub is an open-source 3D Slicer module designed for medical image-to-image (I2I) translation. The ImageTranslator module integrates multiple deep learning models trained for different kind of I2I translation (MRI-to-CT, CBCT-to-CT), providing a user-friendly interface.
<br/>
<br/>
<b>Key Features</b>
<ul>
    <li>Support for multiple pre-trained deep learning models</li>
    <li>GPU acceleration support for faster processing</li>
    <li>Easy custom models integration for advanced users</li>
</ul>
<br/>
<b>Usage</b>
<ul>
    <li>Select an input image</li>
    <li>Choose a pre-trained model from the dropdown menu</li>
    <li>Optionally provide a mask to focus the translation on specific regions</li>
    <li>Click "Apply" to generate the synthetic image</li>
</ul>
<br/>
<b>More info</b>
<ul>
    <li>View the source code on GitHub: <a href="https://github.com/ciroraggio/SlicerI2IHub">https://github.com/ciroraggio/SlicerI2IHub</a></li>
    <li><a href="https://github.com/ciroraggio/SlicerI2IHub/blob/main/README.md#-how-to-integrate-a-custom-model-into-the-imagetranslator-module">How to integrate your models in I2IHub</a></li>
</ul>
<br/>
<b>How to cite</b>
<br/> 
Please cite the relevant publication when using models integrated in this module. Each model's description includes its corresponding citation information.
<br/>
The I2IHub 3D Slicer module should be cited as follows:<br/>
<cite>
Raggio C.B., Zaffino P., Spadea M.F., Slicer-I2IHub: An Open-Source  Extension for Medical Image-to-Image Translation, 2025, https://github.com/ciroraggio/SlicerI2IHub .
</cite>
"""

PRINT_MODULE_SUFFIX = "ImageTranslator -"

def updateButtonStyle(button, style):
    button.setStyleSheet(style)
    
