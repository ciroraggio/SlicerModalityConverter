CONTRIBUTORS = ["Ciro Benito Raggio (Karlsruhe Institute of Technology, Germany), Paolo Zaffino (Magna Graecia University of Catanzaro, Italy), Maria Francesca Spadea (Karlsruhe Institute of Technology, Germany)"]

HELP_TEXT = """
    <b>Description</b>
    <br/>
    I2IHub is a comprehensive and open-source 3D Slicer module designed for medical image-to-image (I2I) translation. The ImageTranslator module integrates multiple deep learning models trained for different kind of I2I translation (MRI-to-CT, CBCT-to-CT), providing a user-friendly interface.
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
    <ol>
        <li>Select an input image</li>
        <li>Choose a pre-trained model from the dropdown menu</li>
        <li>Optionally provide a mask to focus the translation on specific regions</li>
        <li>Click "Apply" to generate the synthetic image</li>
    </ol>
    <br/>
    <b>More info</b>
    <ul>
        <li>View the source code on GitHub: <a href="https://github.com/ciroraggio/I2IHub">https://github.com/ciroraggio/I2IHub</a></li>
        <li>How to integrate your models in I2IHub: <a href="https://github.com/ciroraggio/I2IHub">https://github.com/ciroraggio/I2IHub</a></li>
    </ul>
    <br/>
    <p><b>How to cite</b></p>
    <br/> 
    Please cite the relevant publication when using models integrated in this module. Each model's description includes its corresponding citation information.
    <br/>
    For the main software module, please cite:<br/>
    ...
"""

def updateButtonStyle(button, style):
    button.setStyleSheet(style)
