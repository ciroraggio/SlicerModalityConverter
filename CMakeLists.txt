cmake_minimum_required(VERSION 3.16.3...3.19.7 FATAL_ERROR)

project(ModalityConverter)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "https://github.com/ciroraggio/SlicerModalityConverter")
set(EXTENSION_CATEGORY "Image Synthesis")
set(EXTENSION_CONTRIBUTORS "Ciro Benito Raggio (Karlsruhe Institute of Technology, Germany), Paolo Zaffino (Magna Graecia University of Catanzaro, Italy), Maria Francesca Spadea (Karlsruhe Institute of Technology, Germany)")
set(EXTENSION_DESCRIPTION "SlicerModalityConverter is an open-source 3D Slicer extension designed for medical image-to-image (I2I) translation. The ModalityConverter module integrates multiple deep learning models trained for different kind of I2I translation, providing a user-friendly interface. Custom models can be easily integrated.")
set(EXTENSION_ICONURL "https://raw.githubusercontent.com/ciroraggio/SlicerModalityConverter/main/ModalityConverter.png")
set(EXTENSION_SCREENSHOTURLS "https://raw.githubusercontent.com/ciroraggio/SlicerModalityConverter/main/ModalityConverter/assets/ScreenshotResultExample.png")
set(EXTENSION_DEPENDS "NA") # Specified as a list or "NA" if no dependencies

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(ModalityConverter)
add_subdirectory(ModalityConverter)
## NEXT_MODULE

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})
