#-----------------------------------------------------------------------------
set(MODULE_NAME SyntheticImageQualityAssurance)

#-----------------------------------------------------------------------------
set(MODULE_PYTHON_SCRIPTS
  ${MODULE_NAME}.py
  ${MODULE_NAME}Lib/ModelBase.py
  ${MODULE_NAME}Lib/ModelsImpl/MAESliceQAModel.py
  ${MODULE_NAME}Lib/UI/HelpDialog.py
  ${MODULE_NAME}Lib/UI/utils.py
  ${MODULE_NAME}Lib/Utils/customTransformsUtils.py
  ${MODULE_NAME}Lib/Utils/modelLoadUtils.py
  )

set(MODULE_PYTHON_RESOURCES
  Resources/Icons/${MODULE_NAME}.png
  Resources/UI/${MODULE_NAME}.ui
  Resources/Icons/book.png
  Resources/Models/metadata.json
  )

#-----------------------------------------------------------------------------
slicerMacroBuildScriptedModule(
  NAME ${MODULE_NAME}
  SCRIPTS ${MODULE_PYTHON_SCRIPTS}
  RESOURCES ${MODULE_PYTHON_RESOURCES}
  WITH_GENERIC_TESTS
  )

#-----------------------------------------------------------------------------
if(BUILD_TESTING)

  # Register the unittest subclass in the main script as a ctest.
  # Note that the test will also be available at runtime.
  slicer_add_python_unittest(SCRIPT ${MODULE_NAME}.py)

  # Additional build-time testing
  add_subdirectory(Testing)
endif()
