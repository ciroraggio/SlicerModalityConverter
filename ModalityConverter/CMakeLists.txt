#-----------------------------------------------------------------------------
set(MODULE_NAME ModalityConverter)

#-----------------------------------------------------------------------------
set(MODULE_PYTHON_SCRIPTS
  ${MODULE_NAME}.py
  ${MODULE_NAME}Lib/ModelBase.py
  ${MODULE_NAME}Lib/ModelsImpl/FedSynthBrainBaseModel.py
  ${MODULE_NAME}Lib/ModelsImpl/FedSynthBrainFuModel.py
  ${MODULE_NAME}Lib/ModelsImpl/FedSynthBrainLiModel.py
  ${MODULE_NAME}Lib/ModelsImpl/FedSynthBrainSpadeaModel.py
  ${MODULE_NAME}Lib/ModelsImpl/FedSynthBrainModelsUtils.py
  ${MODULE_NAME}Lib/UI/HelpDialog.py
  ${MODULE_NAME}Lib/UI/utils.py
  )

set(MODULE_PYTHON_RESOURCES
  Resources/Icons/${MODULE_NAME}.png
  Resources/UI/${MODULE_NAME}.ui
  Resources/Icons/book.png
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
