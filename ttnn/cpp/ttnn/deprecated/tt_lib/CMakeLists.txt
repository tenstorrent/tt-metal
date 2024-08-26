
if(NOT WITH_PYTHON_BINDINGS)
    message(FATAL_ERROR "Remove include of tt_lib when WITH_PYTHON_BINDINGS flag is OFF")
endif()

set(TT_LIB_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/tt_lib_bindings.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/type_caster.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/tt_lib_bindings_tensor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/tt_lib_bindings_tensor_pytensor.cpp

    CACHE INTERNAL "tt_lib sources to reuse in ttnn build"
)
