
if(NOT WITH_PYTHON_BINDINGS)
    message(FATAL_ERROR "Remove include of tt_lib when WITH_PYTHON_BINDINGS flag is OFF")
endif()

set(TT_LIB_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/type_caster.cpp

    CACHE INTERNAL "tt_lib sources to reuse in ttnn build"
)
