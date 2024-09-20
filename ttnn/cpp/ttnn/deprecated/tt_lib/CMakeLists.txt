
if(NOT WITH_PYTHON_BINDINGS)
    message(FATAL_ERROR "Remove include of tt_lib when WITH_PYTHON_BINDINGS flag is OFF")
endif()

set(TT_LIB_SRCS
    CACHE INTERNAL "tt_lib sources to reuse in ttnn build"
)
