add_library(ttnn_deprecated INTERFACE)

set_target_properties(
    ttnn_deprecated
    PROPERTIES
        INTERFACE_HEADER_SETS_TO_VERIFY
            api
)

# Globbing non-build files is acceptable for now because devs don't generate packages.
file(GLOB_RECURSE kernels tt_dnn/kernels/*)
target_sources(
    ttnn_deprecated
    INTERFACE
        FILE_SET api
        TYPE HEADERS
        BASE_DIRS ${FixmeOpAPIDir}
        FILES
        FILE_SET kernels
        TYPE HEADERS
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
        FILES ${kernels}
)

install(
    TARGETS
        ttnn_deprecated
    FILE_SET
    kernels
        DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium/ttnn/cpp/ttnn/deprecated
        COMPONENT ttnn-runtime
)
