add_library(ttnn_op_helper INTERFACE)
add_library(TTNN::Ops::Helper ALIAS ttnn_op_helper)

target_sources(
    ttnn_op_helper
    INTERFACE
        FILE_SET kernels
        TYPE HEADERS
        BASE_DIRS ${FixmeOpAPIDir}
        FILES pad_tile.hpp
)

install(
    TARGETS
        ttnn_op_helper
    FILE_SET
    kernels
        DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium/ttnn/cpp/ttnn/operations/kernel_helper_functions
        COMPONENT ttnn-runtime
)
