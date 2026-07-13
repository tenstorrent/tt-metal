include(sources.cmake)

add_library(ttnn_op_experimental_quasar ${LIB_TYPE})
add_library(TTNN::Ops::Experimental::Quasar ALIAS ttnn_op_experimental_quasar)

tt_reuse_precompile_headers(ttnn_op_experimental_quasar TTNN::PCHFull)
# Enable unity builds for faster compilation.
TT_ENABLE_UNITY_BUILD(ttnn_op_experimental_quasar)

set_target_properties(
    ttnn_op_experimental_quasar
    PROPERTIES
        INTERFACE_HEADER_SETS_TO_VERIFY
            api
)

# Match every per-op kernel directory: kernels/, kernels_ng/, kernels_dfb/.
file(GLOB_RECURSE kernels */device/kernels*/*)

target_sources(
    ttnn_op_experimental_quasar
    PUBLIC
        FILE_SET api
        TYPE HEADERS
        BASE_DIRS ${FixmeOpAPIDir}
        FILES ${TTNN_OP_EXPERIMENTAL_QUASAR_API_HEADERS}
        FILE_SET kernels
        TYPE HEADERS
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
        FILES ${kernels}
    PRIVATE
        ${TTNN_OP_EXPERIMENTAL_QUASAR_SRCS}
)

target_link_libraries(ttnn_op_experimental_quasar PRIVATE TT::Metalium PUBLIC TTNN::Core)

install(
    TARGETS
        ttnn_op_experimental_quasar
    FILE_SET
    kernels
        DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium/ttnn/cpp/ttnn/operations/experimental/quasar
        COMPONENT ttnn-runtime
)

install(TARGETS ttnn_op_experimental_quasar FILE_SET api COMPONENT ttnn-dev LIBRARY COMPONENT tar)
