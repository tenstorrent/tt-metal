include(sources.cmake)

add_library(ttnn_op_experimental_conv1d_depthwise ${LIB_TYPE})
add_library(TTNN::Ops::Experimental::Conv1dDepthwise ALIAS ttnn_op_experimental_conv1d_depthwise)

tt_reuse_precompile_headers(ttnn_op_experimental_conv1d_depthwise TTNN::PCH)
TT_ENABLE_UNITY_BUILD(ttnn_op_experimental_conv1d_depthwise)

set_target_properties(
    ttnn_op_experimental_conv1d_depthwise
    PROPERTIES
        INTERFACE_HEADER_SETS_TO_VERIFY
            api
)

file(GLOB_RECURSE kernels device/kernels/*)

target_sources(
    ttnn_op_experimental_conv1d_depthwise
    PUBLIC
        FILE_SET api
        TYPE HEADERS
        BASE_DIRS ${FixmeOpAPIDir}
        FILES ${TTNN_OP_EXPERIMENTAL_CONV1D_DEPTHWISE_API_HEADERS}
        FILE_SET kernels
        TYPE HEADERS
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
        FILES ${kernels}
    PRIVATE
        ${TTNN_OP_EXPERIMENTAL_CONV1D_DEPTHWISE_SRCS}
)

target_link_libraries(ttnn_op_experimental_conv1d_depthwise PRIVATE TT::Metalium PUBLIC TTNN::Core)

install(
    TARGETS
        ttnn_op_experimental_conv1d_depthwise
    FILE_SET
    kernels
        DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium/ttnn/cpp/ttnn/operations/experimental/conv1d_depthwise
        COMPONENT ttnn-runtime
)

install(TARGETS ttnn_op_experimental_conv1d_depthwise FILE_SET api COMPONENT ttnn-dev LIBRARY COMPONENT tar)
