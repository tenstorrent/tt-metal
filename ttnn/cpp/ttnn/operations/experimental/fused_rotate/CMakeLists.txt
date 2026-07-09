include(sources.cmake)

add_library(ttnn_op_experimental_fused_rotate ${LIB_TYPE})
add_library(TTNN::Ops::Experimental::FusedRotate ALIAS ttnn_op_experimental_fused_rotate)

tt_reuse_precompile_headers(ttnn_op_experimental_fused_rotate TTNN::PCH)
TT_ENABLE_UNITY_BUILD(ttnn_op_experimental_fused_rotate)

set_target_properties(
    ttnn_op_experimental_fused_rotate
    PROPERTIES
        INTERFACE_HEADER_SETS_TO_VERIFY
            api
)

file(GLOB_RECURSE kernels device/kernels/*)

target_sources(
    ttnn_op_experimental_fused_rotate
    PUBLIC
        FILE_SET api
        TYPE HEADERS
        BASE_DIRS ${FixmeOpAPIDir}
        FILES ${TTNN_OP_EXPERIMENTAL_FUSED_ROTATE_API_HEADERS}
        FILE_SET kernels
        TYPE HEADERS
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
        FILES ${kernels}
    PRIVATE
        ${TTNN_OP_EXPERIMENTAL_FUSED_ROTATE_SRCS}
)

target_link_libraries(ttnn_op_experimental_fused_rotate PRIVATE TT::Metalium PUBLIC TTNN::Core)

install(
    TARGETS
        ttnn_op_experimental_fused_rotate
    FILE_SET
    kernels
        DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium/ttnn/cpp/ttnn/operations/experimental/fused_rotate
        COMPONENT ttnn-runtime
)

install(TARGETS ttnn_op_experimental_fused_rotate FILE_SET api COMPONENT ttnn-dev LIBRARY COMPONENT tar)
