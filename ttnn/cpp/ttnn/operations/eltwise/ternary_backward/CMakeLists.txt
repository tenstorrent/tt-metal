include(sources.cmake)

add_library(ttnn_op_eltwise_ternary_backward ${LIB_TYPE})
add_library(TTNN::Ops::Eltwise::Ternary::Backward ALIAS ttnn_op_eltwise_ternary_backward)

tt_reuse_precompile_headers(ttnn_op_eltwise_ternary_backward TTNN::PCHLean)
TT_ENABLE_UNITY_BUILD(ttnn_op_eltwise_ternary_backward)

set_target_properties(
    ttnn_op_eltwise_ternary_backward
    PROPERTIES
        INTERFACE_HEADER_SETS_TO_VERIFY
            api
)

target_sources(
    ttnn_op_eltwise_ternary_backward
    PUBLIC
        FILE_SET api
        TYPE HEADERS
        BASE_DIRS ${FixmeOpAPIDir}
        FILES ${TTNN_OP_ELTWISE_TERNARY_BACKWARD_API_HEADERS}
    PRIVATE
        ${TTNN_OP_ELTWISE_TERNARY_BACKWARD_SRCS}
)

target_link_libraries(ttnn_op_eltwise_ternary_backward PRIVATE TT::Metalium PUBLIC TTNN::Core)

install(TARGETS ttnn_op_eltwise_ternary_backward FILE_SET api COMPONENT ttnn-dev LIBRARY COMPONENT tar)
