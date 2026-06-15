include(sources.cmake)

add_library(ttnn_op_eltwise_complex_unary ${LIB_TYPE})
add_library(TTNN::Ops::Eltwise::Complex::Unary ALIAS ttnn_op_eltwise_complex_unary)

tt_reuse_precompile_headers(ttnn_op_eltwise_complex_unary TTNN::PCHLean)
TT_ENABLE_UNITY_BUILD(ttnn_op_eltwise_complex_unary)

set_target_properties(
    ttnn_op_eltwise_complex_unary
    PROPERTIES
        INTERFACE_HEADER_SETS_TO_VERIFY
            api
)

target_sources(
    ttnn_op_eltwise_complex_unary
    PUBLIC
        FILE_SET api
        TYPE HEADERS
        BASE_DIRS ${FixmeOpAPIDir}
        FILES ${TTNN_OP_ELTWISE_COMPLEX_UNARY_API_HEADERS}
    PRIVATE
        ${TTNN_OP_ELTWISE_COMPLEX_UNARY_SRCS}
)

target_link_libraries(ttnn_op_eltwise_complex_unary PRIVATE TT::Metalium PUBLIC TTNN::Core)

install(TARGETS ttnn_op_eltwise_complex_unary FILE_SET api COMPONENT ttnn-dev LIBRARY COMPONENT tar)
