include(sources.cmake)

add_library(ttnn_op_eltwise_complex ${LIB_TYPE})
add_library(TTNN::Ops::Eltwise::Complex ALIAS ttnn_op_eltwise_complex)

tt_reuse_precompile_headers(ttnn_op_eltwise_complex TTNN::PCH)
TT_ENABLE_UNITY_BUILD(ttnn_op_eltwise_complex)

target_sources(
    ttnn_op_eltwise_complex
    PUBLIC
        FILE_SET api
        TYPE HEADERS
        BASE_DIRS ${FixmeOpAPIDir}
        FILES ${TTNN_OP_ELTWISE_COMPLEX_API_HEADERS}
    PRIVATE
        ${TTNN_OP_ELTWISE_COMPLEX_SRCS}
)

target_include_directories(ttnn_op_eltwise_complex PRIVATE ${FixmeOpIncDirs})
target_link_libraries(ttnn_op_eltwise_complex PUBLIC TTNN::Core PRIVATE TT::Metalium)

install(TARGETS ttnn_op_eltwise_complex FILE_SET api COMPONENT ttnn-dev LIBRARY COMPONENT tar)
