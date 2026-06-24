include(sources.cmake)

add_library(ttnn_op_experimental_deepseek_mla_matmul_wo ${LIB_TYPE})
add_library(TTNN::Ops::Experimental::Deepseek::MLA::Matmul_WO ALIAS ttnn_op_experimental_deepseek_mla_matmul_wo)

tt_reuse_precompile_headers(ttnn_op_experimental_deepseek_mla_matmul_wo TT::CommonPCH)
TT_ENABLE_UNITY_BUILD(ttnn_op_experimental_deepseek_mla_matmul_wo)

set_target_properties(
    ttnn_op_experimental_deepseek_mla_matmul_wo
    PROPERTIES
        INTERFACE_HEADER_SETS_TO_VERIFY
            api
)

target_sources(
    ttnn_op_experimental_deepseek_mla_matmul_wo
    PUBLIC
        FILE_SET api
        TYPE HEADERS
        BASE_DIRS ${FixmeOpAPIDir}
        FILES ${TTNN_OP_EXPERIMENTAL_DEEPSEEK_MLA_MATMUL_WO_API_HEADERS}
    PRIVATE
        ${TTNN_OP_EXPERIMENTAL_DEEPSEEK_MLA_MATMUL_WO_SRCS}
)

target_link_libraries(ttnn_op_experimental_deepseek_mla_matmul_wo PRIVATE TT::Metalium PUBLIC TTNN::Core)

install(TARGETS ttnn_op_experimental_deepseek_mla_matmul_wo FILE_SET api COMPONENT ttnn-dev LIBRARY COMPONENT tar)
