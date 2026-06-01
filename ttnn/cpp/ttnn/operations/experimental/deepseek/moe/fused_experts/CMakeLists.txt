include(sources.cmake)

add_library(ttnn_op_experimental_deepseek_moe_fused_experts ${LIB_TYPE})
add_library(TTNN::Ops::Experimental::DeepSeek::MoE::FusedExperts ALIAS ttnn_op_experimental_deepseek_moe_fused_experts)

tt_reuse_precompile_headers(ttnn_op_experimental_deepseek_moe_fused_experts TT::CommonPCH)
TT_ENABLE_UNITY_BUILD(ttnn_op_experimental_deepseek_moe_fused_experts)

set_target_properties(
    ttnn_op_experimental_deepseek_moe_fused_experts
    PROPERTIES
        INTERFACE_HEADER_SETS_TO_VERIFY
            api
)

target_sources(
    ttnn_op_experimental_deepseek_moe_fused_experts
    PUBLIC
        FILE_SET api
        TYPE HEADERS
        BASE_DIRS ${FixmeOpAPIDir}
        FILES ${TTNN_OP_EXPERIMENTAL_DEEPSEEK_MOE_FUSED_EXPERTS_API_HEADERS}
    PRIVATE
        ${TTNN_OP_EXPERIMENTAL_DEEPSEEK_MOE_FUSED_EXPERTS_SRCS}
)

target_link_libraries(ttnn_op_experimental_deepseek_moe_fused_experts PRIVATE TT::Metalium PUBLIC TTNN::Core)

install(TARGETS ttnn_op_experimental_deepseek_moe_fused_experts FILE_SET api COMPONENT ttnn-dev LIBRARY COMPONENT tar)
