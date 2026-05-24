include(sources.cmake)

add_library(ttnn_op_experimental_deepseek_moe_moe_gate_mm ${LIB_TYPE})
add_library(TTNN::Ops::Experimental::DeepSeek::MoE::MoEGateMM ALIAS ttnn_op_experimental_deepseek_moe_moe_gate_mm)

tt_reuse_precompile_headers(ttnn_op_experimental_deepseek_moe_moe_gate_mm TT::CommonPCH)
TT_ENABLE_UNITY_BUILD(ttnn_op_experimental_deepseek_moe_moe_gate_mm)

set_target_properties(
    ttnn_op_experimental_deepseek_moe_moe_gate_mm
    PROPERTIES
        INTERFACE_HEADER_SETS_TO_VERIFY
            api
)

target_sources(
    ttnn_op_experimental_deepseek_moe_moe_gate_mm
    PUBLIC
        FILE_SET api
        TYPE HEADERS
        BASE_DIRS ${FixmeOpAPIDir}
        FILES ${TTNN_OP_EXPERIMENTAL_DEEPSEEK_MOE_MOE_GATE_MM_API_HEADERS}
    PRIVATE
        ${TTNN_OP_EXPERIMENTAL_DEEPSEEK_MOE_MOE_GATE_MM_SRCS}
)

target_link_libraries(ttnn_op_experimental_deepseek_moe_moe_gate_mm PRIVATE TT::Metalium PUBLIC TTNN::Core)

install(TARGETS ttnn_op_experimental_deepseek_moe_moe_gate_mm FILE_SET api COMPONENT ttnn-dev LIBRARY COMPONENT tar)
