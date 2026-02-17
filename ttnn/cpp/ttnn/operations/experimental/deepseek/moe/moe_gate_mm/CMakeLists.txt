add_library(ttnn_op_experimental_deepseek_moe_moe_gate_mm ${LIB_TYPE})
add_library(TTNN::Ops::Experimental::DeepSeek::MoE::MoEGateMM ALIAS ttnn_op_experimental_deepseek_moe_moe_gate_mm)

target_precompile_headers(ttnn_op_experimental_deepseek_moe_moe_gate_mm REUSE_FROM TT::CommonPCH)
TT_ENABLE_UNITY_BUILD(ttnn_op_experimental_deepseek_moe_moe_gate_mm)

target_sources(
    ttnn_op_experimental_deepseek_moe_moe_gate_mm
    PRIVATE
        device/moe_gate_mm_device_operation.cpp
        device/moe_gate_mm_program_factory.cpp
)

target_include_directories(ttnn_op_experimental_deepseek_moe_moe_gate_mm PRIVATE ${FixmeOpIncDirs})
target_link_libraries(
    ttnn_op_experimental_deepseek_moe_moe_gate_mm
    PRIVATE
        TT::Metalium
        TTNN::Core
)

install(TARGETS ttnn_op_experimental_deepseek_moe_moe_gate_mm LIBRARY COMPONENT tar)
