add_library(ttnn_op_experimental_deepseek_prefill_routed_expert_ffn_device ${LIB_TYPE})
add_library(
    TTNN::Ops::Experimental::DeepSeekPrefill::RoutedExpertFFN::Device
    ALIAS ttnn_op_experimental_deepseek_prefill_routed_expert_ffn_device
)

target_precompile_headers(ttnn_op_experimental_deepseek_prefill_routed_expert_ffn_device REUSE_FROM TT::CommonPCH)
TT_ENABLE_UNITY_BUILD(ttnn_op_experimental_deepseek_prefill_routed_expert_ffn_device)

target_sources(
    ttnn_op_experimental_deepseek_prefill_routed_expert_ffn_device
    PRIVATE
        gate_up_matmul_device_op.cpp
        gate_up_matmul_program_factory.cpp
)

target_include_directories(ttnn_op_experimental_deepseek_prefill_routed_expert_ffn_device PRIVATE ${FixmeOpIncDirs})

target_link_libraries(
    ttnn_op_experimental_deepseek_prefill_routed_expert_ffn_device
    PRIVATE
        TT::Metalium
        TTNN::Core
)

install(TARGETS ttnn_op_experimental_deepseek_prefill_routed_expert_ffn_device LIBRARY COMPONENT tar)
