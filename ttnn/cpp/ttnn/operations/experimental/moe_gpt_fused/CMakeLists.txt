add_library(ttnn_op_experimental_moe_gpt_fused ${LIB_TYPE})
add_library(TTNN::Ops::Experimental::MoEGPTFused ALIAS ttnn_op_experimental_moe_gpt_fused)

target_precompile_headers(ttnn_op_experimental_moe_gpt_fused REUSE_FROM TT::CommonPCH)
TT_ENABLE_UNITY_BUILD(ttnn_op_experimental_moe_gpt_fused)

target_sources(
    ttnn_op_experimental_moe_gpt_fused
    PRIVATE
        moe_gpt_fused.cpp
        device/moe_gpt_fused_device_operation.cpp
        device/moe_gpt_fused_program_factory.cpp
)

target_include_directories(ttnn_op_experimental_moe_gpt_fused PRIVATE ${FixmeOpIncDirs})
target_link_libraries(
    ttnn_op_experimental_moe_gpt_fused
    PRIVATE
        TT::Metalium
        TTNN::Core
)

install(TARGETS ttnn_op_experimental_moe_gpt_fused LIBRARY COMPONENT tar)
