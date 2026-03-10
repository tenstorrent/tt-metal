add_library(ttnn_op_experimental_deepseek_prefill_dispatch_combined ${LIB_TYPE})
add_library(
    TTNN::Ops::Experimental::DeepSeek::PrefillDispatchCombined
    ALIAS ttnn_op_experimental_deepseek_prefill_dispatch_combined
)

target_precompile_headers(ttnn_op_experimental_deepseek_prefill_dispatch_combined REUSE_FROM TT::CommonPCH)
TT_ENABLE_UNITY_BUILD(ttnn_op_experimental_deepseek_prefill_dispatch_combined)

target_sources(
    ttnn_op_experimental_deepseek_prefill_dispatch_combined
    PRIVATE
        device/prefill_dispatch_combined_device_operation.cpp
        device/prefill_dispatch_combined_program_factory.cpp
)

target_include_directories(ttnn_op_experimental_deepseek_prefill_dispatch_combined PRIVATE ${FixmeOpIncDirs})
target_link_libraries(
    ttnn_op_experimental_deepseek_prefill_dispatch_combined
    PRIVATE
        TT::Metalium
        TTNN::Core
)

install(TARGETS ttnn_op_experimental_deepseek_prefill_dispatch_combined LIBRARY COMPONENT tar)
