add_library(ttnn_op_experimental_deepseek_prefill_combine ${LIB_TYPE})
add_library(TTNN::Ops::Experimental::DeepSeek::PrefillCombine ALIAS ttnn_op_experimental_deepseek_prefill_combine)

target_precompile_headers(ttnn_op_experimental_deepseek_prefill_combine REUSE_FROM TT::CommonPCH)
TT_ENABLE_UNITY_BUILD(ttnn_op_experimental_deepseek_prefill_combine)

target_sources(
    ttnn_op_experimental_deepseek_prefill_combine
    PRIVATE
        device/prefill_combine_device_operation.cpp
        device/prefill_combine_program_factory.cpp
)

target_include_directories(ttnn_op_experimental_deepseek_prefill_combine PRIVATE ${FixmeOpIncDirs})
target_link_libraries(
    ttnn_op_experimental_deepseek_prefill_combine
    PRIVATE
        TT::Metalium
        TTNN::Core
)

install(TARGETS ttnn_op_experimental_deepseek_prefill_combine LIBRARY COMPONENT tar)
