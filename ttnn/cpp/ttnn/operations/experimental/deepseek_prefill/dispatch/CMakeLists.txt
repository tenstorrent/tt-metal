add_library(ttnn_op_experimental_deepseek_prefill_dispatch ${LIB_TYPE})
add_library(TTNN::Ops::Experimental::DeepSeekPrefill::Dispatch ALIAS ttnn_op_experimental_deepseek_prefill_dispatch)

target_precompile_headers(ttnn_op_experimental_deepseek_prefill_dispatch REUSE_FROM TT::CommonPCH)
TT_ENABLE_UNITY_BUILD(ttnn_op_experimental_deepseek_prefill_dispatch)

target_sources(
    ttnn_op_experimental_deepseek_prefill_dispatch
    PRIVATE
        dispatch.cpp
        device/dispatch_device_operation.cpp
        device/dispatch_program_factory.cpp
)

target_include_directories(ttnn_op_experimental_deepseek_prefill_dispatch PRIVATE ${FixmeOpIncDirs})
target_link_libraries(
    ttnn_op_experimental_deepseek_prefill_dispatch
    PRIVATE
        TT::Metalium
        TTNN::Core
)

install(TARGETS ttnn_op_experimental_deepseek_prefill_dispatch LIBRARY COMPONENT tar)
