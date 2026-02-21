add_library(ttnn_op_experimental_deepseek_prefill_dispatch ${LIB_TYPE})
add_library(TTNN::Ops::Experimental::DeepSeek::PrefillDispatch ALIAS ttnn_op_experimental_deepseek_prefill_dispatch)

target_precompile_headers(ttnn_op_experimental_deepseek_prefill_dispatch REUSE_FROM TT::CommonPCH)
TT_ENABLE_UNITY_BUILD(ttnn_op_experimental_deepseek_prefill_dispatch)

target_sources(
    ttnn_op_experimental_deepseek_prefill_dispatch
    PRIVATE
        device/prefill_dispatch_device_operation.cpp
        device/prefill_dispatch_program_factory.cpp
)

target_include_directories(ttnn_op_experimental_deepseek_prefill_dispatch PRIVATE ${FixmeOpIncDirs})
target_link_libraries(
    ttnn_op_experimental_deepseek_prefill_dispatch
    PRIVATE
        TT::Metalium
        TTNN::Core
)

install(TARGETS ttnn_op_experimental_deepseek_prefill_dispatch LIBRARY COMPONENT tar)
