include(sources.cmake)

add_library(ttnn_op_experimental_deepseek_prefill_dispatch ${LIB_TYPE})
add_library(TTNN::Ops::Experimental::DeepSeekPrefill::Dispatch ALIAS ttnn_op_experimental_deepseek_prefill_dispatch)

target_precompile_headers(ttnn_op_experimental_deepseek_prefill_dispatch REUSE_FROM TT::CommonPCH)
TT_ENABLE_UNITY_BUILD(ttnn_op_experimental_deepseek_prefill_dispatch)

set_target_properties(
    ttnn_op_experimental_deepseek_prefill_dispatch
    PROPERTIES
        INTERFACE_HEADER_SETS_TO_VERIFY
            api
)

target_sources(
    ttnn_op_experimental_deepseek_prefill_dispatch
    PUBLIC
        FILE_SET api
        TYPE HEADERS
        BASE_DIRS ${FixmeOpAPIDir}
        FILES ${TTNN_OP_EXPERIMENTAL_DEEPSEEK_PREFILL_DISPATCH_API_HEADERS}
    PRIVATE
        ${TTNN_OP_EXPERIMENTAL_DEEPSEEK_PREFILL_DISPATCH_SRCS}
)

target_link_libraries(ttnn_op_experimental_deepseek_prefill_dispatch PRIVATE TT::Metalium PUBLIC TTNN::Core)

install(TARGETS ttnn_op_experimental_deepseek_prefill_dispatch FILE_SET api COMPONENT ttnn-dev LIBRARY COMPONENT tar)
