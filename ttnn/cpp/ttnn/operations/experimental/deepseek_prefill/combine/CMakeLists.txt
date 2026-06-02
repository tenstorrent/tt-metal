include(sources.cmake)

add_library(ttnn_op_experimental_deepseek_prefill_combine ${LIB_TYPE})
add_library(TTNN::Ops::Experimental::DeepSeekPrefill::Combine ALIAS ttnn_op_experimental_deepseek_prefill_combine)

target_precompile_headers(ttnn_op_experimental_deepseek_prefill_combine REUSE_FROM TT::CommonPCH)
TT_ENABLE_UNITY_BUILD(ttnn_op_experimental_deepseek_prefill_combine)

set_target_properties(
    ttnn_op_experimental_deepseek_prefill_combine
    PROPERTIES
        INTERFACE_HEADER_SETS_TO_VERIFY
            api
)

target_sources(
    ttnn_op_experimental_deepseek_prefill_combine
    PUBLIC
        FILE_SET api
        TYPE HEADERS
        BASE_DIRS ${FixmeOpAPIDir}
        FILES ${TTNN_OP_EXPERIMENTAL_DEEPSEEK_PREFILL_COMBINE_API_HEADERS}
    PRIVATE
        ${TTNN_OP_EXPERIMENTAL_DEEPSEEK_PREFILL_COMBINE_SRCS}
)

target_link_libraries(ttnn_op_experimental_deepseek_prefill_combine PRIVATE TT::Metalium PUBLIC TTNN::Core)

install(TARGETS ttnn_op_experimental_deepseek_prefill_combine FILE_SET api COMPONENT ttnn-dev LIBRARY COMPONENT tar)
