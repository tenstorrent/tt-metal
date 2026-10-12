include(sources.cmake)

add_library(ttnn_op_experimental_deepseek_prefill_shared_expert_ffn ${LIB_TYPE})
add_library(
    TTNN::Ops::Experimental::DeepSeekPrefill::SharedExpertFfn
    ALIAS ttnn_op_experimental_deepseek_prefill_shared_expert_ffn
)

target_precompile_headers(ttnn_op_experimental_deepseek_prefill_shared_expert_ffn REUSE_FROM TT::CommonPCH)
TT_ENABLE_UNITY_BUILD(ttnn_op_experimental_deepseek_prefill_shared_expert_ffn)

set_target_properties(
    ttnn_op_experimental_deepseek_prefill_shared_expert_ffn
    PROPERTIES
        INTERFACE_HEADER_SETS_TO_VERIFY
            api
)

# No kernels of its own: the program is unified_routed_expert_ffn's, built in its dense mode.
target_sources(
    ttnn_op_experimental_deepseek_prefill_shared_expert_ffn
    PUBLIC
        FILE_SET api
        TYPE HEADERS
        BASE_DIRS ${FixmeOpAPIDir}
        FILES ${TTNN_OP_EXPERIMENTAL_DEEPSEEK_PREFILL_SHARED_EXPERT_FFN_API_HEADERS}
    PRIVATE
        ${TTNN_OP_EXPERIMENTAL_DEEPSEEK_PREFILL_SHARED_EXPERT_FFN_SRCS}
)

target_link_libraries(
    ttnn_op_experimental_deepseek_prefill_shared_expert_ffn
    PRIVATE
        TT::Metalium
        TTNN::Ops::Experimental::DeepSeekPrefill::UnifiedRoutedExpertFfn
    PUBLIC
        TTNN::Core
)

install(
    TARGETS
        ttnn_op_experimental_deepseek_prefill_shared_expert_ffn
    FILE_SET
    api
        COMPONENT ttnn-dev
    LIBRARY
        COMPONENT tar
)

if(TARGET ttnn)
    target_sources(ttnn PRIVATE ${TTNN_OP_EXPERIMENTAL_DEEPSEEK_PREFILL_SHARED_EXPERT_FFN_NANOBIND_SRCS})
endif()
