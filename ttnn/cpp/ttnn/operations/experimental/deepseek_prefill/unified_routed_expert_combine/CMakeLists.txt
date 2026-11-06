include(sources.cmake)

add_library(ttnn_op_experimental_deepseek_prefill_unified_routed_expert_combine ${LIB_TYPE})
add_library(
    TTNN::Ops::Experimental::DeepSeekPrefill::UnifiedRoutedExpertCombine
    ALIAS ttnn_op_experimental_deepseek_prefill_unified_routed_expert_combine
)

tt_reuse_precompile_headers(ttnn_op_experimental_deepseek_prefill_unified_routed_expert_combine TT::CommonPCH)
TT_ENABLE_UNITY_BUILD(ttnn_op_experimental_deepseek_prefill_unified_routed_expert_combine)

set_target_properties(
    ttnn_op_experimental_deepseek_prefill_unified_routed_expert_combine
    PROPERTIES
        INTERFACE_HEADER_SETS_TO_VERIFY
            api
)

# No kernels of its own: the routed expert's are unified_routed_expert_ffn's and combine's are combine_fabric2d's.
target_sources(
    ttnn_op_experimental_deepseek_prefill_unified_routed_expert_combine
    PUBLIC
        FILE_SET api
        TYPE HEADERS
        BASE_DIRS ${FixmeOpAPIDir}
        FILES ${TTNN_OP_EXPERIMENTAL_DEEPSEEK_PREFILL_UNIFIED_ROUTED_EXPERT_COMBINE_API_HEADERS}
    PRIVATE
        ${TTNN_OP_EXPERIMENTAL_DEEPSEEK_PREFILL_UNIFIED_ROUTED_EXPERT_COMBINE_SRCS}
)

target_link_libraries(
    ttnn_op_experimental_deepseek_prefill_unified_routed_expert_combine
    PRIVATE
        TT::Metalium
        TTNN::Ops::Experimental::DeepSeekPrefill::RoutedExpertCombineOverlap
        TTNN::Ops::Experimental::DeepSeekPrefill::UnifiedRoutedExpertFfn
    PUBLIC
        TTNN::Core
)

install(
    TARGETS
        ttnn_op_experimental_deepseek_prefill_unified_routed_expert_combine
    FILE_SET
    api
        COMPONENT ttnn-dev
    LIBRARY
        COMPONENT tar
)

# Python bindings for this op are registered directly on the shared `ttnn`
# Python module target here, rather than in the centrally-owned
# ttnn/sources.cmake. `ttnn` only exists when WITH_PYTHON_BINDINGS is on,
# hence the guard.
if(TARGET ttnn)
    target_sources(ttnn PRIVATE ${TTNN_OP_EXPERIMENTAL_DEEPSEEK_PREFILL_UNIFIED_ROUTED_EXPERT_COMBINE_NANOBIND_SRCS})
endif()
