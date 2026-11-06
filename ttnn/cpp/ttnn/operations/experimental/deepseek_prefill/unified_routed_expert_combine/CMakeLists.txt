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

# The routed expert's kernels and the hook its writers report to combine with; combine's kernels are combine_fabric2d's.
# GLOB_RECURSE: they sit in compute/ and dataflow/, and the headers beside them are included BY those kernels (and by
# combine's) at JIT compile time, so they are install artifacts too.
file(
    GLOB_RECURSE kernels
    device/kernels/*.cpp
    device/kernels/*.hpp
    device/kernels/*.h
)

target_sources(
    ttnn_op_experimental_deepseek_prefill_unified_routed_expert_combine
    PUBLIC
        FILE_SET api
        TYPE HEADERS
        BASE_DIRS ${FixmeOpAPIDir}
        FILES ${TTNN_OP_EXPERIMENTAL_DEEPSEEK_PREFILL_UNIFIED_ROUTED_EXPERT_COMBINE_API_HEADERS}
        FILE_SET kernels
        TYPE HEADERS
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
        FILES ${kernels}
    PRIVATE
        ${TTNN_OP_EXPERIMENTAL_DEEPSEEK_PREFILL_UNIFIED_ROUTED_EXPERT_COMBINE_SRCS}
)

target_link_libraries(
    ttnn_op_experimental_deepseek_prefill_unified_routed_expert_combine
    PRIVATE
        TT::Metalium
        TTNN::Ops::Experimental::DeepSeekPrefill::UnifiedRoutedExpertFfn
    PUBLIC
        TTNN::Core
)

install(
    TARGETS
        ttnn_op_experimental_deepseek_prefill_unified_routed_expert_combine
    FILE_SET
    kernels
        DESTINATION
            ${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium/ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/unified_routed_expert_combine
        COMPONENT ttnn-runtime
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
