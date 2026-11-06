include(sources.cmake)

add_library(ttnn_op_experimental_deepseek_prefill_routed_expert_combine_overlap ${LIB_TYPE})
add_library(
    TTNN::Ops::Experimental::DeepSeekPrefill::RoutedExpertCombineOverlap
    ALIAS ttnn_op_experimental_deepseek_prefill_routed_expert_combine_overlap
)

tt_reuse_precompile_headers(ttnn_op_experimental_deepseek_prefill_routed_expert_combine_overlap TT::CommonPCH)
TT_ENABLE_UNITY_BUILD(ttnn_op_experimental_deepseek_prefill_routed_expert_combine_overlap)

# The routed expert's report hook, included BY the routed-expert writers at JIT compile time, so it is an install
# artifact.
file(GLOB_RECURSE kernels device/kernels/*.hpp)

target_sources(
    ttnn_op_experimental_deepseek_prefill_routed_expert_combine_overlap
    PUBLIC
        FILE_SET api
        TYPE HEADERS
        BASE_DIRS ${FixmeOpAPIDir}
        FILES ${TTNN_OP_EXPERIMENTAL_DEEPSEEK_PREFILL_ROUTED_EXPERT_COMBINE_OVERLAP_API_HEADERS}
        FILE_SET kernels
        TYPE HEADERS
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
        FILES ${kernels}
    PRIVATE
        ${TTNN_OP_EXPERIMENTAL_DEEPSEEK_PREFILL_ROUTED_EXPERT_COMBINE_OVERLAP_SRCS}
)

target_link_libraries(
    ttnn_op_experimental_deepseek_prefill_routed_expert_combine_overlap
    PRIVATE
        TT::Metalium
    PUBLIC
        TTNN::Core
)

install(
    TARGETS
        ttnn_op_experimental_deepseek_prefill_routed_expert_combine_overlap
    FILE_SET
    kernels
        DESTINATION
            ${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium/ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/routed_expert_combine_overlap
        COMPONENT ttnn-runtime
)

install(
    TARGETS
        ttnn_op_experimental_deepseek_prefill_routed_expert_combine_overlap
    FILE_SET
    api
        COMPONENT ttnn-dev
    LIBRARY
        COMPONENT tar
)
