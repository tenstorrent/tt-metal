include(sources.cmake)

add_library(ttnn_op_experimental_deepseek_prefill_hybrid_routed_expert_ffn ${LIB_TYPE})
add_library(
    TTNN::Ops::Experimental::DeepSeekPrefill::HybridRoutedExpertFfn
    ALIAS ttnn_op_experimental_deepseek_prefill_hybrid_routed_expert_ffn
)

target_precompile_headers(ttnn_op_experimental_deepseek_prefill_hybrid_routed_expert_ffn REUSE_FROM TT::CommonPCH)
TT_ENABLE_UNITY_BUILD(ttnn_op_experimental_deepseek_prefill_hybrid_routed_expert_ffn)

set_target_properties(
    ttnn_op_experimental_deepseek_prefill_hybrid_routed_expert_ffn
    PROPERTIES
        INTERFACE_HEADER_SETS_TO_VERIFY
            api
)

# GLOB_RECURSE: this op carries both implementations' kernels, so they sit in compute/ and
# dataflow/ subdirectories as well as beside each other, and the .hpp files among them are
# included BY those kernels at JIT compile time, making them install artifacts too.
file(
    GLOB_RECURSE kernels
    device/kernels/*.cpp
    device/kernels/*.hpp
)

target_sources(
    ttnn_op_experimental_deepseek_prefill_hybrid_routed_expert_ffn
    PUBLIC
        FILE_SET api
        TYPE HEADERS
        BASE_DIRS ${FixmeOpAPIDir}
        FILES ${TTNN_OP_EXPERIMENTAL_DEEPSEEK_PREFILL_HYBRID_ROUTED_EXPERT_FFN_API_HEADERS}
        FILE_SET kernels
        TYPE HEADERS
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
        FILES ${kernels}
    PRIVATE
        ${TTNN_OP_EXPERIMENTAL_DEEPSEEK_PREFILL_HYBRID_ROUTED_EXPERT_FFN_SRCS}
)

target_link_libraries(
    ttnn_op_experimental_deepseek_prefill_hybrid_routed_expert_ffn
    PRIVATE
        TT::Metalium
        # The fused half's blocking geometry is used as it ships rather than copied.
        TTNN::Ops::Experimental::DeepSeekPrefill::MoeFusedSwiGlu
    PUBLIC
        TTNN::Core
)

install(
    TARGETS
        ttnn_op_experimental_deepseek_prefill_hybrid_routed_expert_ffn
    FILE_SET
    kernels
        DESTINATION
            ${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium/ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/hybrid_routed_expert_ffn
        COMPONENT ttnn-runtime
)

install(
    TARGETS
        ttnn_op_experimental_deepseek_prefill_hybrid_routed_expert_ffn
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
    target_sources(ttnn PRIVATE ${TTNN_OP_EXPERIMENTAL_DEEPSEEK_PREFILL_HYBRID_ROUTED_EXPERT_FFN_NANOBIND_SRCS})
endif()
