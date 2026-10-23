include(sources.cmake)

add_library(ttnn_op_experimental_deepseek_prefill_fused_experts_prefill ${LIB_TYPE})
add_library(
    TTNN::Ops::Experimental::DeepSeekPrefill::FusedExpertsPrefill
    ALIAS ttnn_op_experimental_deepseek_prefill_fused_experts_prefill
)

target_precompile_headers(ttnn_op_experimental_deepseek_prefill_fused_experts_prefill REUSE_FROM TT::CommonPCH)
TT_ENABLE_UNITY_BUILD(ttnn_op_experimental_deepseek_prefill_fused_experts_prefill)

set_target_properties(
    ttnn_op_experimental_deepseek_prefill_fused_experts_prefill
    PROPERTIES
        INTERFACE_HEADER_SETS_TO_VERIFY
            api
)

# GLOB_RECURSE: this op's kernels sit in compute/ and dataflow/ subdirectories, and the .hpp beside
# them is included BY those kernels at JIT compile time, so it is an install artifact too.
file(
    GLOB_RECURSE kernels
    device/kernels/*.cpp
    device/kernels/*.hpp
)

target_sources(
    ttnn_op_experimental_deepseek_prefill_fused_experts_prefill
    PUBLIC
        FILE_SET api
        TYPE HEADERS
        BASE_DIRS ${FixmeOpAPIDir}
        FILES ${TTNN_OP_EXPERIMENTAL_DEEPSEEK_PREFILL_FUSED_EXPERTS_PREFILL_API_HEADERS}
        FILE_SET kernels
        TYPE HEADERS
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
        FILES ${kernels}
    PRIVATE
        device/fused_experts_prefill_device_operation.cpp
        device/fused_experts_prefill_program_factory.cpp
        fused_experts_prefill.cpp
)

target_link_libraries(
    ttnn_op_experimental_deepseek_prefill_fused_experts_prefill
    PRIVATE
        TT::Metalium
    PUBLIC
        TTNN::Core
)

install(
    TARGETS
        ttnn_op_experimental_deepseek_prefill_fused_experts_prefill
    FILE_SET
    kernels
        DESTINATION
            ${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium/ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/fused_experts_prefill
        COMPONENT ttnn-runtime
)

install(
    TARGETS
        ttnn_op_experimental_deepseek_prefill_fused_experts_prefill
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
    target_sources(ttnn PRIVATE ${TTNN_OP_EXPERIMENTAL_DEEPSEEK_PREFILL_FUSED_EXPERTS_PREFILL_NANOBIND_SRCS})
endif()
