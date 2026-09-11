include(sources.cmake)

add_library(ttnn_op_experimental_deepseek_prefill_moe_fused_swiglu ${LIB_TYPE})
add_library(
    TTNN::Ops::Experimental::DeepSeekPrefill::MoeFusedSwiGlu
    ALIAS ttnn_op_experimental_deepseek_prefill_moe_fused_swiglu
)

target_precompile_headers(ttnn_op_experimental_deepseek_prefill_moe_fused_swiglu REUSE_FROM TT::CommonPCH)
TT_ENABLE_UNITY_BUILD(ttnn_op_experimental_deepseek_prefill_moe_fused_swiglu)

set_target_properties(
    ttnn_op_experimental_deepseek_prefill_moe_fused_swiglu
    PROPERTIES
        INTERFACE_HEADER_SETS_TO_VERIFY
            api
)

# The .hpp are included BY the kernel .cpp at JIT compile time, so they are install artifacts too:
# without them a packaged install has the three entry points and none of what they include.
file(
    GLOB_RECURSE kernels
    device/kernels/*.cpp
    device/kernels/*.hpp
)

target_sources(
    ttnn_op_experimental_deepseek_prefill_moe_fused_swiglu
    PUBLIC
        FILE_SET api
        TYPE HEADERS
        BASE_DIRS ${FixmeOpAPIDir}
        FILES ${TTNN_OP_EXPERIMENTAL_DEEPSEEK_PREFILL_MOE_FUSED_SWIGLU_API_HEADERS}
        FILE_SET kernels
        TYPE HEADERS
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
        FILES ${kernels}
    PRIVATE
        device/moe_fused_swiglu_device_operation.cpp
        device/moe_fused_swiglu_geometry.cpp
        device/moe_fused_swiglu_program_factory.cpp
        moe_fused_swiglu.cpp
)

target_link_libraries(ttnn_op_experimental_deepseek_prefill_moe_fused_swiglu PRIVATE TT::Metalium PUBLIC TTNN::Core)

install(
    TARGETS
        ttnn_op_experimental_deepseek_prefill_moe_fused_swiglu
    FILE_SET
    kernels
        DESTINATION
            ${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium/ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/moe_fused_swiglu
        COMPONENT ttnn-runtime
)

install(
    TARGETS
        ttnn_op_experimental_deepseek_prefill_moe_fused_swiglu
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
    target_sources(ttnn PRIVATE ${TTNN_OP_EXPERIMENTAL_DEEPSEEK_PREFILL_MOE_FUSED_SWIGLU_NANOBIND_SRCS})
endif()
