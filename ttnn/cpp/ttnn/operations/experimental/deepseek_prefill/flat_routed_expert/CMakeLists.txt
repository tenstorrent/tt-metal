include(sources.cmake)

add_library(ttnn_op_experimental_deepseek_prefill_flat_routed_expert ${LIB_TYPE})
add_library(
    TTNN::Ops::Experimental::DeepSeekPrefill::FlatRoutedExpert
    ALIAS ttnn_op_experimental_deepseek_prefill_flat_routed_expert
)

tt_reuse_precompile_headers(ttnn_op_experimental_deepseek_prefill_flat_routed_expert TT::CommonPCH)

set_target_properties(
    ttnn_op_experimental_deepseek_prefill_flat_routed_expert
    PROPERTIES
        INTERFACE_HEADER_SETS_TO_VERIFY
            api
)

# the kernels and the .hpp beside them (included BY the kernels at JIT compile time) are install artifacts
file(
    GLOB_RECURSE kernels
    device/kernels/*.cpp
    device/kernels/*.hpp
)

target_sources(
    ttnn_op_experimental_deepseek_prefill_flat_routed_expert
    PUBLIC
        FILE_SET api
        TYPE HEADERS
        BASE_DIRS ${FixmeOpAPIDir}
        FILES ${TTNN_OP_EXPERIMENTAL_DEEPSEEK_PREFILL_FLAT_ROUTED_EXPERT_API_HEADERS}
        FILE_SET kernels
        TYPE HEADERS
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
        FILES ${kernels}
    PRIVATE
        device/flat_routed_expert_device_operation.cpp
        device/flat_routed_expert_plan.cpp
        device/flat_routed_expert_program_factory.cpp
        flat_routed_expert.cpp
        flat_routed_expert_weights.cpp
)

target_link_libraries(ttnn_op_experimental_deepseek_prefill_flat_routed_expert PRIVATE TT::Metalium PUBLIC TTNN::Core)

install(
    TARGETS
        ttnn_op_experimental_deepseek_prefill_flat_routed_expert
    FILE_SET
    kernels
        DESTINATION
            ${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium/ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/flat_routed_expert
        COMPONENT ttnn-runtime
)

install(
    TARGETS
        ttnn_op_experimental_deepseek_prefill_flat_routed_expert
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
    target_sources(ttnn PRIVATE ${TTNN_OP_EXPERIMENTAL_DEEPSEEK_PREFILL_FLAT_ROUTED_EXPERT_NANOBIND_SRCS})
endif()
