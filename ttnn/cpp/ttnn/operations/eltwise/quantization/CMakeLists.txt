include(sources.cmake)

add_library(ttnn_op_eltwise_quantization ${LIB_TYPE})
add_library(TTNN::Ops::Eltwise::Quantization ALIAS ttnn_op_eltwise_quantization)

tt_reuse_precompile_headers(ttnn_op_eltwise_quantization TTNN::PCHLean)
TT_ENABLE_UNITY_BUILD(ttnn_op_eltwise_quantization)

target_sources(
    ttnn_op_eltwise_quantization
    PUBLIC
        FILE_SET api
        TYPE HEADERS
        BASE_DIRS ${FixmeOpAPIDir}
        FILES ${TTNN_OP_ELTWISE_QUANTIZATION_API_HEADERS}
    PRIVATE
        ${TTNN_OP_ELTWISE_QUANTIZATION_SRCS}
)

target_include_directories(ttnn_op_eltwise_quantization PRIVATE ${FixmeOpIncDirs})
target_link_libraries(ttnn_op_eltwise_quantization PUBLIC TTNN::Core PRIVATE TT::Metalium)

install(
    TARGETS
        ttnn_op_eltwise_quantization
    FILE_SET
    api
        COMPONENT ttnn-dev
    FILE_SET
    kernels
        DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium/ttnn/cpp/ttnn/operations/eltwise/quantization
        COMPONENT ttnn-runtime
)

install(TARGETS ttnn_op_eltwise_quantization LIBRARY COMPONENT tar)

# Python bindings for this op are registered directly on the shared `ttnn`
# Python module target here, rather than in the centrally-owned
# ttnn/sources.cmake — the target itself is necessarily shared (nanobind
# builds one Python extension module), but which file lists a given op's
# own registration source doesn't need to be. `ttnn` only exists when
# WITH_PYTHON_BINDINGS is on, hence the guard.
#
# The actual file list (TTNN_OP_ELTWISE_QUANTIZATION_NANOBIND_SRCS) lives in sources.cmake, not here: this
# file carries metalium-developers-infra as a required co-owner (so infra
# can land sweeping build-system changes), and routine add/remove/rename of
# a nanobind source shouldn't need to go through infra. This wiring line
# itself is build logic and is expected to change rarely.
if(TARGET ttnn)
    target_sources(ttnn PRIVATE ${TTNN_OP_ELTWISE_QUANTIZATION_NANOBIND_SRCS})
endif()
