include(sources.cmake)

add_library(ttnn_op_eltwise_binary ${LIB_TYPE})
add_library(TTNN::Ops::Eltwise::Binary ALIAS ttnn_op_eltwise_binary)

tt_reuse_precompile_headers(ttnn_op_eltwise_binary TTNN::PCHLean)
TT_ENABLE_UNITY_BUILD(ttnn_op_eltwise_binary)
set_target_properties(
    ttnn_op_eltwise_binary
    PROPERTIES
        INTERFACE_HEADER_SETS_TO_VERIFY
            api
)

# Globbing non-build files is acceptable for now because devs don't generate packages.
file(GLOB_RECURSE kernels device/kernels/*)
target_sources(
    ttnn_op_eltwise_binary
    PUBLIC
        FILE_SET api
        TYPE HEADERS
        BASE_DIRS ${FixmeOpAPIDir}
        FILES ${TTNN_OP_ELTWISE_BINARY_API_HEADERS}
        FILE_SET kernels
        TYPE HEADERS
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
        FILES ${kernels}
    PRIVATE
        ${TTNN_OP_ELTWISE_BINARY_SRCS}
)

target_include_directories(ttnn_op_eltwise_binary PRIVATE ${FixmeOpIncDirs})
target_link_libraries(ttnn_op_eltwise_binary PUBLIC TTNN::Core PRIVATE TT::Metalium)

install(
    TARGETS
        ttnn_op_eltwise_binary
    FILE_SET
    api
        COMPONENT ttnn-dev
    FILE_SET
    kernels
        DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium/ttnn/cpp/ttnn/operations/eltwise/binary
        COMPONENT ttnn-runtime
)

install(TARGETS ttnn_op_eltwise_binary LIBRARY COMPONENT tar)

# Python bindings for this op are registered directly on the shared `ttnn`
# Python module target here, rather than in the centrally-owned
# ttnn/sources.cmake — the target itself is necessarily shared (nanobind
# builds one Python extension module), but which file lists a given op's
# own registration source doesn't need to be. `ttnn` only exists when
# WITH_PYTHON_BINDINGS is on, hence the guard.
#
# The actual file list (TTNN_OP_ELTWISE_BINARY_NANOBIND_SRCS) lives in sources.cmake, not here: this
# file carries metalium-developers-infra as a required co-owner (so infra
# can land sweeping build-system changes), and routine add/remove/rename of
# a nanobind source shouldn't need to go through infra. This wiring line
# itself is build logic and is expected to change rarely.
if(TARGET ttnn)
    target_sources(ttnn PRIVATE ${TTNN_OP_ELTWISE_BINARY_NANOBIND_SRCS})
endif()
