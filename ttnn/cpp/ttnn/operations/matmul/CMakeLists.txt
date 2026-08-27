include(sources.cmake)

add_library(ttnn_op_matmul ${LIB_TYPE})
add_library(TTNN::Ops::Matmul ALIAS ttnn_op_matmul)

set(MATMUL_REGISTRY_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/device/config/registry)
set(MATMUL_REGISTRY_GENERATED_DIR ${MATMUL_REGISTRY_SOURCE_DIR}/checked_in)
set(MATMUL_REGISTRY_GENERATED_SOURCE ${MATMUL_REGISTRY_GENERATED_DIR}/matmul_registry_data.cpp)

# Registry promotion is an offline release step. Normal TTNN builds compile
# the reviewed generated snapshot and therefore need no Python interpreter,
# JSON parser, or codegen action. The emitter test owns the deterministic
# freshness check between the lock and these checked-in files.
set_source_files_properties(
    ${MATMUL_REGISTRY_GENERATED_SOURCE}
    PROPERTIES
        SKIP_LINTING
            TRUE
        SKIP_UNITY_BUILD_INCLUSION
            ON
)

tt_reuse_precompile_headers(ttnn_op_matmul TTNN::PCHFull)
TT_ENABLE_UNITY_BUILD(ttnn_op_matmul)
set_target_properties(
    ttnn_op_matmul
    PROPERTIES
        INTERFACE_HEADER_SETS_TO_VERIFY
            api
)

file(
    GLOB_RECURSE kernels
    device/kernels/*.cpp
    device/kernels/*.hpp
    shared_with_host/*.hpp
)
target_sources(
    ttnn_op_matmul
    PUBLIC
        FILE_SET api
        TYPE HEADERS
        BASE_DIRS ${FixmeOpAPIDir}
        FILES ${TTNN_OP_MATMUL_API_HEADERS}
        FILE_SET kernels
        TYPE HEADERS
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
        FILES ${kernels}
    PRIVATE
        ${TTNN_OP_MATMUL_SRCS}
)

target_include_directories(
    ttnn_op_matmul
    PRIVATE
        ${FixmeOpIncDirs}
        ${MATMUL_REGISTRY_GENERATED_DIR}
)
target_link_libraries(ttnn_op_matmul PUBLIC TTNN::Core PRIVATE TT::Metalium)

install(
    TARGETS
        ttnn_op_matmul
    FILE_SET
    api
        COMPONENT ttnn-dev
    FILE_SET
    kernels
        DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium/ttnn/cpp/ttnn/operations/matmul
        COMPONENT ttnn-runtime
)
install(TARGETS ttnn_op_matmul LIBRARY COMPONENT tar)

# Python bindings for this op are registered directly on the shared `ttnn`
# Python module target here, rather than in the centrally-owned
# ttnn/sources.cmake — the target itself is necessarily shared (nanobind
# builds one Python extension module), but which file lists a given op's
# own registration source doesn't need to be. `ttnn` only exists when
# WITH_PYTHON_BINDINGS is on, hence the guard.
#
# The actual file list (TTNN_OP_MATMUL_NANOBIND_SRCS) lives in sources.cmake, not here: this
# file carries metalium-developers-infra as a required co-owner (so infra
# can land sweeping build-system changes), and routine add/remove/rename of
# a nanobind source shouldn't need to go through infra. This wiring line
# itself is build logic and is expected to change rarely.
if(TARGET ttnn)
    target_sources(ttnn PRIVATE ${TTNN_OP_MATMUL_NANOBIND_SRCS})
endif()
