include(sources.cmake)

add_library(ttnn_op_experimental_test ${LIB_TYPE})
add_library(TTNN::Ops::Experimental::Test ALIAS ttnn_op_experimental_test)

tt_reuse_precompile_headers(ttnn_op_experimental_test TTNN::PCHFull)

set_target_properties(
    ttnn_op_experimental_test
    PROPERTIES
        INTERFACE_HEADER_SETS_TO_VERIFY
            api
)

file(GLOB_RECURSE kernels hang_device/device/kernels/*.cpp)

target_sources(
    ttnn_op_experimental_test
    PUBLIC
        FILE_SET api
        TYPE HEADERS
        BASE_DIRS ${FixmeOpAPIDir}
        FILES
        FILE_SET kernels
        TYPE HEADERS
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
        FILES ${kernels}
    PRIVATE
        ${TTNN_OP_EXPERIMENTAL_TEST_SRCS}
)

target_include_directories(ttnn_op_experimental_test PRIVATE ${FixmeOpIncDirs})
target_link_libraries(
    ttnn_op_experimental_test
    PRIVATE
        TT::Metalium
        TTNN::Core
)

install(
    TARGETS
        ttnn_op_experimental_test
    FILE_SET
    kernels
        DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium/ttnn/cpp/ttnn/operations/experimental/test
        COMPONENT ttnn-runtime
)

install(TARGETS ttnn_op_experimental_test LIBRARY COMPONENT tar)

# Python bindings for this op are registered directly on the shared `ttnn`
# Python module target here, rather than in the centrally-owned
# ttnn/sources.cmake — the target itself is necessarily shared (nanobind
# builds one Python extension module), but which file lists a given op's
# own registration source doesn't need to be. `ttnn` only exists when
# WITH_PYTHON_BINDINGS is on, hence the guard.
#
# The actual file list (TTNN_OP_EXPERIMENTAL_TEST_NANOBIND_SRCS) lives in sources.cmake, not here: this
# file carries metalium-developers-infra as a required co-owner (so infra
# can land sweeping build-system changes), and routine add/remove/rename of
# a nanobind source shouldn't need to go through infra. This wiring line
# itself is build logic and is expected to change rarely.
if(TARGET ttnn)
    target_sources(ttnn PRIVATE ${TTNN_OP_EXPERIMENTAL_TEST_NANOBIND_SRCS})
endif()
