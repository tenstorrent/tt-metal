include(sources.cmake)

add_library(ttnn_op_toy_scaled_add ${LIB_TYPE})
add_library(TTNN::Ops::ToyScaledAdd ALIAS ttnn_op_toy_scaled_add)

tt_reuse_precompile_headers(ttnn_op_toy_scaled_add TTNN::PCHFull)
TT_ENABLE_UNITY_BUILD(ttnn_op_toy_scaled_add)

set_target_properties(
    ttnn_op_toy_scaled_add
    PROPERTIES
        INTERFACE_HEADER_SETS_TO_VERIFY
            api
)

# The kernels stay with the Python package (ttnn/ttnn/operations/toy_scaled_add/kernels); this directory
# holds the argument schema they and the program factory include.
file(GLOB_RECURSE kernels device/kernels/*.hpp)

target_sources(
    ttnn_op_toy_scaled_add
    PUBLIC
        FILE_SET api
        TYPE HEADERS
        BASE_DIRS ${FixmeOpAPIDir}
        FILES ${TTNN_OP_TOY_SCALED_ADD_API_HEADERS}
        FILE_SET kernels
        TYPE HEADERS
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
        FILES ${kernels}
    PRIVATE
        ${TTNN_OP_TOY_SCALED_ADD_SRCS}
)

target_link_libraries(ttnn_op_toy_scaled_add PRIVATE TT::Metalium PUBLIC TTNN::Core)

install(
    TARGETS
        ttnn_op_toy_scaled_add
    FILE_SET
    kernels
        DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium/ttnn/cpp/ttnn/operations/toy_scaled_add
        COMPONENT ttnn-runtime
)

# The program factory launches the kernels by their root-relative path; install them at that same path.
file(GLOB toy_scaled_add_kernels ${PROJECT_SOURCE_DIR}/ttnn/ttnn/operations/toy_scaled_add/kernels/*.cpp)
install(
    FILES
        ${toy_scaled_add_kernels}
    DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium/ttnn/ttnn/operations/toy_scaled_add/kernels
    COMPONENT ttnn-runtime
)

install(TARGETS ttnn_op_toy_scaled_add FILE_SET api COMPONENT ttnn-dev LIBRARY COMPONENT tar)

# Python bindings for this op are registered directly on the shared `ttnn` Python module target; the
# file list lives in sources.cmake. `ttnn` only exists when WITH_PYTHON_BINDINGS is on, hence the guard.
if(TARGET ttnn)
    target_sources(ttnn PRIVATE ${TTNN_OP_TOY_SCALED_ADD_NANOBIND_SRCS})
endif()
