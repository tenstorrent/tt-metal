include(sources.cmake)

add_library(ttnn_op_all_to_all_matmul ${LIB_TYPE})
add_library(TTNN::Ops::AllToAllMatmul ALIAS ttnn_op_all_to_all_matmul)

tt_reuse_precompile_headers(ttnn_op_all_to_all_matmul TTNN::PCHFull)
TT_ENABLE_UNITY_BUILD(ttnn_op_all_to_all_matmul)

set_target_properties(
    ttnn_op_all_to_all_matmul
    PROPERTIES
        INTERFACE_HEADER_SETS_TO_VERIFY
            api
)

# The kernels and the args header they share with the program factory.
file(
    GLOB_RECURSE kernels
    device/kernels/*.cpp
    device/kernels/*.hpp
)

target_sources(
    ttnn_op_all_to_all_matmul
    PUBLIC
        FILE_SET api
        TYPE HEADERS
        BASE_DIRS ${FixmeOpAPIDir}
        FILES ${TTNN_OP_ALL_TO_ALL_MATMUL_API_HEADERS}
        FILE_SET kernels
        TYPE HEADERS
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
        FILES ${kernels}
    PRIVATE
        ${TTNN_OP_ALL_TO_ALL_MATMUL_SRCS}
)

target_link_libraries(ttnn_op_all_to_all_matmul PRIVATE TT::Metalium PUBLIC TTNN::Core)

install(
    TARGETS
        ttnn_op_all_to_all_matmul
    FILE_SET
    kernels
        DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium/ttnn/cpp/ttnn/operations/all_to_all_matmul
        COMPONENT ttnn-runtime
)

install(TARGETS ttnn_op_all_to_all_matmul FILE_SET api COMPONENT ttnn-dev LIBRARY COMPONENT tar)

# Python bindings for this op are registered directly on the shared `ttnn` Python module target; the
# file list lives in sources.cmake. `ttnn` only exists when WITH_PYTHON_BINDINGS is on, hence the guard.
if(TARGET ttnn)
    target_sources(ttnn PRIVATE ${TTNN_OP_ALL_TO_ALL_MATMUL_NANOBIND_SRCS})
endif()
