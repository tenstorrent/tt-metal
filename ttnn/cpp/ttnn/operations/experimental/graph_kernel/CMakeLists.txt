include(sources.cmake)

add_library(ttnn_op_experimental_graph_kernel ${LIB_TYPE})
add_library(TTNN::Ops::Experimental::GraphKernel ALIAS ttnn_op_experimental_graph_kernel)

tt_reuse_precompile_headers(ttnn_op_experimental_graph_kernel TTNN::PCHFull)
TT_ENABLE_UNITY_BUILD(ttnn_op_experimental_graph_kernel)

set_target_properties(
    ttnn_op_experimental_graph_kernel
    PROPERTIES
        INTERFACE_HEADER_SETS_TO_VERIFY
            api
)

file(GLOB_RECURSE kernels device/kernels/*.cpp)

target_sources(
    ttnn_op_experimental_graph_kernel
    PUBLIC
        FILE_SET api
        TYPE HEADERS
        BASE_DIRS ${FixmeOpAPIDir}
        FILES ${TTNN_OP_EXPERIMENTAL_GRAPH_KERNEL_API_HEADERS}
        FILE_SET kernels
        TYPE HEADERS
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
        FILES ${kernels}
    PRIVATE
        ${TTNN_OP_EXPERIMENTAL_GRAPH_KERNEL_SRCS}
)

target_link_libraries(ttnn_op_experimental_graph_kernel PRIVATE TT::Metalium PUBLIC TTNN::Core)

install(
    TARGETS
        ttnn_op_experimental_graph_kernel
    FILE_SET
    kernels
        DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium/ttnn/cpp/ttnn/operations/experimental/graph_kernel
        COMPONENT ttnn-runtime
)

install(TARGETS ttnn_op_experimental_graph_kernel FILE_SET api COMPONENT ttnn-dev LIBRARY COMPONENT tar)

# Python bindings for this op are registered directly on the shared `ttnn`
# Python module target here (see plusone/CMakeLists.txt for the rationale).
if(TARGET ttnn)
    target_sources(ttnn PRIVATE ${TTNN_OP_EXPERIMENTAL_GRAPH_KERNEL_NANOBIND_SRCS})
endif()
