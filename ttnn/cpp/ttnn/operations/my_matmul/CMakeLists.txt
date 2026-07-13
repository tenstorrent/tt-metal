include(sources.cmake)

add_library(ttnn_op_my_matmul ${LIB_TYPE})
add_library(TTNN::Ops::MyMatmul ALIAS ttnn_op_my_matmul)

tt_reuse_precompile_headers(ttnn_op_my_matmul TTNN::PCHLean)
TT_ENABLE_UNITY_BUILD(ttnn_op_my_matmul)

set_target_properties(
    ttnn_op_my_matmul
    PROPERTIES
        INTERFACE_HEADER_SETS_TO_VERIFY
            api
)

file(GLOB_RECURSE kernels device/kernels/*)

target_sources(
    ttnn_op_my_matmul
    PUBLIC
        FILE_SET api
        TYPE HEADERS
        BASE_DIRS ${FixmeOpAPIDir}
        FILES ${TTNN_OP_MY_MATMUL_API_HEADERS}
        FILE_SET kernels
        TYPE HEADERS
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
        FILES ${kernels}
    PRIVATE
        ${TTNN_OP_MY_MATMUL_SRCS}
)

target_link_libraries(ttnn_op_my_matmul PRIVATE TT::Metalium PUBLIC TTNN::Core)

install(
    TARGETS
        ttnn_op_my_matmul
    FILE_SET
    kernels
        DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium/ttnn/cpp/ttnn/operations/my_matmul
        COMPONENT ttnn-runtime
)

install(TARGETS ttnn_op_my_matmul FILE_SET api COMPONENT ttnn-dev LIBRARY COMPONENT tar)
