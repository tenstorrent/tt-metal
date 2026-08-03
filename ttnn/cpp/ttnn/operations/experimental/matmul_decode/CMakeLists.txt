include(sources.cmake)

add_library(ttnn_op_experimental_matmul_decode ${LIB_TYPE})
add_library(TTNN::Ops::Experimental::MatmulDecode ALIAS ttnn_op_experimental_matmul_decode)

tt_reuse_precompile_headers(ttnn_op_experimental_matmul_decode TTNN::PCHFull)
TT_ENABLE_UNITY_BUILD(ttnn_op_experimental_matmul_decode)

set_target_properties(
    ttnn_op_experimental_matmul_decode
    PROPERTIES
        INTERFACE_HEADER_SETS_TO_VERIFY
            api
)

file(GLOB_RECURSE kernels device/kernels/*)

target_sources(
    ttnn_op_experimental_matmul_decode
    PUBLIC
        FILE_SET api
        TYPE HEADERS
        BASE_DIRS ${FixmeOpAPIDir}
        FILES ${TTNN_OP_EXPERIMENTAL_MATMUL_DECODE_API_HEADERS}
        FILE_SET kernels
        TYPE HEADERS
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
        FILES ${kernels}
    PRIVATE
        ${TTNN_OP_EXPERIMENTAL_MATMUL_DECODE_SRCS}
)

target_link_libraries(ttnn_op_experimental_matmul_decode PRIVATE TT::Metalium PUBLIC TTNN::Core)

install(
    TARGETS
        ttnn_op_experimental_matmul_decode
    FILE_SET
    kernels
        DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium/ttnn/cpp/ttnn/operations/experimental/matmul_decode
        COMPONENT ttnn-runtime
)

install(TARGETS ttnn_op_experimental_matmul_decode FILE_SET api COMPONENT ttnn-dev LIBRARY COMPONENT tar)

if(TARGET ttnn)
    target_sources(ttnn PRIVATE ${TTNN_OP_EXPERIMENTAL_MATMUL_DECODE_NANOBIND_SRCS})
endif()
