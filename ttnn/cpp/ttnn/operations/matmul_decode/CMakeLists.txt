include(sources.cmake)

add_library(ttnn_op_matmul_decode ${LIB_TYPE})
add_library(TTNN::Ops::MatmulDecode ALIAS ttnn_op_matmul_decode)

tt_reuse_precompile_headers(ttnn_op_matmul_decode TTNN::PCHFull)
TT_ENABLE_UNITY_BUILD(ttnn_op_matmul_decode)

set_target_properties(
    ttnn_op_matmul_decode
    PROPERTIES
        INTERFACE_HEADER_SETS_TO_VERIFY
            api
)

target_sources(
    ttnn_op_matmul_decode
    PUBLIC
        FILE_SET api
        TYPE HEADERS
        BASE_DIRS ${FixmeOpAPIDir}
        FILES
    PRIVATE
        ${TTNN_OP_MATMUL_DECODE_SRCS}
)

target_include_directories(ttnn_op_matmul_decode PRIVATE ${FixmeOpIncDirs})
target_link_libraries(ttnn_op_matmul_decode PUBLIC TTNN::Core PRIVATE TT::Metalium)

install(TARGETS ttnn_op_matmul_decode LIBRARY COMPONENT tar)

# Python bindings for this op are registered directly on the shared `ttnn`
# Python module target (same pattern as ttnn/cpp/ttnn/operations/matmul).
# The file list (TTNN_OP_MATMUL_DECODE_NANOBIND_SRCS) lives in sources.cmake.
if(TARGET ttnn)
    target_sources(ttnn PRIVATE ${TTNN_OP_MATMUL_DECODE_NANOBIND_SRCS})
endif()
