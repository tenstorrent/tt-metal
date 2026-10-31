include(sources.cmake)

add_library(ttnn_op_experimental_sparse_kv_push ${LIB_TYPE})
add_library(TTNN::Ops::Experimental::SparseKvPush ALIAS ttnn_op_experimental_sparse_kv_push)

tt_reuse_precompile_headers(ttnn_op_experimental_sparse_kv_push TTNN::PCHFull)
TT_ENABLE_UNITY_BUILD(ttnn_op_experimental_sparse_kv_push)

set_target_properties(
    ttnn_op_experimental_sparse_kv_push
    PROPERTIES
        INTERFACE_HEADER_SETS_TO_VERIFY
            api
)

file(
    GLOB_RECURSE kernels
    device/kernels/*.cpp
    device/kernels/*.hpp
)
target_sources(
    ttnn_op_experimental_sparse_kv_push
    PUBLIC
        FILE_SET api
        TYPE HEADERS
        BASE_DIRS ${FixmeOpAPIDir}
        FILES ${TTNN_OP_EXPERIMENTAL_SPARSE_KV_PUSH_API_HEADERS}
        FILE_SET kernels
        TYPE HEADERS
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
        FILES ${kernels}
    PRIVATE
        ${TTNN_OP_EXPERIMENTAL_SPARSE_KV_PUSH_SRCS}
)

target_link_libraries(ttnn_op_experimental_sparse_kv_push PRIVATE TT::Metalium PUBLIC TTNN::Core)

install(
    TARGETS
        ttnn_op_experimental_sparse_kv_push
    FILE_SET
    kernels
        DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium/ttnn/cpp/ttnn/operations/experimental/sparse_kv_push
        COMPONENT ttnn-runtime
    FILE_SET
    api
        COMPONENT ttnn-dev
    LIBRARY
        COMPONENT tar
)

if(TARGET ttnn)
    target_sources(ttnn PRIVATE ${TTNN_OP_EXPERIMENTAL_SPARSE_KV_PUSH_NANOBIND_SRCS})
endif()
