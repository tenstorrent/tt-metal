include(sources.cmake)

add_library(ttnn_op_experimental_ccl ${LIB_TYPE})
add_library(TTNN::Ops::Experimental::CCL ALIAS ttnn_op_experimental_ccl)

tt_reuse_precompile_headers(ttnn_op_experimental_ccl TTNN::PCH)
TT_ENABLE_UNITY_BUILD(ttnn_op_experimental_ccl)

set_target_properties(
    ttnn_op_experimental_ccl
    PROPERTIES
        INTERFACE_HEADER_SETS_TO_VERIFY
            api
)

file(
    GLOB_RECURSE kernels
    all_gather_async/device/kernels/*
    all_gather_concat_heads_fused/device/kernels/*
    all_reduce_async/device/kernels/*
    all_to_all_async/device/kernels/*
    all_to_all_async_generic/device/kernels/*
    all_to_all_dispatch_metadata/device/kernels/*
    llama_all_gather_matmul_async/device/kernels/*
    llama_reduce_scatter/device/kernels/*
    llama_reduce_scatter_create_heads/device/kernels/*
    neighbor_pad_async/device/kernels/*
    reduce_scatter_minimal_async/device/kernels/*
    ring_attention_all_gather_async/device/kernels/*
    rms_allgather/device/kernels/*
    send_recv_async/send_async/device/kernels/*
    send_recv_async/recv_async/device/kernels/*
    slice_reshard_async/device/kernels/*
    strided_all_gather_async/device/kernels/*
    strided_reduce_scatter_async/device/kernels/*
    deepseek_moe_reduce_scatter/device/kernels/*
    deepseek_b1_reduce_to_one/device/kernels/*
    moe_compute/device/kernels/*
    moe/selective_reduce_combine/device/kernels/*
)

target_sources(
    ttnn_op_experimental_ccl
    PUBLIC
        FILE_SET api
        TYPE HEADERS
        BASE_DIRS ${FixmeOpAPIDir}
        FILES ${TTNN_OP_EXPERIMENTAL_CCL_API_HEADERS}
        FILE_SET kernels
        TYPE HEADERS
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
        FILES ${kernels}
    PRIVATE
        ${TTNN_OP_EXPERIMENTAL_CCL_SRCS}
)

target_include_directories(ttnn_op_experimental_ccl PRIVATE ${FixmeOpIncDirs})
target_link_libraries(ttnn_op_experimental_ccl PUBLIC TTNN::Core PRIVATE TT::Metalium)

install(TARGETS ttnn_op_experimental_ccl LIBRARY COMPONENT tar)

install(
    TARGETS
        ttnn_op_experimental_ccl
    FILE_SET
    api
        COMPONENT ttnn-dev
    FILE_SET
    kernels
        DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium/ttnn/cpp/ttnn/operations/experimental/ccl
        COMPONENT ttnn-runtime
)
