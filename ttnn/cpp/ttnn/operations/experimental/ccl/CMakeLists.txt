add_library(ttnn_op_experimental_ccl ${LIB_TYPE})
add_library(TTNN::Ops::Experimental::CCL ALIAS ttnn_op_experimental_ccl)

target_precompile_headers(ttnn_op_experimental_ccl REUSE_FROM TT::CommonPCH)
TT_ENABLE_UNITY_BUILD(ttnn_op_experimental_ccl)

target_sources(
    ttnn_op_experimental_ccl
    PRIVATE
        all_gather_matmul/all_gather_matmul.cpp
        all_gather_matmul/device/all_gather_matmul_op.cpp
        all_gather_matmul/device/multi_core/all_gather_matmul_op_multi_core.cpp
        all_reduce/all_reduce.cpp
        all_reduce/device/all_reduce_op.cpp
        all_reduce_async/device/all_reduce_async_op.cpp
        all_reduce_async/device/all_reduce_async_program_minimal_variants.cpp
        reduce_scatter_async/device/reduce_scatter_async_op.cpp
        reduce_scatter_async/device/reduce_scatter_async_program.cpp
        reduce_scatter_async/reduce_scatter.cpp
        llama_reduce_scatter/llama_reduce_scatter.cpp
        llama_reduce_scatter/device/llama_reduce_scatter_device_operation.cpp
        llama_reduce_scatter/device/llama_reduce_scatter_program_factory.cpp
        llama_reduce_scatter_matmul/rs_matmul.cpp
        llama_reduce_scatter_matmul/device/rs_matmul_op.cpp
        llama_reduce_scatter_matmul/device/rs_matmul_program_factory.cpp
        llama_reduce_scatter_create_heads/llama_reduce_scatter_create_heads.cpp
        llama_reduce_scatter_create_heads/device/llama_reduce_scatter_create_heads_device_op.cpp
        llama_reduce_scatter_create_heads/device/llama_reduce_scatter_create_heads_program_factory.cpp
        all_gather_async/all_gather_async.cpp
        all_gather_async/device/all_gather_async_op.cpp
        all_gather_async/device/all_gather_async_program.cpp
        all_gather_async/device/all_gather_async_program_minimal_variants.cpp
        all_to_all_async/all_to_all_async.cpp
        all_to_all_async/device/all_to_all_program_factory.cpp
        all_to_all_async/device/all_to_all_async_op.cpp
        all_reduce_async/all_reduce_async.cpp
        all_gather_concat_heads_fused/all_gather_concat.cpp
        all_gather_concat_heads_fused/device/all_gather_concat_op.cpp
        all_gather_concat_heads_fused/device/all_gather_concat_program.cpp
        all_gather_matmul_async/all_gather_matmul_async.cpp
        all_gather_matmul_async/device/all_gather_matmul_async_op.cpp
        all_gather_matmul_async/device/multi_core/all_gather_matmul_async_op_multi_core.cpp
        matmul_reduce_scatter_async/matmul_reduce_scatter_async.cpp
        matmul_reduce_scatter_async/device/matmul_reduce_scatter_async_op.cpp
        matmul_reduce_scatter_async/device/multi_core/matmul_reduce_scatter_async_op_multi_core.cpp
        reduce_scatter_minimal_async/reduce_scatter_minimal_async.cpp
        reduce_scatter_minimal_async/device/reduce_scatter_minimal_async_op.cpp
        reduce_scatter_minimal_async/device/reduce_scatter_minimal_async_program.cpp
        all_broadcast_async/all_broadcast_async.cpp
        all_broadcast_async/device/all_broadcast_async_op.cpp
        all_broadcast_async/device/all_broadcast_async_program.cpp
)

target_include_directories(ttnn_op_experimental_ccl PRIVATE ${FixmeOpIncDirs})
target_link_libraries(
    ttnn_op_experimental_ccl
    PRIVATE
        TT::Metalium
        TTNN::Core
)

install(TARGETS ttnn_op_experimental_ccl LIBRARY COMPONENT tar)
