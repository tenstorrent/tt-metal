add_library(ttnn_op_experimental_ccl ${LIB_TYPE})
add_library(TTNN::Ops::Experimental::CCL ALIAS ttnn_op_experimental_ccl)

target_precompile_headers(ttnn_op_experimental_ccl REUSE_FROM TT::CommonPCH)
TT_ENABLE_UNITY_BUILD(ttnn_op_experimental_ccl)

set_target_properties(
    ttnn_op_experimental_ccl
    PROPERTIES
        VERIFY_INTERFACE_HEADER_SETS
            FALSE
)

target_sources(
    ttnn_op_experimental_ccl
    PUBLIC
        FILE_SET kernels
        TYPE HEADERS
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
        FILES
            send_recv_async/send_async/device/kernels/sender_reader.cpp
            send_recv_async/send_async/device/kernels/sender_writer.cpp
            send_recv_async/recv_async/device/kernels/receiver_inplace_writer.cpp
            send_recv_async/recv_async/device/kernels/receiver_writer.cpp
            send_recv_async/recv_async/device/kernels/receiver_reader.cpp
    PRIVATE
        llama_common.cpp
        common.cpp
        ring_attention_all_gather_async/ring_attention_all_gather_async.cpp
        ring_attention_all_gather_async/device/ring_attention_all_gather_async_op.cpp
        ring_attention_all_gather_async/device/ring_attention_all_gather_async_program.cpp
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
        all_gather_async/device/all_gather_async_program_minimal_variants.cpp
        all_gather_command_processor_async/all_gather_command_processor_async.cpp
        all_gather_command_processor_async/device/all_gather_command_processor_async_op.cpp
        all_gather_command_processor_async/device/all_gather_command_processor_async_program.cpp
        llama_all_gather_matmul_async/llama_all_gather_matmul_async.cpp
        llama_all_gather_matmul_async/device/llama_all_gather_matmul_async_op.cpp
        llama_all_gather_matmul_async/device/llama_all_gather_mm_async_program_factory.cpp
        llama_all_gather_matmul_async/device/llama_1d_mm_fusion_program_factory.cpp
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
        send_recv_async/send_async/send_async.cpp
        send_recv_async/send_async/device/send_async_op.cpp
        send_recv_async/send_async/device/send_async_program.cpp
        send_recv_async/recv_async/recv_async.cpp
        send_recv_async/recv_async/device/recv_async_op.cpp
        send_recv_async/recv_async/device/recv_async_program.cpp
)

target_include_directories(ttnn_op_experimental_ccl PRIVATE ${FixmeOpIncDirs})
target_link_libraries(
    ttnn_op_experimental_ccl
    PRIVATE
        TT::Metalium
        TTNN::Core
)

install(TARGETS ttnn_op_experimental_ccl LIBRARY COMPONENT tar)

install(
    TARGETS
        ttnn_op_experimental_ccl
    FILE_SET
    kernels
        DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium/ttnn/cpp/ttnn/operations/experimental/ccl
        COMPONENT ttnn-runtime
)
