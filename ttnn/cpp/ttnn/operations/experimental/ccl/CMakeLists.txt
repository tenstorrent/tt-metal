add_library(ttnn_op_experimental_ccl ${LIB_TYPE})
add_library(TTNN::Ops::Experimental::CCL ALIAS ttnn_op_experimental_ccl)

target_precompile_headers(ttnn_op_experimental_ccl REUSE_FROM TT::CommonPCH)
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
    deepseek_minimal_broadcast/device/kernels/*
)

target_sources(
    ttnn_op_experimental_ccl
    PUBLIC
        FILE_SET api
        TYPE HEADERS
        BASE_DIRS ${FixmeOpAPIDir}
        FILES
            all_reduce_async/all_reduce_async.hpp
            all_gather_async/all_gather_async.hpp
            all_gather_async/device/all_gather_async_device_operation.hpp
            all_gather_async/device/all_gather_async_default_program_factory.hpp
            all_gather_async/device/all_gather_async_llama_sharded_program_factory.hpp
            all_gather_async/device/all_gather_async_device_operation_types.hpp
            composite_common.hpp
            reduce_scatter_minimal_async/device/reduce_scatter_minimal_async_op_device_operation.hpp
            reduce_scatter_minimal_async/device/reduce_scatter_minimal_async_op_device_operation_types.hpp
            reduce_scatter_minimal_async/device/reduce_scatter_ring_program_factory.hpp
            reduce_scatter_minimal_async/device/reduce_scatter_line_program_factory.hpp
            reduce_scatter_minimal_async/reduce_scatter_minimal_async.hpp
        FILE_SET kernels
        TYPE HEADERS
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
        FILES ${kernels}
    PRIVATE
        llama_common.cpp
        composite_common.cpp
        ring_attention_all_gather_async/ring_attention_all_gather_async.cpp
        ring_attention_all_gather_async/device/ring_attention_all_gather_async_op.cpp
        ring_attention_all_gather_async/device/ring_attention_all_gather_async_program.cpp
        all_reduce_async/device/all_reduce_async_device_operation.cpp
        all_reduce_async/device/all_reduce_async_program_factory.cpp
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
        all_gather_async/device/all_gather_async_device_operation.cpp
        all_gather_async/device/all_gather_async_default_program_factory.cpp
        all_gather_async/device/all_gather_async_llama_sharded_program_factory.cpp
        llama_all_gather_matmul_async/llama_all_gather_matmul_async.cpp
        llama_all_gather_matmul_async/device/llama_all_gather_matmul_async_device_operation.cpp
        llama_all_gather_matmul_async/device/llama_all_gather_matmul_async_program_factory.cpp
        llama_all_gather_matmul_async/device/llama_1d_mm_fusion.cpp
        all_to_all_async/all_to_all_async.cpp
        all_to_all_async/device/all_to_all_async_device_operation.cpp
        all_to_all_async/device/all_to_all_async_program_factory.cpp
        all_to_all_async_generic/all_to_all_async_generic.cpp
        all_to_all_async_generic/device/all_to_all_async_generic_device_operation.cpp
        all_to_all_async_generic/device/all_to_all_async_generic_program_factory.cpp
        all_reduce_async/all_reduce_async.cpp
        all_gather_concat_heads_fused/all_gather_concat.cpp
        all_gather_concat_heads_fused/device/all_gather_concat_device_operation.cpp
        all_gather_concat_heads_fused/device/all_gather_concat_program_factory.cpp
        all_gather_matmul_async/all_gather_matmul_async.cpp
        all_gather_matmul_async/device/all_gather_matmul_async_device_operation.cpp
        all_gather_matmul_async/device/all_gather_matmul_async_program_factory.cpp
        strided_all_gather_minimal_matmul_async/strided_all_gather_minimal_matmul_async.cpp
        strided_all_gather_minimal_matmul_async/device/strided_all_gather_minimal_matmul_async_op.cpp
        strided_all_gather_minimal_matmul_async/device/strided_all_gather_minimal_matmul_async_program.cpp
        matmul_reduce_scatter_async/matmul_reduce_scatter_async.cpp
        matmul_reduce_scatter_async/device/matmul_reduce_scatter_async_device_operation.cpp
        matmul_reduce_scatter_async/device/matmul_reduce_scatter_async_program_factory.cpp
        reduce_scatter_minimal_async/reduce_scatter_minimal_async.cpp
        reduce_scatter_minimal_async/device/reduce_scatter_minimal_async_op_device_operation.cpp
        reduce_scatter_minimal_async/device/reduce_scatter_minimal_async_program.cpp
        send_recv_async/send_async/send_async.cpp
        send_recv_async/send_async/device/send_async_op_device_operation.cpp
        send_recv_async/send_async/device/send_async_op_program_factory.cpp
        send_recv_async/recv_async/recv_async.cpp
        send_recv_async/recv_async/device/recv_async_op_device_operation.cpp
        send_recv_async/recv_async/device/recv_async_op_program_factory.cpp
        neighbor_pad_async/neighbor_pad_async.cpp
        neighbor_pad_async/device/neighbor_pad_async_device_operation.cpp
        neighbor_pad_async/device/neighbor_pad_async_program_factory.cpp
        slice_reshard_async/slice_reshard_async.cpp
        slice_reshard_async/device/slice_reshard_async_device_operation.cpp
        slice_reshard_async/device/slice_reshard_async_program_factory.cpp
        strided_all_gather_async/strided_all_gather_async.cpp
        strided_all_gather_async/device/strided_all_gather_async_op.cpp
        strided_all_gather_async/device/strided_all_gather_async_program.cpp
        deepseek_minimal_broadcast/deepseek_minimal_broadcast.cpp
        deepseek_minimal_broadcast/device/deepseek_minimal_broadcast_device_operation.cpp
        deepseek_minimal_broadcast/device/deepseek_minimal_broadcast_program_factory.cpp
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
