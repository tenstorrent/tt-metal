####################################################################################################
# Build TTNN and TTNNCPP
####################################################################################################

##################################################
# Set options
##################################################
if(ENABLE_COVERAGE AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    message(STATUS "Enabling code coverage flags for all ttnn targets")
    add_compile_options(--coverage)
    add_link_options(--coverage)
endif()

set(LIB_TYPE OBJECT)
if(ENABLE_TTNN_SHARED_SUBLIBS)
    # Building various components as .so reduces the final link of ttnn.so (and will make loading a bit slower).
    # This is important for development iterations in TT-NN.
    # TODO: Measure whether linker improvements (choice of linker + parallelism flags) render this obsolete.
    set(LIB_TYPE SHARED)
endif()

add_library(ttnn_core ${LIB_TYPE})
add_library(TTNN::Core ALIAS ttnn_core)

target_compile_definitions(ttnn_core PUBLIC "$<$<CXX_COMPILER_ID:GNU>:DISABLE_NAMESPACE_STATIC_ASSERT>")
target_precompile_headers(ttnn_core REUSE_FROM TT::CommonPCH)

set(TENSOR_FLATBUFFER_SCHEMAS
    ${CMAKE_CURRENT_SOURCE_DIR}/core/tensor/flatbuffer/mesh_shape.fbs
    ${CMAKE_CURRENT_SOURCE_DIR}/core/tensor/flatbuffer/tensor_spec.fbs
    ${CMAKE_CURRENT_SOURCE_DIR}/core/tensor/flatbuffer/tensor.fbs
)

set(TENSOR_FLATBUFFER_GENERATED_HEADERS)
foreach(FBS_FILE ${TENSOR_FLATBUFFER_SCHEMAS})
    GENERATE_FBS_HEADER(${FBS_FILE} TARGET TTNN::Core)
    list(APPEND TENSOR_FLATBUFFER_GENERATED_HEADERS ${FBS_GENERATED_HEADER_FILE})
endforeach()

target_sources(
    ttnn_core
    PRIVATE
        ${TENSOR_FLATBUFFER_GENERATED_HEADERS}
        core/async_runtime.cpp
        core/cluster.cpp
        core/config.cpp
        core/core.cpp
        core/device.cpp
        core/distributed/api.cpp
        core/distributed/distributed_tensor.cpp
        core/distributed/distributed_tensor_config.cpp
        core/distributed/distributed_tensor_config.cpp
        core/events.cpp
        core/global_circular_buffer.cpp
        core/global_semaphore.cpp
        core/graph/graph_argument_serializer.cpp
        core/graph/graph_processor.cpp
        core/graph/graph_trace_utils.cpp
        core/old_infra_device_operation.cpp
        core/reports.cpp
        core/run_operation.cpp
        core/tensor/flatbuffer/tensor_flatbuffer.cpp
        core/tensor/flatbuffer/tensor_spec_flatbuffer.cpp
        core/tensor/host_buffer/functions.cpp
        core/tensor/layout/alignment.cpp
        core/tensor/layout/page_config.cpp
        core/tensor/layout/tensor_layout.cpp
        core/tensor/serialization.cpp
        core/tensor/storage.cpp
        core/tensor/tensor.cpp
        core/tensor/tensor_accessor_args.cpp
        core/tensor/tensor_attributes.cpp
        core/tensor/tensor_impl.cpp
        core/tensor/tensor_ops.cpp
        core/tensor/tensor_spec.cpp
        core/tensor/tensor_utils.cpp
        core/tensor/types.cpp
        core/tensor/xtensor/partition.cpp
)
target_include_directories(
    ttnn_core
    PUBLIC
        api
    PRIVATE
        api/ttnn
        core
        cpp # FIXME: Depend on the Ops needed (make the circular obvious)
)
target_include_directories(ttnn_core SYSTEM PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/flatbuffers)
target_link_libraries(
    ttnn_core
    PRIVATE
        TT::Metalium
        xtensor
        FlatBuffers::FlatBuffers
        Boost::algorithm
)
install(TARGETS ttnn_core LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})

##################################################
# Collect all the files and folders that TTNN will use
##################################################
set(TTNN_SRC_PYBIND
    ${CMAKE_CURRENT_SOURCE_DIR}/core/distributed/distributed_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/core/graph/graph_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/bernoulli/bernoulli_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/conv/conv1d/conv1d_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/conv/conv2d/conv2d_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/conv/conv_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/conv/conv_transpose2d/conv_transpose2d_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/data_movement/data_movement_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/data_movement/concat/concat_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/data_movement/bcast/bcast_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/data_movement/chunk/chunk_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/data_movement/clone/clone_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/data_movement/concat/concat_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/data_movement/copy/copy_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/data_movement/chunk/chunk_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/data_movement/data_movement_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/data_movement/expand/expand_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/data_movement/fill_pad/fill_pad_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/data_movement/fill_rm/fill_rm_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/data_movement/fold/fold_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/data_movement/indexed_fill/indexed_fill_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/data_movement/move/move_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/data_movement/non_zero_indices/non_zero_indices_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/data_movement/pad/pad_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/data_movement/permute/permute_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/data_movement/repeat/repeat_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/data_movement/repeat_interleave/repeat_interleave_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/data_movement/reshape_view/reshape_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/data_movement/reshape_on_device/reshape_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/data_movement/reshape_view/reshape_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/data_movement/reshape_on_device/reshape_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/data_movement/roll/roll_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/data_movement/stack/stack_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/data_movement/sharded/interleaved_to_sharded/interleaved_to_sharded_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/data_movement/sharded/reshard/reshard_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/data_movement/sharded/sharded_to_interleaved/sharded_to_interleaved_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/data_movement/sharded_partial/interleaved_to_sharded_partial/interleaved_to_sharded_partial_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/data_movement/sharded_partial/sharded_to_interleaved_partial/sharded_to_interleaved_partial_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/data_movement/slice/slice_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/data_movement/split/split_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/data_movement/squeeze/squeeze_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/data_movement/stack/stack_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/data_movement/tilize/tilize_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/data_movement/tilize_with_val_padding/tilize_with_val_padding_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/data_movement/transpose/transpose_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/data_movement/unsqueeze/unsqueeze_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/data_movement/untilize/untilize_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/data_movement/untilize_with_unpadding/untilize_with_unpadding_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/data_movement/view/view_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/eltwise/binary/binary_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/eltwise/binary_backward/binary_backward_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/eltwise/complex/complex_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/eltwise/complex_unary/complex_unary_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/eltwise/complex_unary_backward/complex_unary_backward_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/eltwise/quantization/quantization_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/eltwise/ternary/ternary_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/eltwise/ternary_backward/ternary_backward_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/eltwise/unary/unary_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/eltwise/unary_backward/unary_backward_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/embedding/embedding_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/embedding_backward/embedding_backward_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/examples/example/example_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/examples/example_multiple_return/example_multiple_return_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/examples/examples_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/bcast_to/bcast_to_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/cnn/convert_to_chw/convert_to_chw_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/cnn/convert_to_hwc/convert_to_hwc_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/conv3d/conv3d_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/copy/typecast/typecast_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/dropout/dropout_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/experimental_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/matmul/attn_matmul/attn_matmul_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/matmul/group_attn_matmul/group_attn_matmul_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/paged_cache/paged_cache_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/plusone/plusone_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/reduction/argmax/argmax_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/reduction/fast_reduce_nc/fast_reduce_nc_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/reduction/fast_reduce_nc/fast_reduce_nc_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/reshape/view_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/slice_write/slice_write_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/padded_slice/padded_slice_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/ssm/hc_sum_reduce/hc_sum_reduce_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/ssm/prefix_scan/prefix_scan_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/ssm/repeat_and_interleave_eltwise_mul/repeat_and_interleave_eltwise_mul_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/transformer/all_reduce_create_qkv_heads/all_reduce_create_qkv_heads_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/transformer/concatenate_heads/concatenate_heads_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/transformer/create_qkv_heads/create_qkv_heads_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/transformer/create_qkv_heads_from_separate_tensors/create_qkv_heads_from_separate_tensors_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/transformer/nlp_concat_heads/nlp_concat_heads_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/transformer/nlp_concat_heads_decode/nlp_concat_heads_decode_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/transformer/nlp_create_qkv_heads/nlp_create_qkv_heads_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/transformer/nlp_create_qkv_heads_decode/nlp_create_qkv_heads_decode_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/transformer/nlp_create_qkv_heads_falcon7b/nlp_create_qkv_heads_falcon7b_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/transformer/nlp_create_qkv_heads_segformer/nlp_create_qkv_heads_segformer_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/transformer/nlp_create_qkv_heads_vit/nlp_create_qkv_heads_vit_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/transformer/nlp_kv_cache_load_slice/nlp_kv_cache_load_slice_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/transformer/rotary_embedding/rotary_embedding_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/transformer/rotary_embedding_llama/rotary_embedding_llama_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/transformer/rotary_embedding_llama/rotary_embedding_llama_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/transformer/rotary_embedding_llama_fused_qk/rotary_embedding_llama_fused_qk_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/transformer/rotate_half/rotate_half_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/transformer/split_query_key_value_and_split_heads/split_query_key_value_and_split_heads_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/unary_backward/gelu_backward/gelu_backward_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/full/full_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/full_like/full_like_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/generic/generic_op_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/index_fill/index_fill_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/kv_cache/kv_cache_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/loss/loss_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/matmul/matmul_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/moreh/moreh_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/moreh/moreh_abs_pow/moreh_abs_pow_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/moreh/moreh_adam/moreh_adam_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/moreh/moreh_adamw/moreh_adamw_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/moreh/moreh_arange/moreh_arange_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/moreh/moreh_bmm/moreh_bmm_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/moreh/moreh_bmm_backward/moreh_bmm_backward_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/moreh/moreh_clip_grad_norm/moreh_clip_grad_norm_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/moreh/moreh_cumsum/moreh_cumsum_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/moreh/moreh_dot/moreh_dot_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/moreh/moreh_dot/moreh_dot_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/moreh/moreh_dot_backward/moreh_dot_backward_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/moreh/moreh_fold/fold_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/moreh/moreh_getitem/moreh_getitem_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/moreh/moreh_group_norm/moreh_group_norm_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/moreh/moreh_group_norm_backward/moreh_group_norm_backward_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/moreh/moreh_layer_norm/moreh_layer_norm_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/moreh/moreh_layer_norm_backward/moreh_layer_norm_backward_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/moreh/moreh_linear/moreh_linear_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/moreh/moreh_linear_backward/moreh_linear_backward_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/moreh/moreh_matmul/moreh_matmul_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/moreh/moreh_matmul_backward/moreh_matmul_backward_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/moreh/moreh_mean/moreh_mean_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/moreh/moreh_mean_backward/moreh_mean_backward_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/moreh/moreh_nll_loss/moreh_nll_loss_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/moreh/moreh_nll_loss_backward/moreh_nll_loss_backward_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/moreh/moreh_nll_loss_unreduced_backward/moreh_nll_loss_unreduced_backward_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/moreh/moreh_norm/moreh_norm_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/moreh/moreh_norm_backward/moreh_norm_backward_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/moreh/moreh_sgd/moreh_sgd_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/moreh/moreh_softmax/moreh_softmax_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/moreh/moreh_softmax/moreh_softmax_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/moreh/moreh_softmax_backward/moreh_softmax_backward_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/moreh/moreh_sum/moreh_sum_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/moreh/moreh_sum_backward/moreh_sum_backward_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/normalization/batch_norm/batch_norm_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/normalization/groupnorm/groupnorm_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/normalization/layernorm/layernorm_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/normalization/layernorm_distributed/layernorm_distributed_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/normalization/normalization_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/normalization/rmsnorm/rmsnorm_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/normalization/rmsnorm/rmsnorm_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/normalization/rmsnorm_distributed/rmsnorm_distributed_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/normalization/softmax/softmax_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/pool/generic/generic_pools_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/pool/global_avg_pool/global_avg_pool_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/pool/upsample/upsample_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/prefetcher/prefetcher/dram_prefetcher_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/prefetcher/prefetcher_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/reduction/reduction_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/reduction/argmax/argmax_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/reduction/cumprod/cumprod_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/reduction/cumsum/cumsum_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/reduction/moe/moe_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/reduction/sampling/sampling_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/reduction/topk/topk_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/sliding_window/sliding_window_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/transformer/attention_softmax/attention_softmax_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/transformer/concatenate_heads/concatenate_heads_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/transformer/sdpa/sdpa_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/transformer/sdpa_decode/sdpa_decode_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/transformer/split_query_key_value_and_split_heads/split_query_key_value_and_split_heads_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/transformer/transformer_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/uniform/uniform_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/transformer/all_reduce_create_qkv_heads/all_reduce_create_qkv_heads_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/where/where_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/rand/rand_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/data_movement/sort/sort_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/data_movement/gather/gather_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/data_movement/gather/tosa/gather_tosa_${PY_BINDING}.cpp
)

set(CCL_EXPERIMENTAL_TTNN_SRCS_PYBIND
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/ccl/all_gather_async/all_gather_async_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/ccl/all_broadcast_async/all_broadcast_async_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/ccl/all_gather_concat_heads_fused/all_gather_concat_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/ccl/all_gather_matmul/all_gather_matmul_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/ccl/llama_reduce_scatter_matmul/rs_matmul_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/ccl/all_reduce/all_reduce_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/ccl/all_reduce_async/all_reduce_async_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/ccl/ccl_experimental_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/ccl/reduce_scatter_async/reduce_scatter_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/ccl/rms_allgather/rms_allgather_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/scatter/scatter_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/scatter/tosa_scatter_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/ccl/llama_reduce_scatter/llama_reduce_scatter_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/ccl/llama_reduce_scatter_create_heads/llama_reduce_scatter_create_heads_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/ccl/all_to_all_async/all_to_all_async_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/reduce_scatter_minimal_async_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/ccl/all_gather_matmul_async/all_gather_matmul_async_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/ccl/matmul_reduce_scatter_async/matmul_reduce_scatter_async_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/ccl/ring_attention_all_gather_async/ring_attention_all_gather_async_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/ccl/send_recv_async/send_async/send_async_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/ccl/send_recv_async/recv_async/recv_async_${PY_BINDING}.cpp
)

set(CCL_TTNN_SRCS_PYBIND
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/ccl/ccl_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/ccl/all_gather/all_gather_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/ccl/all_to_all_combine/all_to_all_combine_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/ccl/all_to_all_dispatch/all_to_all_dispatch_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/ccl/reduce_scatter/reduce_scatter_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/ccl/barrier/barrier_${PY_BINDING}.cpp
)

#TODO : should be using pybind11_add_module, but right now it introduces many build problems
#pybinds will always be built as a shared library
list(
    APPEND
    TTNN_SRC_PYBIND
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn-${PY_BINDING}/__init__.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn-${PY_BINDING}/activation.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn-${PY_BINDING}/cluster.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn-${PY_BINDING}/core.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn-${PY_BINDING}/device.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn-${PY_BINDING}/events.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn-${PY_BINDING}/fabric.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn-${PY_BINDING}/global_circular_buffer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn-${PY_BINDING}/global_semaphore.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn-${PY_BINDING}/mesh_socket.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn-${PY_BINDING}/profiler.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn-${PY_BINDING}/program_descriptors.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn-${PY_BINDING}/pytensor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn-${PY_BINDING}/reports.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn-${PY_BINDING}/tensor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn-${PY_BINDING}/types.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn-${PY_BINDING}/operations/copy.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn-${PY_BINDING}/operations/core.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn-${PY_BINDING}/operations/creation.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn-${PY_BINDING}/operations/trace.cpp
)

##################################################
# Build the libraries!
##################################################

#there are two ways to build TTNN
# 1) Without Python bindings
#        ttnncpp will be a dynamic library that can be linked to any project
# 2) With python bindings
#        ttnn will be a dynamic library with ttnncpp statically linked to it
# so if a user wants to consume ttnncpp for a project but also is interested on having
# ttnn as well, it should build it first without python bindings and then with them.
# Most of the use cases will link to ttnn without caring a lot about the backend, but
# for those who need this second use case, now it is officially supported

add_library(ttnncpp SHARED)
add_library(TTNN::CPP ALIAS ttnncpp)
add_library(Metalium::TTNNCPP ALIAS ttnncpp) # backwards compatibility. FIXME: remove

target_compile_definitions(ttnncpp PUBLIC "$<$<CXX_COMPILER_ID:GNU>:DISABLE_NAMESPACE_STATIC_ASSERT>")
target_precompile_headers(ttnncpp REUSE_FROM TT::CommonPCH)
TT_ENABLE_UNITY_BUILD(ttnncpp)

set_target_properties(
    ttnncpp
    PROPERTIES
        PREFIX
            "_"
        BUILD_RPATH
            "${PROJECT_BINARY_DIR}/tt_metal;${PROJECT_BINARY_DIR}/ttnn"
        INSTALL_RPATH
            "${PROJECT_BINARY_DIR}/lib;$ORIGIN/build/lib;$ORIGIN" # FIXME: Install RPATH should not have a build path!
        CXX_VISIBILITY_PRESET
            "default" # FIXME: Export the symbols we want, and use hidden visibility instead of default
        ADDITIONAL_CLEAN_FILES
            "${PROJECT_SOURCE_DIR}/ttnn/ttnn/_'${LIBRARY_NAME}'.so;${PROJECT_SOURCE_DIR}/ttnn/'${LIBRARY_NAME}'.egg-info" # FIXME: Confirm this is a relic of the past and can be deleted; No generated files should land in the source tree!
)

target_sources(
    ttnncpp
    PRIVATE
        # FIXME: Move these out to appropriate sub targets
        cpp/ttnn/operations/compute_throttle_utils.cpp
        cpp/ttnn/operations/sharding_utilities.cpp
        cpp/ttnn/operations/trace.cpp
        cpp/ttnn/operations/ccl/sharding_addrgen_helper.cpp
        cpp/ttnn/operations/generic/generic_op.cpp
        cpp/ttnn/operations/generic/device/generic_op_program_factory.cpp
        cpp/ttnn/operations/generic/device/generic_op_device_operation.cpp
        cpp/ttnn/operations/experimental/padded_slice/device/padded_slice_op.cpp
        cpp/ttnn/operations/experimental/padded_slice/device/padded_slice_program_factory.cpp
        cpp/ttnn/operations/experimental/padded_slice/padded_slice.cpp
        cpp/ttnn/operations/experimental/slice_write/device/slice_write_op.cpp
        cpp/ttnn/operations/experimental/slice_write/device/slice_write_program_factory.cpp
        cpp/ttnn/operations/experimental/slice_write/slice_write.cpp
        cpp/ttnn/operations/experimental/ccl/rms_allgather/device/rms_allgather_op.cpp
        cpp/ttnn/operations/experimental/ccl/rms_allgather/rms_allgather.cpp
        cpp/ttnn/operations/experimental/ccl/rms_allgather/device/multi_core/rms_allgather_pf.cpp
        cpp/ttnn/operations/normalization/rmsnorm_distributed/rmsnorm_pre_all_gather.cpp
        cpp/ttnn/operations/normalization/rmsnorm_distributed/rmsnorm_post_all_gather.cpp
        cpp/ttnn/operations/normalization/softmax/softmax.cpp
        cpp/ttnn/operations/normalization/softmax/device/multi_core/softmax_op_multi_core.cpp
        cpp/ttnn/operations/normalization/softmax/device/softmax_op.cpp
        cpp/ttnn/operations/copy/typecast/device/typecast_device_op.cpp
        cpp/ttnn/operations/copy/typecast/device/typecast_program_factory.cpp
        cpp/ttnn/operations/copy/typecast/device/typecast_sharded_program_factory.cpp
        cpp/ttnn/operations/copy/typecast/typecast.cpp
)

target_include_directories(
    ttnncpp
    PUBLIC
        api
        # FIXME: decide what to do about this dir. Device includes complicate matters.
        cpp
)

target_link_libraries(
    ttnncpp
    PRIVATE
        TT::Metalium
        TTNN::Core
        TTNN::Ops::Bernoulli
        TTNN::Ops::CCL
        TTNN::Ops::Conv
        TTNN::Ops::Core
        TTNN::Ops::DataMovement
        TTNN::Ops::Eltwise::Binary
        TTNN::Ops::Eltwise::Binary::Backward
        TTNN::Ops::Eltwise::Binary::NG
        TTNN::Ops::Eltwise::Complex
        TTNN::Ops::Eltwise::Complex::Binary
        TTNN::Ops::Eltwise::Complex::Unary
        TTNN::Ops::Eltwise::Complex::Unary::Backward
        TTNN::Ops::Eltwise::Quantization
        TTNN::Ops::Eltwise::Ternary
        TTNN::Ops::Eltwise::Ternary::Backward
        TTNN::Ops::Eltwise::Unary
        TTNN::Ops::Eltwise::Unary::Backward
        TTNN::Ops::Embedding
        TTNN::Ops::Embedding::Backward
        TTNN::Ops::Examples
        TTNN::Ops::Experimental::AutoFormat
        TTNN::Ops::Experimental::BcastTo
        TTNN::Ops::Experimental::CCL
        TTNN::Ops::Experimental::CNN
        TTNN::Ops::Experimental::Conv3d
        TTNN::Ops::Experimental::Copy
        TTNN::Ops::Experimental::Dropout
        TTNN::Ops::Experimental::Matmul
        TTNN::Ops::Experimental::PagedCache
        TTNN::Ops::Experimental::PlusOne
        TTNN::Ops::Experimental::Reduction
        TTNN::Ops::Experimental::Reshape
        TTNN::Ops::Experimental::Scatter
        TTNN::Ops::Experimental::SSM
        TTNN::Ops::Experimental::Transformer
        TTNN::Ops::Experimental::UnaryBackward
        TTNN::Ops::Experimental::Where
        TTNN::Ops::Full
        TTNN::Ops::FullLike
        TTNN::Ops::IndexFill
        TTNN::Ops::KvCache
        TTNN::Ops::Loss
        TTNN::Ops::Matmul
        TTNN::Ops::Moreh
        TTNN::Ops::Prefetcher
        TTNN::Ops::Reduction
        TTNN::Ops::SlidingWindow
        TTNN::Ops::Transformer
        TTNN::Ops::Uniform
        TTNN::Ops::Pool
        TTNN::Ops::Normalization
        TTNN::Ops::Rand
)
install(TARGETS ttnncpp LIBRARY COMPONENT ttnn-runtime)

install(
    TARGETS
        ttnncpp
    LIBRARY
        DESTINATION ${CMAKE_INSTALL_LIBDIR}
        COMPONENT tar
)

if(WITH_PYTHON_BINDINGS)
    add_library(ttnn SHARED)
    add_library(TTNN::PYTHON ALIAS ttnn)
    add_library(Metalium::TTNN ALIAS ttnn) # backwards compatibility. FIXME: remove
    target_compile_definitions(ttnn PUBLIC "$<$<CXX_COMPILER_ID:GNU>:DISABLE_NAMESPACE_STATIC_ASSERT>")
    TT_ENABLE_UNITY_BUILD(ttnn)

    set_target_properties(
        ttnn
        PROPERTIES
            PREFIX
                "_"
            BUILD_RPATH
                "${PROJECT_BINARY_DIR}/tt_metal;${PROJECT_BINARY_DIR}/ttnn"
            INSTALL_RPATH
                "${PROJECT_BINARY_DIR}/lib;$ORIGIN/build/lib;$ORIGIN" # FIXME: Install RPATH should not have a build path!
            CXX_VISIBILITY_PRESET
                "default" # FIXME: Export the symbols we want, and use hidden visibility instead of default
            ADDITIONAL_CLEAN_FILES
                "${PROJECT_SOURCE_DIR}/ttnn/ttnn/_'${LIBRARY_NAME}'.so;${PROJECT_SOURCE_DIR}/ttnn/'${LIBRARY_NAME}'.egg-info" # FIXME: Confirm this is a relic of the past and can be deleted; No generated files should land in the source tree!
    )

    target_sources(
        ttnn
        PRIVATE
            ${SRC_PYBIND}
            ${TTNN_SRC_PYBIND}
            ${CCL_TTNN_SRCS_PYBIND}
            ${CCL_EXPERIMENTAL_TTNN_SRCS_PYBIND}
    )
    target_include_directories(
        ttnn
        PUBLIC
            ${PROJECT_SOURCE_DIR}/ttnn
            ${CMAKE_CURRENT_SOURCE_DIR}/cpp
    )

    target_link_libraries(
        ttnn
        PUBLIC
            TTNN::CPP
        PRIVATE
            pybind11::module
            Boost::algorithm
            TT::Metalium
    )
    install(TARGETS ttnn LIBRARY COMPONENT ttnn-runtime)

    # Install .so into src files for pybinds implementation
    install(
        TARGETS
            ttnn
        ARCHIVE
            DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY
            DESTINATION ${CMAKE_INSTALL_LIBDIR}
            COMPONENT tar
    )

    install(
        TARGETS
            ttnn
            DESTINATION
            ${PROJECT_SOURCE_DIR}/ttnn/ttnn
            COMPONENT
            tt_pybinds
    )
endif()

if(TTNN_BUILD_TESTS)
    add_subdirectory(test)
endif()

set(FixmeOpIncDirs
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp
)
add_subdirectory(cpp/ttnn/operations/bernoulli)
add_subdirectory(cpp/ttnn/operations/ccl)
add_subdirectory(cpp/ttnn/operations/conv)
add_subdirectory(cpp/ttnn/operations/core)
add_subdirectory(cpp/ttnn/operations/data_movement)
add_subdirectory(cpp/ttnn/operations/eltwise/binary)
add_subdirectory(cpp/ttnn/operations/eltwise/binary_backward)
add_subdirectory(cpp/ttnn/operations/eltwise/binary_ng)
add_subdirectory(cpp/ttnn/operations/eltwise/complex)
add_subdirectory(cpp/ttnn/operations/eltwise/complex_binary)
add_subdirectory(cpp/ttnn/operations/eltwise/complex_unary)
add_subdirectory(cpp/ttnn/operations/eltwise/complex_unary_backward)
add_subdirectory(cpp/ttnn/operations/eltwise/quantization)
add_subdirectory(cpp/ttnn/operations/eltwise/ternary)
add_subdirectory(cpp/ttnn/operations/eltwise/ternary_backward)
add_subdirectory(cpp/ttnn/operations/eltwise/unary)
add_subdirectory(cpp/ttnn/operations/eltwise/unary_backward)
add_subdirectory(cpp/ttnn/operations/embedding)
add_subdirectory(cpp/ttnn/operations/embedding_backward)
add_subdirectory(cpp/ttnn/operations/examples)
add_subdirectory(cpp/ttnn/operations/experimental/auto_format)
add_subdirectory(cpp/ttnn/operations/experimental/bcast_to)
add_subdirectory(cpp/ttnn/operations/experimental/ccl)
add_subdirectory(cpp/ttnn/operations/experimental/cnn)
add_subdirectory(cpp/ttnn/operations/experimental/conv3d)
add_subdirectory(cpp/ttnn/operations/experimental/copy)
add_subdirectory(cpp/ttnn/operations/experimental/dropout)
add_subdirectory(cpp/ttnn/operations/experimental/matmul)
add_subdirectory(cpp/ttnn/operations/experimental/paged_cache)
add_subdirectory(cpp/ttnn/operations/experimental/plusone)
add_subdirectory(cpp/ttnn/operations/experimental/reduction)
add_subdirectory(cpp/ttnn/operations/experimental/reshape)
add_subdirectory(cpp/ttnn/operations/experimental/scatter)
add_subdirectory(cpp/ttnn/operations/experimental/ssm)
add_subdirectory(cpp/ttnn/operations/experimental/transformer)
add_subdirectory(cpp/ttnn/operations/experimental/unary_backward)
add_subdirectory(cpp/ttnn/operations/experimental/where)
add_subdirectory(cpp/ttnn/operations/full)
add_subdirectory(cpp/ttnn/operations/full_like)
add_subdirectory(cpp/ttnn/operations/index_fill)
add_subdirectory(cpp/ttnn/operations/kv_cache)
add_subdirectory(cpp/ttnn/operations/loss)
add_subdirectory(cpp/ttnn/operations/matmul)
add_subdirectory(cpp/ttnn/operations/moreh)
add_subdirectory(cpp/ttnn/operations/prefetcher)
add_subdirectory(cpp/ttnn/operations/reduction)
add_subdirectory(cpp/ttnn/operations/sliding_window)
add_subdirectory(cpp/ttnn/operations/transformer)
add_subdirectory(cpp/ttnn/operations/uniform)
add_subdirectory(cpp/ttnn/operations/pool)
add_subdirectory(cpp/ttnn/operations/normalization)
add_subdirectory(cpp/ttnn/operations/rand)
add_subdirectory(cpp/ttnn/deprecated)
