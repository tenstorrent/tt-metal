####################################################################################################
# Build TTNN and TTNNCPP
####################################################################################################

add_compile_options("$<$<CXX_COMPILER_ID:Clang>:-Wunused-parameter>")

# FIXME: Many lambda functions in ttnn capture variables and don't use them.
add_compile_options("$<$<CXX_COMPILER_ID:Clang>:-Wno-unused-lambda-capture>")

# Suppress narrowing warnings for ttnn (being cleaned up in tt_metal first)
add_compile_options("$<$<CXX_COMPILER_ID:Clang>:-Wno-c++11-narrowing>")
add_compile_options("$<$<CXX_COMPILER_ID:GNU>:-Wno-narrowing>")

##################################################
# Set options
##################################################

set(LIB_TYPE OBJECT)
if(ENABLE_TTNN_SHARED_SUBLIBS)
    # Building various components as .so reduces the final link of ttnn.so (and will make loading a bit slower).
    # This is important for development iterations in TT-NN.
    # TODO: Measure whether linker improvements (choice of linker + parallelism flags) render this obsolete.
    set(LIB_TYPE SHARED)
endif()

add_library(ttnn_core ${LIB_TYPE})
add_library(TTNN::Core ALIAS ttnn_core)

# Enable unity builds for faster compilation
TT_ENABLE_UNITY_BUILD(ttnn_core)

target_precompile_headers(ttnn_core REUSE_FROM TT::CommonPCH)

set(TENSOR_FLATBUFFER_SCHEMAS
    ${CMAKE_CURRENT_SOURCE_DIR}/core/tensor/flatbuffer/mesh_shape.fbs
    ${CMAKE_CURRENT_SOURCE_DIR}/core/tensor/flatbuffer/tensor_spec.fbs
    ${CMAKE_CURRENT_SOURCE_DIR}/core/tensor/flatbuffer/tensor_topology.fbs
    ${CMAKE_CURRENT_SOURCE_DIR}/core/tensor/flatbuffer/tensor.fbs
)

set(TENSOR_FLATBUFFER_GENERATED_HEADERS)
foreach(FBS_FILE ${TENSOR_FLATBUFFER_SCHEMAS})
    GENERATE_FBS_HEADER(${FBS_FILE} TARGET TTNN::Core)
    list(APPEND TENSOR_FLATBUFFER_GENERATED_HEADERS ${FBS_GENERATED_HEADER_FILE})
endforeach()

set_target_properties(
    ttnn_core
    PROPERTIES
        INTERFACE_HEADER_SETS_TO_VERIFY
            api
)
target_sources(
    ttnn_core
    PUBLIC
        FILE_SET api
        TYPE HEADERS
        BASE_DIRS ${FixmeOpAPIDir}
        FILES
        FILE_SET jit_api
        TYPE HEADERS
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
        FILES cpp/ttnn/kernel/kernel_common_utils.hpp cpp/ttnn/kernel/kernel_utils.hpp api/ttnn/tensor/layout/layout.hpp
    PRIVATE
        ${TENSOR_FLATBUFFER_GENERATED_HEADERS}
        core/async_runtime.cpp
        core/cluster.cpp
        core/config.cpp
        core/core.cpp
        core/device.cpp
        core/distributed/api.cpp
        core/distributed/distributed_tensor.cpp
        core/distributed/distributed_configs.cpp
        core/distributed/distribution_mode.cpp
        core/distributed/host_ccl.cpp
        core/distributed/tensor_topology.cpp
        core/distributed/bidirectional_fabric_socket.cpp
        core/distributed/create_socket.cpp
        core/distributed/fabric_socket.cpp
        core/distributed/mpi_socket.cpp
        core/events.cpp
        core/global_circular_buffer.cpp
        core/global_semaphore.cpp
        core/graph/graph_argument_serializer.cpp
        core/graph/graph_processor.cpp
        core/graph/graph_trace_utils.cpp
        core/graph/levelized_graph.cpp
        core/old_infra_device_operation.cpp
        core/reports.cpp
        core/run_operation.cpp
        core/tensor/flatbuffer/tensor_flatbuffer.cpp
        core/tensor/flatbuffer/tensor_spec_flatbuffer.cpp
        core/tensor/host_buffer/functions.cpp
        core/tensor/layout/alignment.cpp
        core/tensor/layout/layout.cpp
        core/tensor/layout/page_config.cpp
        core/tensor/layout/tensor_layout.cpp
        core/tensor/memory_config/memory_config.cpp
        core/tensor/serialization.cpp
        core/tensor/storage.cpp
        core/tensor/tensor.cpp
        core/tensor/tensor_attributes.cpp
        core/tensor/tensor_impl.cpp
        core/tensor/tensor_ops.cpp
        core/tensor/tensor_spec.cpp
        core/tensor/tensor_utils.cpp
        core/tensor/types.cpp
        core/tensor/unit_mesh/unit_mesh_utils.cpp
        core/tensor/xtensor/partition.cpp
        core/tensor/to_string.cpp
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
    PUBLIC
        TT::Metalium
    PRIVATE
        xtensor
        FlatBuffers::FlatBuffers
        Boost::algorithm
)

install(
    TARGETS
        ttnn_core
    LIBRARY
        DESTINATION ${CMAKE_INSTALL_LIBDIR}
    FILE_SET
    jit_api
        DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium/ttnn
        COMPONENT ttnn-runtime
)

##################################################
# Collect all the files and folders that TTNN will use
##################################################
set(TTNN_SRC_PYBIND
    ${CMAKE_CURRENT_SOURCE_DIR}/core/distributed/distributed_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/core/graph/graph_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/bernoulli/bernoulli_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/conv/conv1d/conv1d_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/conv/conv2d/conv2d_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/conv/conv_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/conv/conv_transpose2d/conv_transpose2d_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/data_movement/data_movement_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/data_movement/concat/concat_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/data_movement/bcast/bcast_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/data_movement/chunk/chunk_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/data_movement/clone/clone_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/data_movement/concat/concat_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/data_movement/copy/copy_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/data_movement/chunk/chunk_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/data_movement/expand/expand_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/data_movement/fill_pad/fill_pad_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/data_movement/fill_rm/fill_rm_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/data_movement/fold/fold_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/data_movement/indexed_fill/indexed_fill_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/data_movement/moe_expert_token_remap/moe_expert_token_remap_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/data_movement/moe_routing_remap/moe_routing_remap_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/data_movement/move/move_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/data_movement/non_zero_indices/non_zero_indices_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/data_movement/pad/pad_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/data_movement/permute/permute_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/data_movement/repeat/repeat_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/data_movement/repeat_interleave/repeat_interleave_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/data_movement/reshape_view/reshape_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/data_movement/reshape_on_device/reshape_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/data_movement/reshape_view/reshape_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/data_movement/reshape_on_device/reshape_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/data_movement/roll/roll_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/data_movement/stack/stack_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/data_movement/scatter/scatter_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/data_movement/scatter/tosa_scatter_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/data_movement/sharded/interleaved_to_sharded/interleaved_to_sharded_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/data_movement/sharded/reshard/reshard_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/data_movement/sharded/sharded_to_interleaved/sharded_to_interleaved_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/data_movement/sharded_partial/interleaved_to_sharded_partial/interleaved_to_sharded_partial_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/data_movement/sharded_partial/sharded_to_interleaved_partial/sharded_to_interleaved_partial_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/data_movement/slice/slice_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/data_movement/split/split_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/data_movement/squeeze/squeeze_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/data_movement/stack/stack_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/data_movement/tilize/tilize_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/data_movement/tilize_with_val_padding/tilize_with_val_padding_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/data_movement/transpose/transpose_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/data_movement/unsqueeze/unsqueeze_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/data_movement/untilize/untilize_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/data_movement/untilize_with_unpadding/untilize_with_unpadding_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/data_movement/view/view_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/eltwise/binary/binary_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/eltwise/binary_backward/binary_backward_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/eltwise/complex/complex_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/eltwise/complex_unary/complex_unary_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/eltwise/complex_unary_backward/complex_unary_backward_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/eltwise/quantization/quantization_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/eltwise/ternary/ternary_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/eltwise/ternary_backward/ternary_backward_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/eltwise/unary/unary_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/eltwise/unary_backward/unary_backward_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/embedding/embedding_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/embedding_backward/embedding_backward_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/examples/example/example_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/examples/example_multiple_return/example_multiple_return_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/examples/examples_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/bcast_to/bcast_to_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/cnn/convert_to_chw/convert_to_chw_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/cnn/convert_to_hwc/convert_to_hwc_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/conv3d/conv3d_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/minimal_matmul/minimal_matmul_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/copy/typecast/typecast_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/dropout/dropout_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/adaptive_pool/adaptive_pools_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/experimental_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/matmul/attn_matmul/attn_matmul_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/matmul/group_attn_matmul/group_attn_matmul_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/paged_cache/paged_cache_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/plusone/plusone_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/reduction/fast_reduce_nc/fast_reduce_nc_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/reduction/fast_reduce_nc/fast_reduce_nc_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/reshape/view_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/slice_write/slice_write_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/padded_slice/padded_slice_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/ssm/hc_sum_reduce/hc_sum_reduce_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/ssm/prefix_scan/prefix_scan_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/ssm/repeat_and_interleave_eltwise_mul/repeat_and_interleave_eltwise_mul_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/transformer/all_reduce_create_qkv_heads/all_reduce_create_qkv_heads_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/transformer/concatenate_heads/concatenate_heads_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/transformer/create_qkv_heads/create_qkv_heads_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/transformer/create_qkv_heads_from_separate_tensors/create_qkv_heads_from_separate_tensors_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/transformer/nlp_concat_heads/nlp_concat_heads_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/transformer/nlp_concat_heads_decode/nlp_concat_heads_decode_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/transformer/nlp_concat_heads_boltz/nlp_concat_heads_boltz_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/transformer/nlp_create_qkv_heads/nlp_create_qkv_heads_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/transformer/nlp_create_qkv_heads_decode/nlp_create_qkv_heads_decode_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/transformer/nlp_create_qkv_heads_falcon7b/nlp_create_qkv_heads_falcon7b_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/transformer/nlp_create_qkv_heads_segformer/nlp_create_qkv_heads_segformer_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/transformer/nlp_create_qkv_heads_vit/nlp_create_qkv_heads_vit_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/transformer/nlp_create_qkv_heads_boltz/nlp_create_qkv_heads_boltz_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/transformer/nlp_kv_cache_load_slice/nlp_kv_cache_load_slice_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/transformer/fused_distributed_rmsnorm/rmsnorm_distributed_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/transformer/dit_layernorm_pre_all_gather/dit_layernorm_pre_all_gather_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/transformer/dit_layernorm_post_all_gather/dit_layernorm_post_all_gather_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/transformer/rotary_embedding/rotary_embedding_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/transformer/rotary_embedding_llama/rotary_embedding_llama_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/transformer/rotary_embedding_llama/rotary_embedding_llama_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/transformer/rotary_embedding_llama_fused_qk/rotary_embedding_llama_fused_qk_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/transformer/rotate_half/rotate_half_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/transformer/split_query_key_value_and_split_heads/split_query_key_value_and_split_heads_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/unary_backward/gelu_backward/gelu_backward_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/reduction/integral_image/intimg_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/full/full_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/full_like/full_like_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/generic/generic_op_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/index_fill/index_fill_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/kv_cache/kv_cache_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/loss/loss_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/matmul/matmul_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/moreh/moreh_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/moreh/moreh_abs_pow/moreh_abs_pow_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/moreh/moreh_adam/moreh_adam_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/moreh/moreh_adamw/moreh_adamw_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/moreh/moreh_arange/moreh_arange_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/moreh/moreh_bmm/moreh_bmm_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/moreh/moreh_bmm_backward/moreh_bmm_backward_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/moreh/moreh_clip_grad_norm/moreh_clip_grad_norm_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/moreh/moreh_cumsum/moreh_cumsum_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/moreh/moreh_dot/moreh_dot_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/moreh/moreh_dot/moreh_dot_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/moreh/moreh_dot_backward/moreh_dot_backward_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/moreh/moreh_fold/fold_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/moreh/moreh_getitem/moreh_getitem_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/moreh/moreh_group_norm/moreh_group_norm_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/moreh/moreh_group_norm_backward/moreh_group_norm_backward_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/moreh/moreh_layer_norm/moreh_layer_norm_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/moreh/moreh_layer_norm_backward/moreh_layer_norm_backward_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/moreh/moreh_linear/moreh_linear_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/moreh/moreh_linear_backward/moreh_linear_backward_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/moreh/moreh_matmul/moreh_matmul_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/moreh/moreh_matmul_backward/moreh_matmul_backward_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/moreh/moreh_mean/moreh_mean_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/moreh/moreh_mean_backward/moreh_mean_backward_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/moreh/moreh_nll_loss/moreh_nll_loss_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/moreh/moreh_nll_loss_backward/moreh_nll_loss_backward_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/moreh/moreh_nll_loss_unreduced_backward/moreh_nll_loss_unreduced_backward_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/moreh/moreh_norm/moreh_norm_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/moreh/moreh_norm_backward/moreh_norm_backward_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/moreh/moreh_sgd/moreh_sgd_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/moreh/moreh_softmax/moreh_softmax_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/moreh/moreh_softmax/moreh_softmax_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/moreh/moreh_softmax_backward/moreh_softmax_backward_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/moreh/moreh_sum/moreh_sum_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/moreh/moreh_sum_backward/moreh_sum_backward_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/normalization/batch_norm/batch_norm_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/normalization/groupnorm/groupnorm_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/normalization/layernorm/layernorm_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/normalization/layernorm_distributed/layernorm_distributed_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/normalization/normalization_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/normalization/rmsnorm/rmsnorm_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/normalization/rmsnorm/rmsnorm_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/normalization/rmsnorm_distributed/rmsnorm_distributed_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/normalization/softmax/softmax_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/pool/generic/generic_pools_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/pool/global_avg_pool/global_avg_pool_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/pool/grid_sample/grid_sample_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/pool/upsample/upsample_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/prefetcher/prefetcher/dram_prefetcher_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/prefetcher/prefetcher_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/reduction/reduction_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/reduction/argmax/argmax_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/reduction/accumulation/cumprod/cumprod_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/reduction/accumulation/cumsum/cumsum_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/reduction/accumulation/ema/ema_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/reduction/moe/moe_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/reduction/sampling/sampling_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/reduction/topk/topk_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/reduction/manual_seed/manual_seed_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/reduction/deepseek_grouped_gate/deepseek_grouped_gate_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/sliding_window/sliding_window_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/transformer/attention_softmax/attention_softmax_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/transformer/concatenate_heads/concatenate_heads_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/transformer/sdpa/sdpa_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/transformer/sdpa_decode/sdpa_decode_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/transformer/sdpa_windowed/sdpa_windowed_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/transformer/split_query_key_value_and_split_heads/split_query_key_value_and_split_heads_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/transformer/transformer_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/uniform/uniform_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/transformer/all_reduce_create_qkv_heads/all_reduce_create_qkv_heads_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/where/where_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/rand/rand_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/data_movement/sort/sort_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/data_movement/gather/gather_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/data_movement/gather/tosa/gather_tosa_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/test/hang_device/hang_device_operation_nanobind.cpp
)

set(CCL_EXPERIMENTAL_TTNN_SRCS_PYBIND
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/ccl/llama_all_gather_matmul_async/llama_all_gather_matmul_async_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/ccl/all_gather_async/all_gather_async_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/ccl/strided_all_gather_async/strided_all_gather_async_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/ccl/all_gather_concat_heads_fused/all_gather_concat_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/ccl/llama_reduce_scatter_matmul/rs_matmul_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/ccl/all_reduce_async/all_reduce_async_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/ccl/ccl_experimental_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/ccl/rms_allgather/rms_allgather_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/ccl/llama_reduce_scatter/llama_reduce_scatter_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/ccl/llama_reduce_scatter_create_heads/llama_reduce_scatter_create_heads_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/ccl/all_to_all_async/all_to_all_async_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/ccl/all_to_all_async_generic/all_to_all_async_generic_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/reduce_scatter_minimal_async_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/ccl/all_gather_matmul_async/all_gather_matmul_async_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/ccl/strided_all_gather_minimal_matmul_async/strided_all_gather_minimal_matmul_async_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/ccl/matmul_reduce_scatter_async/matmul_reduce_scatter_async_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/ccl/ring_attention_all_gather_async/ring_attention_all_gather_async_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/ccl/send_recv_async/send_async/send_async_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/ccl/send_recv_async/recv_async/recv_async_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/ccl/neighbor_pad_async/neighbor_pad_async_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/ccl/slice_reshard_async/slice_reshard_async_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/ccl/deepseek_minimal_broadcast/deepseek_minimal_broadcast_nanobind.cpp
)

set(TTNN_P2P_PYBIND ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/point_to_point/point_to_point_nanobind.cpp)

set(CCL_TTNN_SRCS_PYBIND
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/ccl/ccl_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/ccl/all_broadcast/all_broadcast_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/ccl/all_to_all_combine/all_to_all_combine_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/ccl/reduce_to_root/reduce_to_root_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/ccl/all_to_all_dispatch/all_to_all_dispatch_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/ccl/broadcast/broadcast_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/ccl/mesh_partition/mesh_partition_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/ccl/reduce_scatter/reduce_scatter_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/ccl/all_gather/all_gather_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/ccl/all_reduce/all_reduce_nanobind.cpp
)

set(DEBUG_TTNN_SRCS_PYBIND
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/debug/debug_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/debug/apply_device_delay_nanobind.cpp
)

#TODO : should be using nanobind_add_module, but right now it introduces many build problems
#nanobinds will always be built as a shared library
list(
    APPEND
    TTNN_SRC_PYBIND
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn-nanobind/__init__.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn-nanobind/activation.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn-nanobind/cluster.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn-nanobind/core.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn-nanobind/device.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn-nanobind/events.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn-nanobind/fabric.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn-nanobind/global_circular_buffer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn-nanobind/global_semaphore.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn-nanobind/mesh_socket.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn-nanobind/profiler.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn-nanobind/program_descriptors.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn-nanobind/pytensor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn-nanobind/reports.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn-nanobind/tensor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn-nanobind/types.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn-nanobind/operations/copy.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn-nanobind/operations/core.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn-nanobind/operations/creation.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn-nanobind/operations/trace.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn-nanobind/tensor_accessor_args.cpp
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
add_library(TTNN::TTNN ALIAS ttnncpp)
add_library(Metalium::TTNNCPP ALIAS ttnncpp) # backwards compatibility. FIXME: remove

target_precompile_headers(ttnncpp REUSE_FROM TT::CommonPCH)
TT_ENABLE_UNITY_BUILD(ttnncpp)

set_target_properties(
    ttnncpp
    PROPERTIES
        EXPORT_NAME
            TTNN
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
        INTERFACE_HEADER_SETS_TO_VERIFY
            api
)

file(GLOB_RECURSE ttnn_kernels cpp/ttnn/operations/copy/typecast/device/kernels/*)
target_sources(
    ttnncpp
    PUBLIC
        FILE_SET api
        TYPE HEADERS
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/api ${CMAKE_CURRENT_SOURCE_DIR}/cpp
        FILES
            api/tools/profiler/op_profiler.hpp
            api/ttnn/async_runtime.hpp
            api/ttnn/cluster.hpp
            api/ttnn/common/constants.hpp
            api/ttnn/common/guard.hpp
            api/ttnn/common/queue_id.hpp
            api/ttnn/config.hpp
            api/ttnn/core.hpp
            api/ttnn/decorators.hpp
            api/ttnn/device.hpp
            api/ttnn/device_operation.hpp
            api/ttnn/distributed/api.hpp
            api/ttnn/distributed/bidirectional_fabric_socket.hpp
            api/ttnn/distributed/create_socket.hpp
            api/ttnn/distributed/distributed_configs.hpp
            api/ttnn/distributed/distributed_nanobind.hpp
            api/ttnn/distributed/distributed_tensor.hpp
            api/ttnn/distributed/fabric_socket.hpp
            api/ttnn/distributed/host_ccl.hpp
            api/ttnn/distributed/isocket.hpp
            api/ttnn/distributed/mpi_socket.hpp
            api/ttnn/distributed/tensor_topology.hpp
            api/ttnn/distributed/types.hpp
            api/ttnn/events.hpp
            api/ttnn/global_circular_buffer.hpp
            api/ttnn/global_semaphore.hpp
            api/ttnn/graph/graph_argument_serializer.hpp
            api/ttnn/graph/graph_consts.hpp
            api/ttnn/graph/graph_operation_queries.hpp
            api/ttnn/graph/graph_processor.hpp
            api/ttnn/graph/graph_nanobind.hpp
            api/ttnn/graph/graph_query_op_constraints.hpp
            api/ttnn/graph/graph_query_op_runtime.hpp
            api/ttnn/graph/graph_trace_utils.hpp
            api/ttnn/graph/levelized_graph.hpp
            api/ttnn/mesh_device_operation_adapter.hpp
            api/ttnn/mesh_device_operation_utils.hpp
            api/ttnn/old_infra_device_operation.hpp
            api/ttnn/operation.hpp
            api/ttnn/operation_concepts.hpp
            api/ttnn/reports.hpp
            api/ttnn/run_operation.hpp
            api/ttnn/tensor/host_buffer/functions.hpp
            api/ttnn/tensor/layout/alignment.hpp
            api/ttnn/tensor/layout/layout.hpp
            api/ttnn/tensor/layout/page_config.hpp
            api/ttnn/tensor/layout/tensor_layout.hpp
            api/ttnn/tensor/memory_config/memory_config.hpp
            api/ttnn/tensor/serialization.hpp
            api/ttnn/tensor/shape/shape.hpp
            api/ttnn/tensor/storage.hpp
            api/ttnn/tensor/tensor.hpp
            api/ttnn/tensor/tensor_attributes.hpp
            api/ttnn/tensor/tensor_impl.hpp
            api/ttnn/tensor/tensor_spec.hpp
            api/ttnn/tensor/tensor_utils.hpp
            api/ttnn/tensor/to_string.hpp
            api/ttnn/tensor/types.hpp
            api/ttnn/tensor/unit_mesh/unit_mesh_utils.hpp
            api/ttnn/tensor/xtensor/conversion_utils.hpp
            api/ttnn/tensor/xtensor/partition.hpp
            api/ttnn/tensor/xtensor/xtensor_all_includes.hpp
            api/ttnn/types.hpp
            cpp/ttnn/operations/copy/typecast/typecast.hpp
            cpp/ttnn/operations/creation.hpp
            cpp/ttnn/operations/functions.hpp
            cpp/ttnn/operations/compute_throttle_utils.hpp
            cpp/ttnn/operations/cb_utils.hpp
            cpp/ttnn/operations/math.hpp
            cpp/ttnn/operations/trace.hpp
        FILE_SET kernels
        TYPE HEADERS
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
        FILES ${ttnn_kernels}
    PRIVATE
        # FIXME: Move these out to appropriate sub targets
        cpp/ttnn/operations/compute_throttle_utils.cpp
        cpp/ttnn/operations/trace.cpp
        cpp/ttnn/operations/ccl/sharding_addrgen_helper.cpp
        cpp/ttnn/operations/generic/generic_op.cpp
        cpp/ttnn/operations/generic/device/generic_op_program_factory.cpp
        cpp/ttnn/operations/generic/device/generic_op_device_operation.cpp
        cpp/ttnn/operations/data_movement/reshape_view/reshape_common.cpp
        cpp/ttnn/operations/experimental/ccl/rms_allgather/device/rms_allgather_device_operation.cpp
        cpp/ttnn/operations/experimental/ccl/rms_allgather/device/rms_allgather_program_factory.cpp
        cpp/ttnn/operations/experimental/ccl/rms_allgather/rms_allgather.cpp
        cpp/ttnn/operations/experimental/test/hang_device/hang_device_program_factory.cpp
        cpp/ttnn/operations/normalization/rmsnorm_distributed/rmsnorm_pre_all_gather.cpp
        cpp/ttnn/operations/normalization/rmsnorm_distributed/rmsnorm_post_all_gather.cpp
        cpp/ttnn/operations/copy/typecast/device/typecast_device_op.cpp
        cpp/ttnn/operations/copy/typecast/device/typecast_program_factory.cpp
        cpp/ttnn/operations/copy/typecast/device/typecast_sharded_program_factory.cpp
        cpp/ttnn/operations/copy/typecast/typecast.cpp
)

target_include_directories(
    ttnncpp
    PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/api>"
        # FIXME: decide what to do about this dir. Device includes complicate matters.
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/cpp>"
)

target_link_libraries(
    ttnncpp
    PUBLIC
        TT::Metalium
        xtensor
    PRIVATE
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
        TTNN::Ops::Experimental::AdaptivePool
        TTNN::Ops::Experimental::BcastTo
        TTNN::Ops::Experimental::CCL
        TTNN::Ops::Experimental::CNN
        TTNN::Ops::Experimental::Conv3d
        TTNN::Ops::Experimental::Copy
        TTNN::Ops::Experimental::Dropout
        TTNN::Ops::Experimental::Matmul
        TTNN::Ops::Experimental::PaddedSlice
        TTNN::Ops::Experimental::MinimalMatmul
        TTNN::Ops::Experimental::PagedCache
        TTNN::Ops::Experimental::PlusOne
        TTNN::Ops::Experimental::Reduction
        TTNN::Ops::Experimental::Reshape
        TTNN::Ops::Experimental::SliceWrite
        TTNN::Ops::Experimental::SSM
        TTNN::Ops::Experimental::Transformer
        TTNN::Ops::Experimental::UnaryBackward
        TTNN::Ops::Experimental::Where
        TTNN::Ops::Experimental::Test
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
        TTNN::Ops::Debug
        TTNN::Ops::PointToPoint
        TTNN::Ops::Pool
        TTNN::Ops::Normalization
        TTNN::Ops::Rand
)
if(TT_INSTALL)
    install(
        TARGETS
            ttnncpp
        LIBRARY
            DESTINATION ${CMAKE_INSTALL_LIBDIR}
            COMPONENT tar
    )
    install(
        TARGETS
            ttnncpp
        EXPORT TT-NN
        FILE_SET
        api
            COMPONENT ttnn-dev
        FILE_SET
        kernels
            DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium/ttnn
            COMPONENT ttnn-runtime
        LIBRARY
            COMPONENT ttnn-runtime
    )
endif()

if(WITH_PYTHON_BINDINGS)
    add_library(ttnn SHARED)
    add_library(TTNN::PYTHON ALIAS ttnn)
    add_library(Metalium::TTNN ALIAS ttnn) # backwards compatibility. FIXME: remove
    TT_ENABLE_UNITY_BUILD(ttnn)

    set_target_properties(
        ttnn
        PROPERTIES
            EXPORT_NAME
                Python
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
            ${DEBUG_TTNN_SRCS_PYBIND}
            ${TTNN_P2P_PYBIND}
    )
    target_include_directories(
        ttnn
        PUBLIC
            "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>"
            "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/cpp>"
    )

    # TODO: This block was removed in main. Figure out how to interop with
    # manylinux work
    # nanobind requires this to be invoked before nanobind is included
    if(CMAKE_VERSION VERSION_LESS 3.18)
        set(DEV_MODULE Development)
    else()
        set(DEV_MODULE Development.Module)
    endif()

    set(Python3_FIND_STRATEGY LOCATION)
    find_package(
        Python3
        REQUIRED
        COMPONENTS
            Interpreter
            ${DEV_MODULE}
        OPTIONAL_COMPONENTS
            Development.SABIModule
    )
    message(STATUS "Python3 include dirs: ${Python3_INCLUDE_DIRS}")

    # https://nanobind.readthedocs.io/en/latest/api_cmake.html#low-level-interface
    # -static Perform a static library build (without this suffix, a shared build is used)
    # -abi3   Perform a stable ABI build targeting Python v3.12+.
    # -ft     Perform a build that opts into the Python 3.13+ free-threaded behavior.

    set(NANOBIND_LIBNAME "nanobind-static-abi3")
    nanobind_build_library(${NANOBIND_LIBNAME})
    get_target_property(NANOBIND_INCLUDES ${NANOBIND_LIBNAME} INTERFACE_INCLUDE_DIRECTORIES)
    set_target_properties(
        ${NANOBIND_LIBNAME}
        PROPERTIES
            INTERFACE_SYSTEM_INCLUDE_DIRECTORIES
                "${NANOBIND_INCLUDES}"
    )
    add_library(TT::Nanobind ALIAS ${NANOBIND_LIBNAME})

    # GCC warns about visibility mismatches when using nanobind with -fvisibility=default
    # This affects code that uses nanobind types (not just nanobind headers themselves)
    target_compile_options(${NANOBIND_LIBNAME} INTERFACE "$<$<CXX_COMPILER_ID:GNU>:-Wno-attributes>")

    # find_package(Python3) required for here.
    # nanobind doesn't build otherwise.
    # PUBLIC on the headers also required
    target_include_directories(${NANOBIND_LIBNAME} PUBLIC ${Python3_INCLUDE_DIRS})

    target_link_libraries(
        ttnn
        PUBLIC
            "$<BUILD_INTERFACE:TTNN::CPP>"
        PRIVATE
            TT::Nanobind
            Boost::algorithm
            TT::Metalium
    )
    # .. enable size optimizations
    nanobind_opt_size(ttnn)

    # .. enable link time optimization
    nanobind_lto(ttnn)

    # .. set the default symbol visibility to 'hidden'
    #nanobind_set_visibility(ttnn)

    # .. strip unneeded symbols and debug info from the binary (only active in release builds)
    #nanobind_strip(ttnn)

    # .. disable the stack protector
    #nanobind_disable_stack_protector(ttnn)

    # .. set the Python extension suffix
    #nanobind_extension(ttnn)

    # .. set important compilation flags
    nanobind_compile_options(ttnn)

    # .. set important linker flags
    nanobind_link_options(ttnn)

    if(TT_INSTALL)
        install(TARGETS ttnn EXPORT TT-NN LIBRARY COMPONENT ttnn-runtime)
    endif()

    # Install .so into src files for nanobind implementation
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
if(TT_INSTALL)
    install(EXPORT TT-NN NAMESPACE TTNN:: DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/tt-nn COMPONENT ttnn-dev)
endif()

if(TTNN_BUILD_TESTS)
    add_subdirectory(test)
endif()

set(FixmeOpAPIDir ${CMAKE_CURRENT_SOURCE_DIR}/cpp)
set(FixmeOpIncDirs
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp
)

add_subdirectory(cpp/ttnn/deprecated)
add_subdirectory(cpp/ttnn/operations/bernoulli)
add_subdirectory(cpp/ttnn/operations/ccl)
add_subdirectory(cpp/ttnn/operations/conv)
add_subdirectory(cpp/ttnn/operations/core)
add_subdirectory(cpp/ttnn/operations/data_movement)
add_subdirectory(cpp/ttnn/operations/debug)
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
add_subdirectory(cpp/ttnn/operations/experimental/adaptive_pool)
add_subdirectory(cpp/ttnn/operations/experimental/bcast_to)
add_subdirectory(cpp/ttnn/operations/experimental/ccl)
add_subdirectory(cpp/ttnn/operations/experimental/cnn)
add_subdirectory(cpp/ttnn/operations/experimental/conv3d)
add_subdirectory(cpp/ttnn/operations/experimental/copy)
add_subdirectory(cpp/ttnn/operations/experimental/dropout)
add_subdirectory(cpp/ttnn/operations/experimental/matmul)
add_subdirectory(cpp/ttnn/operations/experimental/minimal_matmul)
add_subdirectory(cpp/ttnn/operations/experimental/padded_slice)
add_subdirectory(cpp/ttnn/operations/experimental/paged_cache)
add_subdirectory(cpp/ttnn/operations/experimental/plusone)
add_subdirectory(cpp/ttnn/operations/experimental/reduction)
add_subdirectory(cpp/ttnn/operations/experimental/reshape)
add_subdirectory(cpp/ttnn/operations/experimental/slice_write)
add_subdirectory(cpp/ttnn/operations/experimental/ssm)
add_subdirectory(cpp/ttnn/operations/experimental/test)
add_subdirectory(cpp/ttnn/operations/experimental/transformer)
add_subdirectory(cpp/ttnn/operations/experimental/unary_backward)
add_subdirectory(cpp/ttnn/operations/experimental/where)
add_subdirectory(cpp/ttnn/operations/full)
add_subdirectory(cpp/ttnn/operations/full_like)
add_subdirectory(cpp/ttnn/operations/index_fill)
add_subdirectory(cpp/ttnn/operations/kernel_helper_functions)
add_subdirectory(cpp/ttnn/operations/kv_cache)
add_subdirectory(cpp/ttnn/operations/loss)
add_subdirectory(cpp/ttnn/operations/matmul)
add_subdirectory(cpp/ttnn/operations/moreh)
add_subdirectory(cpp/ttnn/operations/normalization)
add_subdirectory(cpp/ttnn/operations/point_to_point)
add_subdirectory(cpp/ttnn/operations/pool)
add_subdirectory(cpp/ttnn/operations/prefetcher)
add_subdirectory(cpp/ttnn/operations/rand)
add_subdirectory(cpp/ttnn/operations/reduction)
add_subdirectory(cpp/ttnn/operations/sliding_window)
add_subdirectory(cpp/ttnn/operations/transformer)
add_subdirectory(cpp/ttnn/operations/uniform)
add_subdirectory(examples)
