// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <optional>

#include "ttnn/operations/experimental/reduction/deepseek_moe_fast_reduce_nc_fused/device/deepseek_moe_fast_reduce_nc_fused_device_operation.hpp"

#include "ttnn/operations/moreh/moreh_helper_functions.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim {

DeepseekMoEFastReduceNCFusedDeviceOperation::program_factory_t
DeepseekMoEFastReduceNCFusedDeviceOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return DeepseekMoEFastReduceNCFusedMeshWorkloadFactory{};
}

void DeepseekMoEFastReduceNCFusedDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const ttnn::Tensor& input_tensor = tensor_args.input_tensor;
    TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "Input tensor must be on device");
    TT_FATAL(input_tensor.buffer() != nullptr, "Input tensor must have a buffer");

    const ttnn::Tensor& scores_tensor = tensor_args.scores_tensor;
    TT_FATAL(scores_tensor.storage_type() == StorageType::DEVICE, "Scores tensor must be on device");
    TT_FATAL(scores_tensor.buffer() != nullptr, "Scores tensor must have a buffer");

    const ttnn::Tensor& expert_indices_tensor = tensor_args.expert_indices_tensor;
    TT_FATAL(expert_indices_tensor.storage_type() == StorageType::DEVICE, "Expert indices tensor must be on device");
    TT_FATAL(expert_indices_tensor.buffer() != nullptr, "Expert indices tensor must have a buffer");

    const ttnn::Tensor& expert_mapping_tensor = tensor_args.expert_mapping_tensor;
    TT_FATAL(expert_mapping_tensor.storage_type() == StorageType::DEVICE, "Expert mapping tensor must be on device");
    TT_FATAL(expert_mapping_tensor.buffer() != nullptr, "Expert mapping tensor must have a buffer");

    TT_FATAL(
        operation_attributes.cluster_axis <= 1,
        "cluster_axis must be 0 or 1 (got {})",
        operation_attributes.cluster_axis);
}

void DeepseekMoEFastReduceNCFusedDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate_on_program_cache_hit(operation_attributes, tensor_args);

    const ttnn::Tensor& input_tensor = tensor_args.input_tensor;
    const auto& input_shape = input_tensor.padded_shape();
    const auto& input_logical_shape = input_tensor.logical_shape();
    const uint32_t input_rank = input_shape.rank();
    const uint32_t reduction_dim = operation_attributes.reduce_dim;

    const uint32_t split_dim = input_rank - 1;

    // validate tensor
    operations::check_tensor(
        input_tensor, "DeepseekMoEFastReduceNCFused", "input", {DataType::BFLOAT16, DataType::BFLOAT8_B});
    TT_FATAL(input_tensor.layout() == ttnn::Layout::TILE, "input tensor must be tiled");

    // validate tile shape: the program factory sizes every circular buffer from a buffer page size,
    // while the JIT unpack/pack tile size is generated from the CB data format and the *default*
    // 32x32 tile. A non-default input tile would make those two disagree, so pin it here.
    const auto input_tile = input_tensor.tensor_spec().tile();
    TT_FATAL(
        input_tile.get_height() == tt::constants::TILE_HEIGHT && input_tile.get_width() == tt::constants::TILE_WIDTH,
        "input tensor must use a {}x{} tile, but has {}x{}",
        tt::constants::TILE_HEIGHT,
        tt::constants::TILE_WIDTH,
        input_tile.get_height(),
        input_tile.get_width());

    // validate split size before it is used as a divisor
    TT_FATAL(operation_attributes.split_size > 0, "split_size must be greater than 0");
    TT_FATAL(
        operation_attributes.split_size <= input_shape[-1],
        "split_size ({}) must not exceed the input tensor padded width ({})",
        operation_attributes.split_size,
        input_shape[-1]);
    TT_FATAL(
        operation_attributes.split_size <= input_logical_shape[-1],
        "split_size ({}) must not exceed the input tensor logical width ({})",
        operation_attributes.split_size,
        input_logical_shape[-1]);

    const uint32_t num_output_tensors = input_shape[-1] / operation_attributes.split_size;

    // validate rank
    TT_FATAL(input_rank > 2, "input tensor rank must be greater than 2, but has {}", input_rank);

    // validate reduction dim
    TT_FATAL(
        reduction_dim <= input_rank - 3,
        "reduction dim must be between 0 and {}, but has {}",
        input_rank - 3,
        reduction_dim);

    // validate split dim
    uint32_t split_dim_size = input_shape[split_dim];
    TT_FATAL(
        split_dim_size % (num_output_tensors * tt::constants::TILE_WIDTH) == 0,
        "input tensor width must be divisible by {}",
        num_output_tensors * tt::constants::TILE_WIDTH);

    // The number of output tensors that actually gets created is derived from the *logical* width
    // (see compute_output_specs / create_output_tensors), while the program factory slices the
    // *padded* width: slice_Wt = (padded_W / TILE_WIDTH) / output_tensors.size(). The shared writer
    // kernel derives its slice id from the padded width and uses it to index a fixed-size array of
    // output_tensors.size() accessors, so the padded width in tiles must be an exact multiple of the
    // number of created output tensors. Otherwise the last tiles of a row index one past the end of
    // that array, pulling a garbage accessor + get_noc_addr function pointer off the stack.
    const uint32_t num_created_output_tensors = input_logical_shape[-1] / operation_attributes.split_size;
    const uint32_t input_tensor_Wt = input_shape[-1] / tt::constants::TILE_WIDTH;
    TT_FATAL(
        input_tensor_Wt % num_created_output_tensors == 0,
        "input tensor width in tiles ({}, from padded width {}) must be divisible by the number of output "
        "tensors ({}, from logical width {} / split_size {})",
        input_tensor_Wt,
        input_shape[-1],
        num_created_output_tensors,
        input_logical_shape[-1],
        operation_attributes.split_size);

    // Same invariant seen from the page-id side: the writer walks output pages as
    // (input_row_index * slice_Wt + intra_slice_offset), which only stays inside an output buffer if a
    // slice is exactly as wide (in tiles) as an output tensor.
    const uint32_t output_tensor_Wt =
        compute_output_specs(operation_attributes, tensor_args).padded_shape()[-1] / tt::constants::TILE_WIDTH;
    TT_FATAL(
        output_tensor_Wt == input_tensor_Wt / num_created_output_tensors,
        "output tensor width in tiles ({}) must equal the input slice width in tiles ({})",
        output_tensor_Wt,
        input_tensor_Wt / num_created_output_tensors);

    // validate tensor layout
    const ttnn::Tensor& scores_tensor = tensor_args.scores_tensor;
    operations::check_tensor(
        scores_tensor, "DeepseekMoEFastReduceNCFused", "scores", {DataType::BFLOAT16}, ttnn::Layout::ROW_MAJOR);

    // scores shape: [tokens, 1, seq, experts_k] where experts_k == reduction_dim_size
    const auto& scores_shape = scores_tensor.logical_shape();
    const uint32_t scores_rank = scores_shape.rank();
    TT_FATAL(scores_rank == 4, "scores tensor must be rank 4, but has {}", scores_rank);
    TT_FATAL(input_rank == 4, "input tensor must be rank 4, but has {}", input_rank);

    const uint32_t reduction_dim_size = input_shape[reduction_dim];
    const uint32_t num_shared_experts_val = operation_attributes.num_shared_experts;
    TT_FATAL(
        num_shared_experts_val <= reduction_dim_size,
        "num_shared_experts ({}) must not exceed reduction dim size ({})",
        num_shared_experts_val,
        reduction_dim_size);
    const uint32_t num_routed_experts = reduction_dim_size - num_shared_experts_val;
    TT_FATAL(
        scores_shape[-1] == num_routed_experts,
        "scores last dim ({}) must equal num_routed_experts ({} = reduction_dim_size {} - num_shared_experts {})",
        scores_shape[-1],
        num_routed_experts,
        reduction_dim_size,
        num_shared_experts_val);

    const uint32_t num_tokens = input_shape[-2];
    TT_FATAL(
        (scores_shape[0] > num_tokens - tt::constants::TILE_WIDTH) && (scores_shape[0] <= num_tokens),
        "scores dim 0 (tokens in slice = {}) must be between {} and {} for the current fused kernel",
        scores_shape[0],
        num_tokens - tt::constants::TILE_WIDTH + 1,
        num_tokens);
    // scores_shape[0] is passed to the reader as its num_tokens compile-time arg, and the reader
    // tilizes the scores into exactly one 32x32 tile per expert (cb_scores is sized
    // reduction_dim_size * scores_tile_size). Token t is stored at element (t - 16) * 16 of face 2, so
    // anything past TILE_HEIGHT tokens writes outside the expert's tile - and past the end of
    // cb_scores altogether on the last expert.
    TT_FATAL(
        scores_shape[0] <= tt::constants::TILE_HEIGHT,
        "scores dim 0 (tokens in slice = {}) must not exceed the tile height ({}); the fused reader builds "
        "exactly one score tile per expert",
        scores_shape[0],
        tt::constants::TILE_HEIGHT);
}

tt::tt_metal::TensorSpec DeepseekMoEFastReduceNCFusedDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const uint32_t reduction_dim = operation_attributes.reduce_dim;
    const tt::tt_metal::MemoryConfig& output_memory_config = operation_attributes.output_memory_config;
    const ttnn::Tensor& input_tensor = tensor_args.input_tensor;
    const auto& input_shape = input_tensor.logical_shape();

    const uint32_t num_output_tensors = input_shape[-1] / operation_attributes.split_size;
    // Guard the divisor: split_size > logical width yields zero output tensors, which would divide by
    // zero here (and in validate_on_program_cache_miss) rather than reporting a usable error.
    TT_FATAL(
        num_output_tensors > 0,
        "split_size ({}) must not exceed the input tensor logical width ({})",
        operation_attributes.split_size,
        input_shape[-1]);
    const uint32_t split_dim = input_shape.rank() - 1;

    auto output_shape = input_tensor.logical_shape();
    output_shape[reduction_dim] = 1;  // keepdim = true
    output_shape[split_dim] /= num_output_tensors;

    return tt::tt_metal::TensorSpec(
        output_shape,
        operations::TensorLayout(input_tensor.dtype(), tt::tt_metal::PageConfig(Layout::TILE), output_memory_config));
}

std::vector<ttnn::Tensor> DeepseekMoEFastReduceNCFusedDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const ttnn::Tensor& input_tensor = tensor_args.input_tensor;

    const tt::tt_metal::TensorSpec& output_tensor_spec = compute_output_specs(operation_attributes, tensor_args);

    const uint32_t num_output_tensors = input_tensor.logical_shape()[-1] / operation_attributes.split_size;
    std::vector<ttnn::Tensor> output_tensors(num_output_tensors);
    for (uint32_t i = 0; i < num_output_tensors; ++i) {
        output_tensors[i] = create_device_tensor(output_tensor_spec, input_tensor.device());
    }

    return output_tensors;
}

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

std::vector<ttnn::Tensor> deepseek_moe_fast_reduce_nc_fused(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& scores_tensor,
    const ttnn::Tensor& expert_indices_tensor,
    const ttnn::Tensor& expert_mapping_tensor,
    uint32_t reduce_dim,
    uint64_t split_size,
    uint32_t cluster_axis,
    const tt::tt_metal::MemoryConfig& output_memory_config,
    uint32_t num_shared_experts,
    float shared_expert_scale,
    const ttnn::DeviceComputeKernelConfig& compute_kernel_config) {
    using OperationType = ttnn::experimental::prim::DeepseekMoEFastReduceNCFusedDeviceOperation;

    return ttnn::device_operation::launch<OperationType>(
        OperationType::operation_attributes_t{
            reduce_dim,
            split_size,
            cluster_axis,
            output_memory_config,
            num_shared_experts,
            shared_expert_scale,
            compute_kernel_config},
        OperationType::tensor_args_t{input_tensor, scores_tensor, expert_indices_tensor, expert_mapping_tensor});
}

}  // namespace ttnn::prim
