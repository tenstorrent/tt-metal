// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <utility>

#include "ttnn/tensor/types.hpp"
#include "combine_device_operation.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::combine {

void CombineDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    // Validate layouts
    TT_FATAL(
        tensor_args.dispatched_buffer.layout() == tt::tt_metal::Layout::TILE ||
            tensor_args.dispatched_buffer.layout() == tt::tt_metal::Layout::ROW_MAJOR,
        "Dispatched buffer must be TILE_LAYOUT or ROW_MAJOR layout");
    TT_FATAL(
        tensor_args.dispatched_metadata.layout() == tt::tt_metal::Layout::ROW_MAJOR,
        "Dispatched metadata must be ROW_MAJOR layout");
    TT_FATAL(
        tensor_args.expert_token_counts.layout() == tt::tt_metal::Layout::ROW_MAJOR,
        "Experts token counter must be ROW_MAJOR layout");
    TT_FATAL(
        tensor_args.expert_region_offsets.layout() == tt::tt_metal::Layout::ROW_MAJOR,
        "Expert region offsets must be ROW_MAJOR layout");

    // Validate dtypes
    TT_FATAL(
        tensor_args.dispatched_buffer.dtype() == DataType::BFLOAT16 ||
            (tensor_args.dispatched_buffer.dtype() == DataType::BFLOAT8_B &&
             tensor_args.dispatched_buffer.layout() == tt::tt_metal::Layout::TILE),
        "Dispatched buffer must be BFLOAT16 or BFLOAT8_B with TILE layout, got {}",
        tensor_args.dispatched_buffer.dtype());
    TT_FATAL(
        tensor_args.dispatched_metadata.dtype() == DataType::INT32,
        "Dispatched metadata must be INT32, got {}",
        tensor_args.dispatched_metadata.dtype());
    TT_FATAL(
        tensor_args.expert_token_counts.dtype() == DataType::INT32 ||
            tensor_args.expert_token_counts.dtype() == DataType::UINT32,
        "Experts token counter must be INT32 or UINT32, got {}",
        tensor_args.expert_token_counts.dtype());
    TT_FATAL(
        tensor_args.expert_region_offsets.dtype() == DataType::INT32 ||
            tensor_args.expert_region_offsets.dtype() == DataType::UINT32,
        "Expert region offsets must be INT32 or UINT32, got {}",
        tensor_args.expert_region_offsets.dtype());
    TT_FATAL(
        tensor_args.expert_region_offsets.tensor_spec().logical_shape() ==
            tensor_args.expert_token_counts.tensor_spec().logical_shape(),
        "expert_region_offsets shape {} must match expert_token_counts shape {}",
        tensor_args.expert_region_offsets.tensor_spec().logical_shape(),
        tensor_args.expert_token_counts.tensor_spec().logical_shape());

    // Validate output memory config
    TT_FATAL(
        !operation_attributes.output_mem_config.is_sharded(),
        "Output memory config must be interleaved (L1 or DRAM), not sharded");

    // Validate tensor shapes are compatible
    // Dispatch outputs are 4D: (per_device_batch, 1, max_dispatch_buffer_token_size, hidden_dim/metadata_len)
    // Counter is 3D: (num_dispatch_groups, per_device_batch, num_routed_experts)
    auto dispatched_shape = tensor_args.dispatched_buffer.tensor_spec().logical_shape();
    auto metadata_shape = tensor_args.dispatched_metadata.tensor_spec().logical_shape();
    auto counter_shape = tensor_args.expert_token_counts.tensor_spec().logical_shape();

    TT_FATAL(
        dispatched_shape[0] == metadata_shape[0] && dispatched_shape[0] == counter_shape[0],
        "First dimension (per_device_batch) must match across all input tensors");
    TT_FATAL(
        dispatched_shape[2] == metadata_shape[2],
        "Flat buffer dim must match: dispatched[2]={} vs metadata[2]={}",
        dispatched_shape[2],
        metadata_shape[2]);
    TT_FATAL(
        counter_shape[-1] % operation_attributes.experts_per_chip == 0,
        "counter last dim (num_routed_experts={}) must be divisible by experts_per_chip={}",
        counter_shape[-1],
        operation_attributes.experts_per_chip);
}

void CombineDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& /*tensor_args*/) {
    // Empty for now
}

CombineDeviceOperation::spec_return_value_t CombineDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    // Get input shape to extract hidden_dim
    auto dispatched_shape = tensor_args.dispatched_buffer.tensor_spec().logical_shape();
    uint32_t hidden_dim = dispatched_shape[-1];

    // Output shape: (1, 1, seq_len_per_chip, num_experts_per_tok, hidden_dim)
    auto output_shape = ttnn::Shape(
        {1, 1, operation_attributes.seq_len_per_chip, operation_attributes.num_experts_per_tok, hidden_dim});

    // Memory config and layout
    auto mem_config = operation_attributes.output_mem_config;
    auto layout = tt::tt_metal::Layout::ROW_MAJOR;

    // Create TensorSpec with BFLOAT16 dtype (output of expert computations)
    auto output_spec = TensorSpec(
        Shape(output_shape),
        tt::tt_metal::TensorLayout(DataType::BFLOAT16, tt::tt_metal::PageConfig(layout), mem_config));

    return output_spec;
}

CombineDeviceOperation::tensor_return_value_t CombineDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    auto output_spec = compute_output_specs(operation_attributes, tensor_args);
    return create_device_tensor(output_spec, tensor_args.dispatched_buffer.device());
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::combine

namespace ttnn::prim {
ttnn::Tensor prefill_combine(
    const ttnn::Tensor& dispatched_buffer,
    const ttnn::Tensor& dispatched_metadata,
    const ttnn::Tensor& expert_token_counts,
    const ttnn::Tensor& expert_region_offsets,
    uint32_t dispatch_group_size,
    uint32_t experts_per_chip,
    uint32_t num_experts_per_tok,
    uint32_t seq_len_per_chip,
    std::optional<uint32_t> axis,
    uint32_t num_links,
    tt::tt_fabric::Topology topology,
    const ttnn::MemoryConfig& memory_config,
    const CoreRangeSet& worker_core_range_set,
    bool init_zeros,
    bool use_l1_small_for_semaphores) {
    using OperationType = ttnn::operations::experimental::deepseek_prefill::combine::CombineDeviceOperation;
    return ttnn::device_operation::launch<OperationType>(
        OperationType::operation_attributes_t{
            .dispatch_group_size = dispatch_group_size,
            .experts_per_chip = experts_per_chip,
            .num_experts_per_tok = num_experts_per_tok,
            .seq_len_per_chip = seq_len_per_chip,
            .axis = axis,
            .num_links = num_links,
            .topology = topology,
            .output_mem_config = memory_config,
            .worker_core_range_set = worker_core_range_set,
            .init_zeros = init_zeros,
            .use_l1_small_for_semaphores = use_l1_small_for_semaphores},
        OperationType::tensor_args_t{
            .dispatched_buffer = dispatched_buffer,
            .dispatched_metadata = dispatched_metadata,
            .expert_token_counts = expert_token_counts,
            .expert_region_offsets = expert_region_offsets});
}
}  // namespace ttnn::prim
