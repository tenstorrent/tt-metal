// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <utility>

#include "ttnn/tensor/types.hpp"
#include "prefill_combine_device_operation.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

namespace ttnn::operations::experimental::deepseek::prefill_combine {

void PrefillCombineDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {

    // Validate layouts
    TT_FATAL(
        tensor_args.dispatched_buffer.layout() == tt::tt_metal::Layout::ROW_MAJOR,
        "Dispatched buffer must be ROW_MAJOR layout");
    TT_FATAL(
        tensor_args.dispatched_metadata.layout() == tt::tt_metal::Layout::ROW_MAJOR,
        "Dispatched metadata must be ROW_MAJOR layout");
    TT_FATAL(
        tensor_args.experts_tok_counter.layout() == tt::tt_metal::Layout::ROW_MAJOR,
        "Experts token counter must be ROW_MAJOR layout");

    // Validate dtypes
    TT_FATAL(
        tensor_args.dispatched_buffer.dtype() == DataType::BFLOAT16,
        "Dispatched buffer must be BFLOAT16, got {}",
        tensor_args.dispatched_buffer.dtype());
    TT_FATAL(
        tensor_args.dispatched_metadata.dtype() == DataType::INT32,
        "Dispatched metadata must be INT32, got {}",
        tensor_args.dispatched_metadata.dtype());
    TT_FATAL(
        tensor_args.experts_tok_counter.dtype() == DataType::INT32,
        "Experts token counter must be INT32, got {}",
        tensor_args.experts_tok_counter.dtype());

    // Validate output memory config
    TT_FATAL(
        !operation_attributes.output_mem_config.is_sharded(),
        "Output memory config must be DRAM interleaved, not sharded");

    // Validate tensor shapes are compatible
    // Dispatch outputs are 5D: (per_device_batch, 1, experts_per_chip, max_dispatched_tokens, hidden_dim/metadata_len)
    // Counter is 2D: (per_device_batch, experts_per_chip)
    auto dispatched_shape = tensor_args.dispatched_buffer.tensor_spec().logical_shape();
    auto metadata_shape = tensor_args.dispatched_metadata.tensor_spec().logical_shape();
    auto counter_shape = tensor_args.experts_tok_counter.tensor_spec().logical_shape();

    TT_FATAL(
        dispatched_shape[0] == metadata_shape[0] && dispatched_shape[0] == counter_shape[0],
        "First dimension (per_device_batch) must match across all input tensors");
    TT_FATAL(
        dispatched_shape[2] == metadata_shape[2] && dispatched_shape[2] == counter_shape[1],
        "experts_per_chip dimension must match: dispatched[2]={}, metadata[2]={}, counter[1]={}",
        dispatched_shape[2],
        metadata_shape[2],
        counter_shape[1]);
}

void PrefillCombineDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& /*tensor_args*/) {
    // Empty for now
}

PrefillCombineDeviceOperation::spec_return_value_t PrefillCombineDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {

    // Get input shape to extract hidden_dim
    auto dispatched_shape = tensor_args.dispatched_buffer.tensor_spec().logical_shape();
    uint32_t per_device_batch = dispatched_shape[0];  // Should be 1 for sharded input
    uint32_t hidden_dim = dispatched_shape[-1];

    // Output shape: (per_device_batch, 1, seq_len_per_chip, num_experts_per_tok, hidden_dim)
    // The extra dimension (1) allows proper composition across 2D mesh
    // For sharded input on dim 0, per-device shape is (1, 1, seq_len_per_chip, num_experts_per_tok, hidden_dim)
    auto output_shape = ttnn::Shape(
        {per_device_batch,
         1,
         operation_attributes.seq_len_per_chip,
         operation_attributes.num_experts_per_tok,
         hidden_dim});

    // Memory config and layout
    auto mem_config = operation_attributes.output_mem_config;
    auto layout = tt::tt_metal::Layout::ROW_MAJOR;

    // Create TensorSpec with BFLOAT16 dtype (output of expert computations)
    auto output_spec = TensorSpec(
        Shape(output_shape),
        tt::tt_metal::TensorLayout(DataType::BFLOAT16, tt::tt_metal::PageConfig(layout), mem_config));

    return output_spec;
}

PrefillCombineDeviceOperation::tensor_return_value_t PrefillCombineDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {

    auto output_spec = compute_output_specs(operation_attributes, tensor_args);
    return create_device_tensor(output_spec, tensor_args.dispatched_buffer.device());
}

}  // namespace ttnn::operations::experimental::deepseek::prefill_combine

namespace ttnn::prim {
ttnn::Tensor prefill_combine(
    const ttnn::Tensor& dispatched_buffer,
    const ttnn::Tensor& dispatched_metadata,
    const ttnn::Tensor& experts_tok_counter,
    uint32_t num_chips,
    uint32_t experts_per_chip,
    uint32_t num_experts_per_tok,
    uint32_t seq_len_per_chip,
    std::optional<uint32_t> axis,
    uint32_t num_links,
    tt::tt_fabric::Topology topology,
    const ttnn::MemoryConfig& memory_config,
    const CoreRangeSet& worker_core_range_set) {

    using OperationType = ttnn::operations::experimental::deepseek::prefill_combine::PrefillCombineDeviceOperation;
    return ttnn::device_operation::launch<OperationType>(
        OperationType::operation_attributes_t{
            .num_chips = num_chips,
            .experts_per_chip = experts_per_chip,
            .num_experts_per_tok = num_experts_per_tok,
            .seq_len_per_chip = seq_len_per_chip,
            .axis = axis,
            .num_links = num_links,
            .topology = topology,
            .output_mem_config = memory_config,
            .worker_core_range_set = worker_core_range_set},
        OperationType::tensor_args_t{
            .dispatched_buffer = dispatched_buffer,
            .dispatched_metadata = dispatched_metadata,
            .experts_tok_counter = experts_tok_counter});
}
}  // namespace ttnn::prim
