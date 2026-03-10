// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <utility>

#include "ttnn/tensor/types.hpp"
#include "prefill_dispatch_combined_device_operation.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include <tt-metalium/hal.hpp>

namespace ttnn::operations::experimental::deepseek::prefill_dispatch_combined {

void PrefillDispatchCombinedDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    TT_FATAL(
        tensor_args.input_tensor.layout() == tt::tt_metal::Layout::ROW_MAJOR, "Input tensor must be ROW_MAJOR layout");
    TT_FATAL(
        tensor_args.weights_tensor.layout() == tt::tt_metal::Layout::ROW_MAJOR,
        "Weights tensor must be ROW_MAJOR layout");
    TT_FATAL(
        tensor_args.indices_tensor.layout() == tt::tt_metal::Layout::ROW_MAJOR,
        "Indices tensor must be ROW_MAJOR layout");
    TT_FATAL(
        tensor_args.chip_to_n_routed_expert_offset_tensor.layout() == tt::tt_metal::Layout::ROW_MAJOR,
        "Chip to expert offset tensor must be ROW_MAJOR layout");

    TT_FATAL(
        tensor_args.input_tensor.dtype() == DataType::BFLOAT16,
        "Input tensor must be BFLOAT16, got {}",
        tensor_args.input_tensor.dtype());
    TT_FATAL(
        tensor_args.weights_tensor.dtype() == DataType::BFLOAT16,
        "Weights tensor must be BFLOAT16, got {}",
        tensor_args.weights_tensor.dtype());
    TT_FATAL(
        tensor_args.indices_tensor.dtype() == DataType::INT32 || tensor_args.indices_tensor.dtype() == DataType::UINT32,
        "Indices tensor must be INT32 or UINT32, got {}",
        tensor_args.indices_tensor.dtype());
    TT_FATAL(
        tensor_args.chip_to_n_routed_expert_offset_tensor.dtype() == DataType::INT32,
        "Chip to expert offset tensor must be INT32, got {}",
        tensor_args.chip_to_n_routed_expert_offset_tensor.dtype());

    TT_FATAL(
        !operation_attributes.output_mem_config.is_sharded(),
        "Output memory config must be DRAM interleaved, not sharded");
}

void PrefillDispatchCombinedDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& /*tensor_args*/) {}

PrefillDispatchCombinedDeviceOperation::spec_return_value_t
PrefillDispatchCombinedDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    uint32_t experts_per_chip = operation_attributes.experts_per_chip;
    uint32_t metadata_len = operation_attributes.metadata_len;
    uint32_t max_dispatched_tokens_per_expert = operation_attributes.max_dispatched_tokens_per_expert;

    auto input_shape = tensor_args.input_tensor.tensor_spec().logical_shape();
    uint32_t per_device_batch = input_shape[0];
    uint32_t hidden_dim = input_shape[-1];

    // Compute padded metadata width in bfloat16 units
    const auto l1_alignment = tt::tt_metal::hal::get_l1_alignment();
    uint32_t metadata_bytes = metadata_len * sizeof(int32_t);
    uint32_t padded_metadata_bytes = tt::round_up(metadata_bytes, l1_alignment);
    uint32_t padded_metadata_bf16 = padded_metadata_bytes / sizeof(uint16_t);
    uint32_t combined_width = padded_metadata_bf16 + hidden_dim;

    auto mem_config = operation_attributes.output_mem_config;
    auto layout = tt::tt_metal::Layout::ROW_MAJOR;

    auto combined_shape =
        ttnn::Shape({per_device_batch, experts_per_chip, max_dispatched_tokens_per_expert, combined_width});
    auto counter_shape = ttnn::Shape({per_device_batch, experts_per_chip});

    auto combined_spec = TensorSpec(
        Shape(combined_shape),
        tt::tt_metal::TensorLayout(DataType::BFLOAT16, tt::tt_metal::PageConfig(layout), mem_config));

    auto counter_spec = TensorSpec(
        Shape(counter_shape),
        tt::tt_metal::TensorLayout(DataType::INT32, tt::tt_metal::PageConfig(layout), mem_config));

    return {combined_spec, counter_spec};
}

PrefillDispatchCombinedDeviceOperation::topology_return_value_t
PrefillDispatchCombinedDeviceOperation::compute_output_topologies(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input_tensor;
    const auto& input_topology = input_tensor.tensor_topology();

    auto output_topology = tt::tt_metal::TensorTopology(
        input_topology.distribution_shape(), input_topology.placements(), input_topology.mesh_coords());

    return {output_topology, output_topology};
}

PrefillDispatchCombinedDeviceOperation::tensor_return_value_t
PrefillDispatchCombinedDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    auto output_spec = compute_output_specs(operation_attributes, tensor_args);

    auto combined_tensor = create_device_tensor(output_spec[0], tensor_args.input_tensor.device());
    auto experts_counter_tensor = create_device_tensor(output_spec[1], tensor_args.input_tensor.device());
    return {combined_tensor, experts_counter_tensor};
}

}  // namespace ttnn::operations::experimental::deepseek::prefill_dispatch_combined

namespace ttnn::prim {
ttnn::operations::experimental::deepseek::prefill_dispatch_combined::PrefillDispatchCombinedDeviceOperation::
    tensor_return_value_t
    prefill_dispatch_combined(
        const ttnn::Tensor& input_tensor,
        const ttnn::Tensor& weights_tensor,
        const ttnn::Tensor& indices_tensor,
        const ttnn::Tensor& chip_to_n_routed_expert_offset_tensor,
        uint32_t num_chips,
        uint32_t experts_per_chip,
        uint32_t n_routed_experts,
        uint32_t num_experts_per_tok,
        uint32_t metadata_len,
        uint32_t max_dispatched_tokens_per_expert,
        std::optional<uint32_t> axis,
        uint32_t num_links,
        tt::tt_fabric::Topology topology,
        const ttnn::MemoryConfig& memory_config,
        const CoreRangeSet& worker_core_range_set) {
    using OperationType =
        ttnn::operations::experimental::deepseek::prefill_dispatch_combined::PrefillDispatchCombinedDeviceOperation;
    return ttnn::device_operation::launch<OperationType>(
        OperationType::operation_attributes_t{
            .num_chips = num_chips,
            .experts_per_chip = experts_per_chip,
            .n_routed_experts = n_routed_experts,
            .num_experts_per_tok = num_experts_per_tok,
            .metadata_len = metadata_len,
            .max_dispatched_tokens_per_expert = max_dispatched_tokens_per_expert,
            .axis = axis,
            .num_links = num_links,
            .topology = topology,
            .output_mem_config = memory_config,
            .worker_core_range_set = worker_core_range_set},
        OperationType::tensor_args_t{
            .input_tensor = input_tensor,
            .weights_tensor = weights_tensor,
            .indices_tensor = indices_tensor,
            .chip_to_n_routed_expert_offset_tensor = chip_to_n_routed_expert_offset_tensor});
}
}  // namespace ttnn::prim
