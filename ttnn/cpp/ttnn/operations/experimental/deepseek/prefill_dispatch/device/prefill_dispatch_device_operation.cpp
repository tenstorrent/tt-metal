// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <utility>

#include "ttnn/tensor/types.hpp"
#include "prefill_dispatch_device_operation.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

namespace ttnn::operations::experimental::deepseek::prefill_dispatch {

void PrefillDispatchDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    // Validate input tensor layouts are ROW_MAJOR
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
        tensor_args.expert_dispatch_table_tensor.layout() == tt::tt_metal::Layout::ROW_MAJOR,
        "Expert dispatch table tensor must be ROW_MAJOR layout");

    // Validate input tensor dtypes
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
        tensor_args.expert_dispatch_table_tensor.dtype() == DataType::INT32,
        "Expert dispatch table tensor must be INT32, got {}",
        tensor_args.expert_dispatch_table_tensor.dtype());

    // Validate output memory config is DRAM interleaved (not sharded)
    TT_FATAL(
        !operation_attributes.output_mem_config.is_sharded(),
        "Output memory config must be DRAM interleaved, not sharded");

    // Validate dimension compatibility
    auto input_shape = tensor_args.input_tensor.tensor_spec().logical_shape();
    auto weights_shape = tensor_args.weights_tensor.tensor_spec().logical_shape();
    auto indices_shape = tensor_args.indices_tensor.tensor_spec().logical_shape();

    // TT_FATAL(
    //     input_shape[-1] == weights_shape[-1],
    //     "Input hidden dim {} must match weights hidden dim {}",
    //     input_shape[-1],
    //     weights_shape[-1]);
}

void PrefillDispatchDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& /*tensor_args*/) {
    // Empty for now
}

PrefillDispatchDeviceOperation::spec_return_value_t PrefillDispatchDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    // Extract necessary dimensions from operation attributes
    uint32_t experts_per_chip = operation_attributes.experts_per_chip;
    uint32_t metadata_len = operation_attributes.metadata_len;
    uint32_t max_dispatched_tokens_per_expert = operation_attributes.max_dispatched_tokens_per_expert;

    // Get the input tensor's per-device shape (sharded dimension)
    auto input_shape = tensor_args.input_tensor.tensor_spec().logical_shape();

    // Extract per-device batch size from the input's first dimension (sharded on dim 0)
    uint32_t per_device_batch = input_shape[0];  // This is 1 for input sharded on dim 0
    uint32_t hidden_dim = input_shape[-1];

    // Memory config for all output tensors (inherits sharding from input)
    auto mem_config = operation_attributes.output_mem_config;

    // Layout for all output tensors
    auto layout = tt::tt_metal::Layout::ROW_MAJOR;

    // Define output shapes - these are PER-DEVICE shapes (not global shapes)
    // When sharded on dim 0, each device should get shape [1, ...]
    auto dispatch_buffer_shape =
        ttnn::Shape({per_device_batch, 1, experts_per_chip, max_dispatched_tokens_per_expert, hidden_dim});
    auto dispatch_metadata_shape =
        ttnn::Shape({per_device_batch, 1, experts_per_chip, max_dispatched_tokens_per_expert, metadata_len});

    // Create TensorSpec objects with correct dtypes
    auto dispatch_buffer_spec = TensorSpec(
        Shape(dispatch_buffer_shape),
        tt::tt_metal::TensorLayout(DataType::BFLOAT16, tt::tt_metal::PageConfig(layout), mem_config));

    auto dispatch_metadata_spec = TensorSpec(
        Shape(dispatch_metadata_shape),
        tt::tt_metal::TensorLayout(DataType::INT32, tt::tt_metal::PageConfig(layout), mem_config));

    return {dispatch_buffer_spec, dispatch_metadata_spec};
}

PrefillDispatchDeviceOperation::topology_return_value_t PrefillDispatchDeviceOperation::compute_output_topologies(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& tensor_args) {
    // Output tensors should have the same distribution topology as input tensor (sharded on dim 0)
    const auto& input_tensor = tensor_args.input_tensor;
    const auto& input_topology = input_tensor.tensor_topology();

    // Both output tensors use the same topology as the input
    // (sharded on dimension 0 across the mesh)
    auto output_topology = tt::tt_metal::TensorTopology(
        input_topology.distribution_shape(), input_topology.placements(), input_topology.mesh_coords());

    return {output_topology, output_topology};
}

PrefillDispatchDeviceOperation::tensor_return_value_t PrefillDispatchDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    auto output_spec = compute_output_specs(operation_attributes, tensor_args);

    auto output_tensor = create_device_tensor(output_spec[0], tensor_args.input_tensor.device());
    auto metadata_tensor = create_device_tensor(output_spec[1], tensor_args.input_tensor.device());
    return {output_tensor, metadata_tensor};
}

}  // namespace ttnn::operations::experimental::deepseek::prefill_dispatch

namespace ttnn::prim {
ttnn::operations::experimental::deepseek::prefill_dispatch::PrefillDispatchDeviceOperation::tensor_return_value_t
prefill_dispatch(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& weights_tensor,
    const ttnn::Tensor& indices_tensor,
    const ttnn::Tensor& chip_to_n_routed_expert_offset_tensor,
    const ttnn::Tensor& expert_dispatch_table_tensor,
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
    using OperationType = ttnn::operations::experimental::deepseek::prefill_dispatch::PrefillDispatchDeviceOperation;
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
            .chip_to_n_routed_expert_offset_tensor = chip_to_n_routed_expert_offset_tensor,
            .expert_dispatch_table_tensor = expert_dispatch_table_tensor});
}
}  // namespace ttnn::prim
