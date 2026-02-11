// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <utility>

#include "ttnn/tensor/types.hpp"
#include "all_to_all_dispatch_metadata_device_operation.hpp"
#include "ttnn/device_operation.hpp"
#include "cpp/ttnn/operations/data_movement/common/common.hpp"
#include <tt-metalium/work_split.hpp>

namespace ttnn::operations::experimental::ccl {

AllToAllDispatchMetadataDeviceOperation::program_factory_t

    void
    AllToAllDispatchMetadataDeviceOperation::validate_on_program_cache_miss(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    auto input_tensor = tensor_args.input_tensor;
    auto indices_tensor = tensor_args.expert_indices_tensor;
    auto scores_tensor = tensor_args.expert_scores_tensor;

    TT_FATAL(input_tensor.layout() == tt::tt_metal::Layout::ROW_MAJOR, "Input tensor must be in row major layout");
    TT_FATAL(indices_tensor.layout() == tt::tt_metal::Layout::ROW_MAJOR, "Indices tensor must be in row major layout");
    TT_FATAL(scores_tensor.layout() == tt::tt_metal::Layout::ROW_MAJOR, "Scores tensor must be in row major layout");

    TT_FATAL(input_tensor.dtype() == tt::tt_metal::DataType::BFLOAT16, "Input tensor must be bfloat16");
    TT_FATAL(indices_tensor.dtype() == tt::tt_metal::DataType::UINT16, "Indices tensor must be uint16");
    TT_FATAL(scores_tensor.dtype() == tt::tt_metal::DataType::BFLOAT16, "Scores tensor must be bfloat16");

    // Validate scores tensor has same shape as indices tensor
    auto indices_shape = indices_tensor.tensor_spec().logical_shape();
    auto scores_shape = scores_tensor.tensor_spec().logical_shape();
    TT_FATAL(
        indices_shape == scores_shape,
        "Scores tensor shape {} must match indices tensor shape {}",
        scores_shape,
        indices_shape);

    auto output_specs = compute_output_specs(operation_attributes, tensor_args);

    // Validate persistent mode: if cross_device_semaphore is provided, all 3 output tensors must be provided
    // This is required to skip init_semaphore safely - all fabric write targets must be persistent
    if (operation_attributes.cross_device_semaphore.has_value()) {
        TT_FATAL(
            tensor_args.optional_output_tensors.has_value(),
            "When cross_device_semaphore is provided, all 3 output tensors (tokens, indices, scores) must be provided "
            "as persistent buffers to enable semaphore-free mode");
    }

    if (tensor_args.optional_output_tensors.has_value()) {
        auto output_tensors = tensor_args.optional_output_tensors.value();
        const auto& sparse_token_tensor = output_tensors[0];
        const auto& indices_tensor_out = output_tensors[1];
        const auto& scores_tensor_out = output_tensors[2];
        TT_FATAL(
            sparse_token_tensor.layout() == tt::tt_metal::Layout::ROW_MAJOR,
            "Output tensor must be in row major layout");
        TT_FATAL(
            indices_tensor_out.layout() == tt::tt_metal::Layout::ROW_MAJOR,
            "Output indices tensor must be in row major layout");
        TT_FATAL(
            scores_tensor_out.layout() == tt::tt_metal::Layout::ROW_MAJOR,
            "Output scores tensor must be in row major layout");

        // Note: When persistent output tensors are provided, the drain_sync_tilizer_core is extracted
        // from their shard spec in the program factory. Validation of single-core sharding happens there.

        TT_FATAL(
            output_specs[0] == sparse_token_tensor.tensor_spec(),
            "Optional sparse output token tensor spec {} does not match computed output spec {}",
            sparse_token_tensor.tensor_spec(),
            output_specs[0]);
        TT_FATAL(
            output_specs[1] == indices_tensor_out.tensor_spec(),
            "Optional indices tensor spec {} does not match computed output spec {}",
            indices_tensor_out.tensor_spec(),
            output_specs[1]);
        TT_FATAL(
            output_specs[2] == scores_tensor_out.tensor_spec(),
            "Optional scores tensor spec {} does not match computed output spec {}",
            scores_tensor_out.tensor_spec(),
            output_specs[2]);
    }
    TT_FATAL(operation_attributes.num_links > 0, "Number of links must be greater than 0");

    auto input_shape = input_tensor.tensor_spec().logical_shape();
    TT_FATAL(
        input_shape.rank() == 4 && (input_shape.rank() == indices_shape.rank()),
        "Input and indices tensor must have the same number of dimensions");
    for (uint32_t i = 0; i < indices_shape.rank() - 1; i++) {
        TT_FATAL(
            input_shape[i] == indices_shape[i],
            "Input and indices tensor must have the same shape for all dimensions except the last. Mismatch at "
            "dimension {} with shape {} and {}",
            i,
            input_shape[i],
            indices_shape[i]);
    }

    // Validate expert mapping tensor shape: new format is [devices, experts]
    auto mapping_shape = tensor_args.expert_mapping_tensor.tensor_spec().logical_shape();
    TT_FATAL(
        mapping_shape.rank() == 2,
        "Expert mapping tensor must have rank 2 with shape [devices, experts], got rank {}",
        mapping_shape.rank());

    auto* mesh_device = input_tensor.device();
    const auto& mesh_view = mesh_device->get_view();
    uint32_t expected_devices = mesh_view.num_devices();
    TT_FATAL(
        mapping_shape[0] == expected_devices,
        "Expert mapping tensor first dimension must equal number of devices ({}), got {}",
        expected_devices,
        mapping_shape[0]);
}

void AllToAllDispatchMetadataDeviceOperation::validate_on_program_cache_hit(
    [[maybe_unused]] const operation_attributes_t& operation_attributes,
    [[maybe_unused]] const tensor_args_t& tensor_args) {}

AllToAllDispatchMetadataDeviceOperation::spec_return_value_t
AllToAllDispatchMetadataDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    auto input_tensor = tensor_args.input_tensor;
    auto input_shape = input_tensor.tensor_spec().logical_shape();
    auto indices_shape = tensor_args.expert_indices_tensor.tensor_spec().logical_shape();
    auto mapping_shape = tensor_args.expert_mapping_tensor.tensor_spec().logical_shape();

    auto* mesh_device = input_tensor.device();
    const auto& mesh_view = mesh_device->get_view();

    // experts are expert parallel across devices
    // tokens are data parallel across devices
    // when axis is specified, we assume that tokens are only data parallel across the specified axis, and duplicated
    // along the other axis the indices match the token tensor the mapping tensor maps the experts to where they are on
    // the device mesh the mapping tensor is generally the same for all devices, except for the case where we have a
    // shared expert in that case, we can hide the fact that the expert is also on the other devices by setting the
    // mapping tensor to 0 for all other devices if axis is specified, we only route the tokens along the specified
    // axis, and skip any experts that are not on the specified axis

    uint32_t dispatch_devices = mesh_view.num_devices();
    uint32_t hidden_size = input_shape[-1];
    if (operation_attributes.axis.has_value()) {
        uint32_t axis = operation_attributes.axis.value();
        log_debug(tt::LogOp, "axis: {}", axis);
        dispatch_devices = axis == 0 ? mesh_view.num_rows() : mesh_view.num_cols();
    }
    uint32_t total_tokens = input_shape[0] * input_shape[1] * input_shape[2] * dispatch_devices;

    uint32_t selected_experts_k = indices_shape[-1];

    auto output_shape = ttnn::Shape({1, total_tokens, hidden_size});
    auto metadata_shape = ttnn::Shape({1, total_tokens, selected_experts_k});

    // Output tokens tensor - use input tensor's memory config (DRAM interleaved)
    auto dram_memory_config =
        tt::tt_metal::MemoryConfig{tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::DRAM};
    auto output_tokens_spec = TensorSpec(
        Shape(output_shape),
        tt::tt_metal::TensorLayout(
            input_tensor.dtype(), tt::tt_metal::PageConfig(input_tensor.layout()), dram_memory_config));

    // Create sharded memory config for indices/scores on drain_sync_tilizer_core
    // Use provided drain_sync_tilizer_core, or default to (0, 0) if not provided
    // (When persistent output tensors are provided, we extract the core from their shard spec instead)
    CoreCoord drain_core = operation_attributes.drain_sync_tilizer_core.value_or(CoreCoord(0, 0));
    CoreRangeSet drain_core_range_set({CoreRange(drain_core, drain_core)});

    auto indices_shard_spec = tt::tt_metal::ShardSpec(
        drain_core_range_set, {total_tokens, selected_experts_k}, tt::tt_metal::ShardOrientation::ROW_MAJOR);

    auto indices_sharded_mem_config = tt::tt_metal::MemoryConfig{
        tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED, tt::tt_metal::BufferType::L1, indices_shard_spec};

    auto scores_shard_spec = tt::tt_metal::ShardSpec(
        drain_core_range_set, {total_tokens, selected_experts_k}, tt::tt_metal::ShardOrientation::ROW_MAJOR);

    auto scores_sharded_mem_config = tt::tt_metal::MemoryConfig{
        tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED, tt::tt_metal::BufferType::L1, scores_shard_spec};

    // Indices tensor spec - sharded to drain core
    auto indices_spec = TensorSpec(
        Shape(metadata_shape),
        tt::tt_metal::TensorLayout(
            tensor_args.expert_indices_tensor.dtype(),
            tt::tt_metal::PageConfig(tensor_args.expert_indices_tensor.layout()),
            indices_sharded_mem_config));

    // Scores tensor spec - sharded to drain core (same shape as indices, different dtype)
    auto scores_spec = TensorSpec(
        Shape(metadata_shape),
        tt::tt_metal::TensorLayout(
            tensor_args.expert_scores_tensor.dtype(),
            tt::tt_metal::PageConfig(tensor_args.expert_scores_tensor.layout()),
            scores_sharded_mem_config));

    log_debug(tt::LogOp, "indices_spec shape: {}", indices_spec.logical_shape());
    log_debug(tt::LogOp, "scores_spec shape: {}", scores_spec.logical_shape());
    log_debug(tt::LogOp, "drain_sync_tilizer_core: ({}, {})", drain_core.x, drain_core.y);

    if (tensor_args.optional_output_tensors.has_value()) {
        auto output_tensors = tensor_args.optional_output_tensors.value();
        auto preallocated_output_spec = output_tensors[0].tensor_spec();
        auto preallocated_indices_spec = output_tensors[1].tensor_spec();
        auto preallocated_scores_spec = output_tensors[2].tensor_spec();
        return {preallocated_output_spec, preallocated_indices_spec, preallocated_scores_spec};
    }
    return {output_tokens_spec, indices_spec, scores_spec};
}

AllToAllDispatchMetadataDeviceOperation::tensor_return_value_t
AllToAllDispatchMetadataDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    if (tensor_args.optional_output_tensors.has_value()) {
        return tensor_args.optional_output_tensors.value();
    }
    auto output_spec = compute_output_specs(operation_attributes, tensor_args);

    auto output_tensor = create_device_tensor(output_spec[0], tensor_args.input_tensor.device());
    auto indices_tensor = create_device_tensor(output_spec[1], tensor_args.input_tensor.device());
    auto scores_tensor = create_device_tensor(output_spec[2], tensor_args.input_tensor.device());
    return {output_tensor, indices_tensor, scores_tensor};
}

}  // namespace ttnn::operations::experimental::ccl

namespace ttnn::prim {
ttnn::operations::experimental::ccl::AllToAllDispatchMetadataDeviceOperation::tensor_return_value_t
all_to_all_dispatch_metadata(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& expert_indices_tensor,
    const ttnn::Tensor& expert_scores_tensor,
    const ttnn::Tensor& expert_mapping_tensor,
    std::optional<uint32_t> axis,
    const std::optional<std::array<ttnn::Tensor, 3>>& optional_output_tensors,
    uint32_t num_links,
    tt::tt_fabric::Topology topology,
    const CoreRangeSet& worker_core_range_set,
    const std::optional<CoreCoord>& drain_sync_tilizer_core,
    ttnn::operations::experimental::ccl::WorkerMode worker_mode,
    const CoreRangeSet& mux_core_range_set,
    ttnn::operations::experimental::ccl::DispatchAlgorithm dispatch_algorithm,
    const std::optional<ttnn::GlobalSemaphore>& cross_device_semaphore) {
    using OperationType = ttnn::operations::experimental::ccl::AllToAllDispatchMetadataDeviceOperation;
    return ttnn::device_operation::launch<OperationType>(
        OperationType::operation_attributes_t{
            .worker_core_range_set = worker_core_range_set,
            .axis = axis,
            .num_links = num_links,
            .topology = topology,
            .drain_sync_tilizer_core = drain_sync_tilizer_core,
            .worker_mode = worker_mode,
            .mux_core_range_set = mux_core_range_set,
            .dispatch_algorithm = dispatch_algorithm,
            .cross_device_semaphore = cross_device_semaphore},
        OperationType::tensor_args_t{
            .input_tensor = input_tensor,
            .expert_indices_tensor = expert_indices_tensor,
            .expert_scores_tensor = expert_scores_tensor,
            .expert_mapping_tensor = expert_mapping_tensor,
            .optional_output_tensors = optional_output_tensors});
}
}  // namespace ttnn::prim
