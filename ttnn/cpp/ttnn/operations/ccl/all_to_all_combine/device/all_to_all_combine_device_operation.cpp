// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <utility>

#include "ttnn/tensor/tensor_ops.hpp"
#include "ttnn/tensor/types.hpp"
#include "all_to_all_combine_device_operation.hpp"
#include "ttnn/device_operation.hpp"
#include "cpp/ttnn/operations/data_movement/common/common.hpp"
#include <tt-metalium/work_split.hpp>

namespace ttnn::operations::ccl {

AllToAllCombineDeviceOperation::program_factory_t AllToAllCombineDeviceOperation::select_program_factory(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& /*tensor_args*/) {
    return AllToAllCombineFromSparse{};
}

void AllToAllCombineDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input_tensor;
    const auto& metadata_tensor = tensor_args.metadata_tensor;
    const auto& mapping_tensor = tensor_args.mapping_tensor;

    TT_FATAL(input_tensor.layout() == tt::tt_metal::Layout::ROW_MAJOR, "Input tensor must be in row major layout");
    TT_FATAL(
        metadata_tensor.layout() == tt::tt_metal::Layout::ROW_MAJOR, "Metadata tensor must be in row major layout");

    TT_FATAL(input_tensor.dtype() == tt::tt_metal::DataType::BFLOAT16, "Input tensor must be bfloat16");
    TT_FATAL(metadata_tensor.dtype() == tt::tt_metal::DataType::UINT16, "Metadata tensor must be uint16");

    TT_FATAL(mapping_tensor.dtype() == tt::tt_metal::DataType::UINT16, "Indices tensor must be uint32");
    TT_FATAL(mapping_tensor.layout() == tt::tt_metal::Layout::ROW_MAJOR, "Metadata tensor must be in row major layout");

    TT_FATAL(!operation_attributes.output_mem_config.is_sharded(), "Output memory config must not be sharded");

    if (tensor_args.optional_output_tensor.has_value()) {
        const auto output_spec = compute_output_specs(operation_attributes, tensor_args);
        const auto& output_tensor = tensor_args.optional_output_tensor.value();

        TT_FATAL(
            output_tensor.layout() == tt::tt_metal::Layout::ROW_MAJOR, "Output tensor must be in row major layout");

        TT_FATAL(
            output_spec == output_tensor.tensor_spec(),
            "Optional sparse output token tensor spec {} does not match computed output spec {}",
            output_tensor.tensor_spec(),
            output_spec);
    }

    const auto& input_shape = input_tensor.tensor_spec().logical_shape();
    const auto& metadata_shape = metadata_tensor.tensor_spec().logical_shape();
    const auto& mapping_shape = mapping_tensor.tensor_spec().logical_shape();

    const auto mesh_view = input_tensor.device()->get_view();
    const auto mesh_rows = mesh_view.num_rows();
    const auto mesh_cols = mesh_view.num_cols();
    const auto batch = metadata_shape[1];
    const auto seq = metadata_shape[2];
    const auto experts = mapping_shape[2];

    TT_FATAL(
        experts % mesh_rows * mesh_cols == 0,
        "Experts {} must be evenly divisible by devices",
        experts,
        mesh_rows * mesh_cols);

    TT_FATAL(
        (input_shape.rank() == 4) && (metadata_shape.rank() == 4) && (mapping_shape.rank() == 4),
        "Input, metadata, and mapping tensors must all be rank 4");

    const auto num_devices = mesh_view.num_devices();

    TT_FATAL(
        experts % num_devices == 0,
        "Number of experts {} should be evenly divisible by devices: {}",
        experts,
        num_devices);

    if (operation_attributes.locally_reduced) {
        TT_FATAL(
            input_shape[0] == 1,
            "Expecting input dim 0 equal to num devices: {}, got: {}",
            num_devices,
            input_shape[0]);
    } else {
        TT_FATAL(
            input_shape[0] == experts / num_devices,
            "Expected input shape dim 0: {} to be equal to expert mapping dim 2: {}",
            input_shape[0],
            mapping_shape[2]);
    }

    TT_FATAL(operation_attributes.axis.has_value(), "Axis must be specified at the moment");
    const auto& axis = operation_attributes.axis.value();
    const auto& axis_group = (axis == 0) ? mesh_rows : mesh_cols;
    TT_FATAL(
        operation_attributes.output_shard_dim == 1 || operation_attributes.output_shard_dim == 2,
        "Output shard dimension must be 1 or 2, got {}. Output shard dimension is used to determine the dimension to "
        "shard the output tokens along.",
        operation_attributes.output_shard_dim);
    uint32_t output_shard_dim = operation_attributes.output_shard_dim;
    if (output_shard_dim == 1) {
        TT_FATAL(batch % axis_group == 0, "Batch {} must be divisible by axis group", batch, axis_group);
    } else if (output_shard_dim == 2) {
        TT_FATAL(seq % axis_group == 0, "Sequence length {} must be divisible by axis group", seq, axis_group);
    }

    TT_FATAL(
        operation_attributes.num_links > 0,
        "Number of links must be greater than 0, got {}",
        operation_attributes.num_links);
}

void AllToAllCombineDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& /*tensor_args*/) {}

AllToAllCombineDeviceOperation::spec_return_value_t AllToAllCombineDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    using namespace tt::tt_metal;
    uint32_t output_shard_dim = operation_attributes.output_shard_dim;

    const auto& input_tensor = tensor_args.input_tensor;
    const auto& input_shape = input_tensor.tensor_spec().logical_shape();
    const auto& metadata_shape = tensor_args.metadata_tensor.tensor_spec().logical_shape();

    auto* mesh_device = input_tensor.device();
    const auto& mesh_view = mesh_device->get_view();

    const auto num_devices = mesh_view.num_devices();

    const uint32_t hidden_size = input_shape[-1];
    const uint32_t batch_size = metadata_shape[1];
    const uint32_t seq_size = metadata_shape[2];

    const uint32_t selected_experts_k = metadata_shape[-1];

    const auto& axis = operation_attributes.axis;
    const uint32_t replicate_dim = axis.has_value() ? mesh_device->shape()[!axis.value()] : 1;

    const uint32_t total_batch_per_device_size =
        (output_shard_dim == 1) ? (batch_size * replicate_dim) / num_devices : batch_size;
    const uint32_t total_seq_per_device_size =
        (output_shard_dim == 2) ? (seq_size * replicate_dim) / num_devices : seq_size;

    auto output_shape =
        ttnn::Shape({selected_experts_k, total_batch_per_device_size, total_seq_per_device_size, hidden_size});

    auto mem_config = operation_attributes.output_mem_config;
    return TensorSpec(
        Shape(output_shape),
        TensorLayout(tensor_args.input_tensor.dtype(), PageConfig(tensor_args.input_tensor.layout()), mem_config));
}

AllToAllCombineDeviceOperation::tensor_return_value_t AllToAllCombineDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto output_spec = compute_output_specs(operation_attributes, tensor_args);
    return tensor_args.optional_output_tensor.value_or(
        create_device_tensor(output_spec, tensor_args.input_tensor.device()));
}

}  // namespace ttnn::operations::ccl

namespace ttnn::prim {
ttnn::Tensor all_to_all_combine(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& expert_mapping_tensor,
    const ttnn::Tensor& expert_metadata_tensor,
    const uint32_t num_links,
    const tt::tt_fabric::Topology topology,
    const ttnn::MemoryConfig& memory_config,
    const std::optional<uint32_t>& axis,
    const std::optional<ttnn::Tensor>& optional_output_tensor,
    const bool locally_reduced,
    const CoreRangeSet& worker_core_range_set,
    uint32_t output_shard_dim) {
    using OperationType = ttnn::operations::ccl::AllToAllCombineDeviceOperation;
    return ttnn::device_operation::launch<OperationType>(
        OperationType::operation_attributes_t{
            .output_mem_config = memory_config,
            .axis = axis,
            .num_links = num_links,
            .topology = topology,
            .locally_reduced = locally_reduced,
            .worker_core_range_set = worker_core_range_set,
            .output_shard_dim = output_shard_dim,
        },
        OperationType::tensor_args_t{
            .input_tensor = input_tensor,
            .mapping_tensor = expert_mapping_tensor,
            .metadata_tensor = expert_metadata_tensor,
            .optional_output_tensor = optional_output_tensor});
}
}  // namespace ttnn::prim
