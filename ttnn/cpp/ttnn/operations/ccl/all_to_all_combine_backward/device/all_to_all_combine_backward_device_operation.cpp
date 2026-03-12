// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <utility>

#include "ttnn/tensor/tensor_ops.hpp"
#include "ttnn/tensor/types.hpp"
#include "all_to_all_combine_backward_device_operation.hpp"
#include "ttnn/device_operation.hpp"
#include "cpp/ttnn/operations/data_movement/common/common.hpp"
#include <tt-metalium/work_split.hpp>

namespace ttnn::operations::ccl {

void AllToAllCombineBackwardDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& grad_output = tensor_args.grad_output;
    const auto& metadata_tensor = tensor_args.metadata_tensor;
    const auto& mapping_tensor = tensor_args.mapping_tensor;

    TT_FATAL(
        grad_output.layout() == tt::tt_metal::Layout::ROW_MAJOR, "Grad output tensor must be in row major layout");
    TT_FATAL(
        metadata_tensor.layout() == tt::tt_metal::Layout::ROW_MAJOR, "Metadata tensor must be in row major layout");
    TT_FATAL(mapping_tensor.layout() == tt::tt_metal::Layout::ROW_MAJOR, "Mapping tensor must be in row major layout");

    TT_FATAL(grad_output.dtype() == tt::tt_metal::DataType::BFLOAT16, "Grad output tensor must be bfloat16");
    TT_FATAL(metadata_tensor.dtype() == tt::tt_metal::DataType::UINT16, "Metadata tensor must be uint16");
    TT_FATAL(mapping_tensor.dtype() == tt::tt_metal::DataType::UINT16, "Mapping tensor must be uint16");

    TT_FATAL(!operation_attributes.output_mem_config.is_sharded(), "Output memory config must not be sharded");

    if (tensor_args.optional_output_tensor.has_value()) {
        const auto output_spec = compute_output_specs(operation_attributes, tensor_args);
        const auto& output_tensor = tensor_args.optional_output_tensor.value();

        TT_FATAL(
            output_tensor.layout() == tt::tt_metal::Layout::ROW_MAJOR,
            "Optional output tensor must be in row major layout");

        TT_FATAL(
            output_spec == output_tensor.tensor_spec(),
            "Optional output tensor spec {} does not match computed output spec {}",
            output_tensor.tensor_spec(),
            output_spec);
    }

    const auto& grad_shape = grad_output.tensor_spec().logical_shape();
    const auto& metadata_shape = metadata_tensor.tensor_spec().logical_shape();
    const auto& mapping_shape = mapping_tensor.tensor_spec().logical_shape();

    const auto mesh_view = grad_output.device()->get_view();
    const auto mesh_rows = mesh_view.num_rows();
    const auto mesh_cols = mesh_view.num_cols();
    const auto batch = metadata_shape[1];
    const auto seq = metadata_shape[2];
    const auto experts = mapping_shape[2];

    TT_FATAL(
        (grad_shape.rank() == 4) && (metadata_shape.rank() == 4) && (mapping_shape.rank() == 4),
        "Grad output, metadata, and mapping tensors must all be rank 4");

    const auto num_devices = mesh_view.num_devices();

    TT_FATAL(
        experts % num_devices == 0,
        "Number of experts {} should be evenly divisible by devices: {}",
        experts,
        num_devices);

    TT_FATAL(
        experts % mesh_rows * mesh_cols == 0,
        "Experts {} must be evenly divisible by devices",
        experts,
        mesh_rows * mesh_cols);

    TT_FATAL(operation_attributes.axis.has_value(), "Axis must be specified");
    const auto& axis = operation_attributes.axis.value();
    const auto& axis_group = (axis == 0) ? mesh_rows : mesh_cols;

    TT_FATAL(
        operation_attributes.output_shard_dim == 1 || operation_attributes.output_shard_dim == 2,
        "Output shard dimension must be 1 or 2, got {}",
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

void AllToAllCombineBackwardDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& /*tensor_args*/) {}

AllToAllCombineBackwardDeviceOperation::spec_return_value_t
AllToAllCombineBackwardDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    using namespace tt::tt_metal;

    const auto& grad_shape = tensor_args.grad_output.tensor_spec().logical_shape();
    const auto& metadata_shape = tensor_args.metadata_tensor.tensor_spec().logical_shape();
    const auto& mapping_shape = tensor_args.mapping_tensor.tensor_spec().logical_shape();

    auto* mesh_device = tensor_args.grad_output.device();
    const auto& mesh_view = mesh_device->get_view();
    const uint32_t num_devices = mesh_view.num_devices();

    const uint32_t hidden_size = grad_shape[-1];
    const uint32_t batch_size = metadata_shape[1];   // global batch
    const uint32_t seq_size = metadata_shape[2];     // global seq
    const uint32_t experts = mapping_shape[-2];      // total experts
    const uint32_t experts_per_device = experts / num_devices;

    // The backward output (grad of forward input) has same shape as forward input:
    // [K_or_1, B_global, S, H]
    const uint32_t k_or_1 = operation_attributes.locally_reduced ? 1 : experts_per_device;

    auto output_shape = ttnn::Shape({k_or_1, batch_size, seq_size, hidden_size});

    auto mem_config = operation_attributes.output_mem_config;
    return TensorSpec(
        Shape(output_shape),
        TensorLayout(
            tensor_args.grad_output.dtype(), PageConfig(tensor_args.grad_output.layout()), mem_config));
}

AllToAllCombineBackwardDeviceOperation::tensor_return_value_t
AllToAllCombineBackwardDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto output_spec = compute_output_specs(operation_attributes, tensor_args);
    return tensor_args.optional_output_tensor.value_or(
        create_device_tensor(output_spec, tensor_args.grad_output.device()));
}

}  // namespace ttnn::operations::ccl

namespace ttnn::prim {
ttnn::Tensor all_to_all_combine_backward(
    const ttnn::Tensor& grad_output,
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
    using OperationType = ttnn::operations::ccl::AllToAllCombineBackwardDeviceOperation;
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
            .grad_output = grad_output,
            .mapping_tensor = expert_mapping_tensor,
            .metadata_tensor = expert_metadata_tensor,
            .optional_output_tensor = optional_output_tensor});
}
}  // namespace ttnn::prim
