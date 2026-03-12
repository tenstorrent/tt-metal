// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <utility>

#include "ttnn/tensor/tensor_ops.hpp"
#include "ttnn/tensor/types.hpp"
#include "all_to_all_dispatch_backward_device_operation.hpp"
#include "ttnn/device_operation.hpp"
#include "cpp/ttnn/operations/data_movement/common/common.hpp"
#include <tt-metalium/work_split.hpp>

namespace ttnn::operations::ccl {

void AllToAllDispatchBackwardDeviceOperation::validate_on_program_cache_miss(
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

    TT_FATAL(
        (grad_shape.rank() == 4) && (metadata_shape.rank() == 4) && (mapping_shape.rank() == 4),
        "Grad output, metadata, and mapping tensors must all be rank 4");

    TT_FATAL(operation_attributes.axis.has_value(), "Axis must be specified");

    TT_FATAL(
        operation_attributes.output_shard_dim == 1 || operation_attributes.output_shard_dim == 2,
        "Output shard dimension must be 1 or 2, got {}",
        operation_attributes.output_shard_dim);

    TT_FATAL(
        operation_attributes.num_links > 0,
        "Number of links must be greater than 0, got {}",
        operation_attributes.num_links);
}

void AllToAllDispatchBackwardDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& /*tensor_args*/) {}

AllToAllDispatchBackwardDeviceOperation::spec_return_value_t
AllToAllDispatchBackwardDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    using namespace tt::tt_metal;

    const auto& grad_shape = tensor_args.grad_output.tensor_spec().logical_shape();

    auto* mesh_device = tensor_args.grad_output.device();
    const auto& mesh_view = mesh_device->get_view();
    const uint32_t num_devices = mesh_view.num_devices();

    const uint32_t hidden_size = grad_shape[-1];

    // Dispatch forward input was [1, B_per_device, S, H].
    // Dispatch forward output (= backward input) is [1, B_per_device * dispatch_devices, S, H].
    // dispatch_devices = number of devices along the dispatch axis.
    const uint32_t dispatch_devices =
        operation_attributes.axis.has_value()
            ? (uint32_t)mesh_device->shape()[operation_attributes.axis.value()]
            : num_devices;

    uint32_t batch_per_device, seq;
    if (operation_attributes.output_shard_dim == 1) {
        batch_per_device = grad_shape[1] / dispatch_devices;
        seq = grad_shape[2];
    } else {
        batch_per_device = grad_shape[1];
        seq = grad_shape[2] / dispatch_devices;
    }

    // Output: [dispatch_devices, B_per_device, S, H] — expanded form.
    // Each dispatch device writes to its own slot (non-overlapping).
    // The caller sums over dim 0 to get the final [1, B_per_device, S, H].
    auto output_shape = ttnn::Shape({dispatch_devices, batch_per_device, seq, hidden_size});

    auto mem_config = operation_attributes.output_mem_config;
    return TensorSpec(
        Shape(output_shape),
        TensorLayout(
            tensor_args.grad_output.dtype(), PageConfig(tensor_args.grad_output.layout()), mem_config));
}

AllToAllDispatchBackwardDeviceOperation::tensor_return_value_t
AllToAllDispatchBackwardDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto output_spec = compute_output_specs(operation_attributes, tensor_args);
    return tensor_args.optional_output_tensor.value_or(
        create_device_tensor(output_spec, tensor_args.grad_output.device()));
}

}  // namespace ttnn::operations::ccl

namespace ttnn::prim {
ttnn::Tensor all_to_all_dispatch_backward(
    const ttnn::Tensor& grad_output,
    const ttnn::Tensor& expert_mapping_tensor,
    const ttnn::Tensor& expert_metadata_tensor,
    uint32_t num_links,
    tt::tt_fabric::Topology topology,
    const ttnn::MemoryConfig& memory_config,
    const std::optional<uint32_t>& axis,
    const std::optional<ttnn::Tensor>& optional_output_tensor,
    const CoreRangeSet& worker_core_range_set,
    uint32_t output_shard_dim) {
    using OperationType = ttnn::operations::ccl::AllToAllDispatchBackwardDeviceOperation;
    return ttnn::device_operation::launch<OperationType>(
        OperationType::operation_attributes_t{
            .output_mem_config = memory_config,
            .axis = axis,
            .num_links = num_links,
            .topology = topology,
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
