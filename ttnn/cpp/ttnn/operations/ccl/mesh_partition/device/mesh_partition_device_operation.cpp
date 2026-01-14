// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <utility>

#include "ttnn/tensor/types.hpp"
#include "mesh_partition_device_operation.hpp"
#include "ttnn/device_operation.hpp"
#include "cpp/ttnn/operations/data_movement/common/common.hpp"
#include <tt-metalium/work_split.hpp>
#include "ttnn/tensor/tensor_ops.hpp"

namespace ttnn::operations::ccl {

namespace detail {
uint32_t get_cluster_axis_size(const ttnn::Tensor& input_tensor, const std::optional<uint32_t>& cluster_axis) {
    auto* mesh_device = input_tensor.device();
    const auto& mesh_view = mesh_device->get_view();
    return cluster_axis.has_value() ? ((cluster_axis.value() == 0) ? mesh_view.num_rows() : mesh_view.num_cols())
                                    : mesh_view.num_devices();
}
}  // namespace detail

MeshPartitionDeviceOperation::program_factory_t MeshPartitionDeviceOperation::select_program_factory(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& /*tensor_args*/) {
    return MeshPartition{};
}

void MeshPartitionDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    auto input_tensor = tensor_args.input_tensor;
    uint32_t rank = input_tensor.logical_shape().rank();
    auto output_spec = compute_output_specs(operation_attributes, tensor_args);
    if (tensor_args.optional_output_tensor.has_value()) {
        TT_FATAL(
            tensor_args.optional_output_tensor.value().tensor_spec() == output_spec,
            "Output tensor spec must match computed output spec");
    }
    const auto& output_shape = output_spec.logical_shape();
    const auto& input_shape = input_tensor.logical_shape();

    TT_FATAL(
        !(operation_attributes.cluster_axis.has_value() && operation_attributes.cluster_axis.value() > 1),
        "Only support cluster axis of None, 0 or 1");

    TT_FATAL(operation_attributes.dim < rank, "dim must be less than the rank of the input tensor");

    const uint32_t cluster_axis_size = detail::get_cluster_axis_size(input_tensor, operation_attributes.cluster_axis);

    TT_FATAL(
        cluster_axis_size > 1,
        "Partition has only been tested with mesh axis size > 1, but has {} devices",
        cluster_axis_size);
    TT_FATAL(
        input_shape[operation_attributes.dim] % cluster_axis_size == 0,
        "input shape {} must be divisible by cluster axis size {}",
        input_tensor.logical_shape(),
        cluster_axis_size);

    if (input_tensor.layout() == ttnn::TILE_LAYOUT) {
        TT_FATAL(
            output_shape == output_spec.padded_shape(),
            "output shape {} must be equal to padded shape {} for tiled inputs",
            output_shape,
            output_spec.padded_shape());
    }
}

void MeshPartitionDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& /*tensor_args*/) {}

MeshPartitionDeviceOperation::spec_return_value_t MeshPartitionDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    auto input_tensor = tensor_args.input_tensor;
    auto output_shape = input_tensor.logical_shape();

    const uint32_t cluster_axis_size = detail::get_cluster_axis_size(input_tensor, operation_attributes.cluster_axis);

    output_shape[operation_attributes.dim] = output_shape[operation_attributes.dim] / cluster_axis_size;
    return {TensorSpec(
        Shape(output_shape),
        tt::tt_metal::TensorLayout(
            input_tensor.dtype(),
            tt::tt_metal::PageConfig(input_tensor.layout()),
            operation_attributes.output_mem_config))};
}

MeshPartitionDeviceOperation::tensor_return_value_t MeshPartitionDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    if (tensor_args.optional_output_tensor.has_value()) {
        return tensor_args.optional_output_tensor.value();
    }

    auto output_spec = compute_output_specs(operation_attributes, tensor_args);

    auto tensor = create_device_tensor(output_spec, tensor_args.input_tensor.device());
    return tensor;
}

}  // namespace ttnn::operations::ccl

namespace ttnn::prim {
ttnn::Tensor mesh_partition(
    const ttnn::Tensor& input_tensor,
    int32_t dim,
    std::optional<uint32_t> cluster_axis,
    const ttnn::MemoryConfig& memory_config,
    const std::optional<ttnn::Tensor>& optional_output_tensor) {
    using OperationType = ttnn::operations::ccl::MeshPartitionDeviceOperation;
    return ttnn::device_operation::launch<OperationType>(
        OperationType::operation_attributes_t{
            .dim = (dim < 0 ? uint32_t(input_tensor.logical_shape().rank() + dim) : (uint32_t)dim),
            .cluster_axis = cluster_axis,
            .output_mem_config = memory_config,
        },
        OperationType::tensor_args_t{.input_tensor = input_tensor, .optional_output_tensor = optional_output_tensor});
}
}  // namespace ttnn::prim
