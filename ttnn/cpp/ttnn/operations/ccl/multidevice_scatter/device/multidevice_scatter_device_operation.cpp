// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <utility>

#include "ttnn/tensor/types.hpp"
#include "multidevice_scatter_device_operation.hpp"
#include "cpp/ttnn/operations/data_movement/common/common.hpp"
#include <tt-metalium/work_split.hpp>

namespace ttnn::operations::ccl {

MultiDeviceScatterDeviceOperation::program_factory_t MultiDeviceScatterDeviceOperation::select_program_factory(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return MultiDeviceScatter{};
}

void MultiDeviceScatterDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    auto input_tensor = tensor_args.input_tensor;
    uint32_t rank = input_tensor.logical_shape().rank();
    auto output_spec = compute_output_specs(operation_attributes, tensor_args);
    auto output_shape = output_spec.logical_shape();
    auto input_shape = input_tensor.logical_shape();

    TT_FATAL(operation_attributes.dim < rank, "dim must be less than the rank of the input tensor");

    auto mesh_device = input_tensor.mesh_device();
    const auto& mesh_view = mesh_device->get_view();
    const uint32_t cluster_axis_size =
        (operation_attributes.cluster_axis == 0) ? mesh_view.num_rows() : mesh_view.num_cols();

    TT_FATAL(
        cluster_axis_size > 1,
        "Scatter has only been tested with mesh axis size > 1, but has {} devices",
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

void MultiDeviceScatterDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {}

MultiDeviceScatterDeviceOperation::spec_return_value_t MultiDeviceScatterDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    using namespace tt::tt_metal;

    auto input_tensor = tensor_args.input_tensor;
    auto output_shape = input_tensor.logical_shape();
    const auto& mesh_view = input_tensor.mesh_device()->get_view();

    const uint32_t num_devices = (operation_attributes.cluster_axis == 0) ? mesh_view.num_rows() : mesh_view.num_cols();

    output_shape[operation_attributes.dim] = output_shape[operation_attributes.dim] / num_devices;
    return {TensorSpec(
        Shape(output_shape),
        TensorLayout(input_tensor.dtype(), PageConfig(input_tensor.layout()), operation_attributes.output_mem_config))};
}

MultiDeviceScatterDeviceOperation::tensor_return_value_t MultiDeviceScatterDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    auto output_spec = compute_output_specs(operation_attributes, tensor_args);

    auto tensor = create_device_tensor(output_spec, tensor_args.input_tensor.device());
    return tensor;
}

std::tuple<MultiDeviceScatterDeviceOperation::operation_attributes_t, MultiDeviceScatterDeviceOperation::tensor_args_t>
MultiDeviceScatterDeviceOperation::invoke(
    const ttnn::Tensor& input_tensor,
    const int32_t dim,
    const uint32_t cluster_axis,
    const ttnn::MemoryConfig& memory_config) {
    return {
        operation_attributes_t{
            .dim = (dim < 0 ? uint32_t(input_tensor.logical_shape().rank() + dim) : (uint32_t)dim),
            .cluster_axis = cluster_axis,
            .output_mem_config = memory_config,
        },
        tensor_args_t{.input_tensor = input_tensor}};
}

}  // namespace ttnn::operations::ccl
