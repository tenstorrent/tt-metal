// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <utility>

#include "ttnn/tensor/types.hpp"
#include "all_to_all_combine_device_operation.hpp"
#include "cpp/ttnn/operations/data_movement/common/common.hpp"
#include <tt-metalium/work_split.hpp>

namespace ttnn::operations::ccl {

AllToAllCombineDeviceOperation::program_factory_t AllToAllCombineDeviceOperation::select_program_factory(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return AllToAllCombineFromSparse{};
}

//! TODO
void AllToAllCombineDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {}

// !TODO
void AllToAllCombineDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {}

AllToAllCombineDeviceOperation::spec_return_value_t AllToAllCombineDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    using namespace tt::tt_metal;

    const auto& input_tensor = tensor_args.input_tensor;
    const auto& input_shape = input_tensor.get_tensor_spec().logical_shape();
    const auto& metadata_shape = tensor_args.metadata_tensor.get_tensor_spec().logical_shape();

    auto mesh_device = input_tensor.mesh_device();
    const auto& mesh_view = mesh_device->get_view();

    const auto num_devices = mesh_view.num_devices();

    const uint32_t hidden_size = input_shape[-1];
    const uint32_t batch_size = metadata_shape[1];
    const uint32_t selected_experts_k = metadata_shape[-1];

    const auto& axis = operation_attributes.axis;
    const uint32_t batch_replicate_dim = axis.has_value() ? mesh_device->shape()[axis.value()] : 1;
    const uint32_t total_batch_size = batch_size * batch_replicate_dim;

    TT_ASSERT(total_batch_size % num_devices == 0);
    const uint32_t total_batch_per_device_size = total_batch_size / num_devices;

    auto output_shape = ttnn::Shape({selected_experts_k, total_batch_per_device_size, 1, hidden_size});

    auto mem_config = operation_attributes.output_mem_config;
    return TensorSpec(
        Shape(output_shape),
        TensorLayout(
            tensor_args.input_tensor.get_dtype(), PageConfig(tensor_args.input_tensor.get_layout()), mem_config));
}

AllToAllCombineDeviceOperation::tensor_return_value_t AllToAllCombineDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto output_spec = compute_output_specs(operation_attributes, tensor_args);
    auto output_tensor = create_device_tensor(output_spec, tensor_args.input_tensor.device());
    return output_tensor;
}

std::tuple<AllToAllCombineDeviceOperation::operation_attributes_t, AllToAllCombineDeviceOperation::tensor_args_t>
AllToAllCombineDeviceOperation::invoke(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& expert_mapping_tensor,
    const ttnn::Tensor& expert_metadata_tensor,
    const uint32_t num_links,
    const tt::tt_fabric::Topology topology,
    const ttnn::MemoryConfig& memory_config,
    // tt::tt_metal::SubDeviceId subdevice_id,
    const GlobalSemaphore& global_semaphore,
    const std::optional<uint32_t>& axis) {
    return {
        operation_attributes_t{//.subdevice_id = std::move(subdevice_id),
                               .output_mem_config = memory_config,
                               .axis = axis,
                               .num_links = num_links,
                               .topology = topology,
                               .cross_device_semaphore = global_semaphore},
        tensor_args_t{
            .input_tensor = input_tensor,
            .mapping_tensor = expert_mapping_tensor,
            .metadata_tensor = expert_metadata_tensor}};
}

}  // namespace ttnn::operations::ccl
