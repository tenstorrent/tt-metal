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

AllToAllCombineDeviceOperation::program_factory_t AllToAllCOmbineDeviceOperation::select_program_factory(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return AllToAllCombineSparse{};
}

//! TODO
void AllToAllCombineDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {}

// !TODO
void AllToAllCombineDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {}

AllToAllDCombineDeviceOperation::spec_return_value_t AllToAllCombineDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    using namespace tt::tt_metal;

    auto input_tensor = tensor_args.input_tensor;
    auto input_shape = input_tensor.get_tensor_spec().logical_shape();
    auto indices_shape = tensor_args.expert_indices_tensor.get_tensor_spec().logical_shape();
    auto mapping_shape = tensor_args.expert_mapping_tensor.get_tensor_spec().logical_shape();

    auto mesh_device = input_tensor.mesh_device();
    const auto& mesh_view = mesh_device->get_view();
    
    const auto num_devices = mesh_view.num_devices();

    const uint32_t hidden_size = input_shape[-1];
    const uint32_t batch_size = input_shape[0] * num_devices;
    const uint32_t selected_experts_k = indices_shape[-1];
    
    TT_ASSERT(batch_size%num_devices==0);

    auto output_shape = ttnn::Shape({selected_experts_k, batch_size/num_devices, 1, hidden_size});

    auto mem_config = operation_attributes.output_mem_config;
    return {
        TensorSpec(
            Shape(metadata_shape),
            TensorLayout(
                tensor_args.expert_indices_tensor.get_dtype(),
                PageConfig(tensor_args.expert_indices_tensor.get_layout()),
                mem_config))};
}

AllToAllCombineDeviceOperation::tensor_return_value_t AllToAllCombineDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto output_spec = compute_output_specs(operation_attributes, tensor_args);
    auto output_tensor = create_device_tensor(output_spec[0], tensor_args.input_tensor.device());
    return {output_tensor, metadata_tensor};
}

std::tuple<AllToAllCombineDeviceOperation::operation_attributes_t, AllToAllCombineDeviceOperation::tensor_args_t>
AllToAllDispatchDeviceOperation::invoke(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& expert_indices_tensor,
    const ttnn::Tensor& expert_mapping_tensor,
    const uint32_t num_links,
    const tt::tt_fabric::Topology topology,
    const ttnn::MemoryConfig& memory_config,
    tt::tt_metal::SubDeviceId subdevice_id,
    const GlobalSemaphore& global_semaphore) {
    return {
        operation_attributes_t{
            .subdevice_id = std::move(subdevice_id),
            .output_mem_config = memory_config,
            .num_links = num_links,
            .topology = topology,
            .cross_device_semaphore = global_semaphore},
        tensor_args_t{
            .input_tensor = input_tensor,
            .expert_indices_tensor = expert_indices_tensor,
            .expert_mapping_tensor = expert_mapping_tensor}};
}

}  // namespace ttnn::operations::ccl
