// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <utility>

#include "ttnn/tensor/types.hpp"
#include "reduce_scatter_device_operation.hpp"
#include "cpp/ttnn/operations/data_movement/common/common.hpp"

namespace ttnn::operations::ccl {

ReduceScatterDeviceOperation::program_factory_t ReduceScatterDeviceOperation::select_program_factory(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return ReduceScatterProgram{};
}

void ReduceScatterDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate_on_program_cache_hit(operation_attributes, tensor_args);
    auto output_specs = compute_output_specs(operation_attributes, tensor_args);
    auto input_tensor = tensor_args.input_tensor;
    uint32_t target_ring_size = ::ttnn::ccl::get_topological_dimension(input_tensor, operation_attributes.cluster_axis);
    ttnn::operations::experimental::ccl::reduce_scatter_minimal_async::detail::reduce_scatter_common_validates(
        input_tensor,
        operation_attributes.topology,
        operation_attributes.dim,
        operation_attributes.num_links,
        target_ring_size,
        operation_attributes.memory_config,
        tensor_args.optional_output_tensor);
}

void ReduceScatterDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    if (tensor_args.optional_output_tensor.has_value()) {
        auto output_specs = compute_output_specs(operation_attributes, tensor_args);
        TT_FATAL(
            tensor_args.optional_output_tensor.value().tensor_spec() == output_specs.at(1),
            "Output tensor spec {} does not match computed output spec {}",
            tensor_args.optional_output_tensor.value().tensor_spec(),
            output_specs.at(1));
    }
}

ReduceScatterDeviceOperation::spec_return_value_t ReduceScatterDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input_tensor;
    auto inter_shape = input_tensor.tensor_spec().logical_shape();

    if (operation_attributes.topology == ::ttnn::ccl::Topology::Linear) {
        inter_shape[0] *= 2;
    }

    auto output_shape = input_tensor.logical_shape();
    uint32_t target_ring_size = ::ttnn::ccl::get_topological_dimension(input_tensor, operation_attributes.cluster_axis);
    output_shape[operation_attributes.dim] /= target_ring_size;
    // For now default to tt::tt_metal::BufferType::DRAM to prevent CB overflows.
    // TODO: add L1 estimation similar to the one in all_to_all_dispatch and choose to use L1 as an intermediate buffer
    // if enough space is available #30043. L1 estimation has to be done outside the program cache
    auto mem_config = operation_attributes.memory_config;
    auto intermediate_mem_config =
        MemoryConfig(tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::DRAM);
    return {
        TensorSpec(
            inter_shape,
            tt::tt_metal::TensorLayout(
                input_tensor.dtype(), input_tensor.tensor_spec().page_config(), intermediate_mem_config)),
        TensorSpec(
            output_shape,
            tt::tt_metal::TensorLayout(input_tensor.dtype(), input_tensor.tensor_spec().page_config(), mem_config)),
    };
}

ReduceScatterDeviceOperation::tensor_return_value_t ReduceScatterDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    auto output_specs = compute_output_specs(operation_attributes, tensor_args);
    ttnn::Tensor output_tensor = tensor_args.optional_output_tensor.value_or(
        create_device_tensor(output_specs.at(1), tensor_args.input_tensor.device()));
    ttnn::Tensor intermediate_tensor = create_device_tensor(output_specs.at(0), tensor_args.input_tensor.device());
    return {intermediate_tensor, output_tensor};
}

ttsl::hash::hash_t ReduceScatterDeviceOperation::compute_program_hash(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    auto input_tensor = tensor_args.input_tensor;
    auto subdevice_id = operation_attributes.subdevice_id;
    auto* mesh_device = input_tensor.device();
    auto sd_id = subdevice_id.value_or(mesh_device->get_sub_device_ids().at(0));
    auto subdevice_core_range_set = mesh_device->worker_cores(tt::tt_metal::HalProgrammableCoreType::TENSIX, sd_id);
    return tt::tt_metal::operation::hash_operation<ReduceScatterDeviceOperation>(
        operation_attributes.dim,
        operation_attributes.num_links,
        operation_attributes.cluster_axis,
        operation_attributes.memory_config,
        subdevice_core_range_set,
        operation_attributes.topology,
        input_tensor);
}

std::tuple<ReduceScatterDeviceOperation::operation_attributes_t, ReduceScatterDeviceOperation::tensor_args_t>
ReduceScatterDeviceOperation::invoke(
    const ttnn::Tensor& input_tensor,
    uint32_t dim,
    std::optional<uint32_t> cluster_axis,
    const std::optional<tt::tt_metal::SubDeviceId>& subdevice_id,
    const ttnn::MemoryConfig& memory_config,
    const std::optional<ttnn::Tensor>& optional_output_tensor,
    uint32_t num_links,
    tt::tt_fabric::Topology topology) {
    return {
        operation_attributes_t{
            .memory_config = memory_config,
            .dim = dim,
            .cluster_axis = cluster_axis,
            .subdevice_id = subdevice_id,
            .topology = topology,
            .num_links = num_links},
        tensor_args_t{.input_tensor = input_tensor, .optional_output_tensor = optional_output_tensor}};
}

}  // namespace ttnn::operations::ccl
