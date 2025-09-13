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
    auto page_size = input_tensor.buffer()->page_size();
    TT_FATAL(
        page_size % input_tensor.buffer()->alignment() == 0,
        "page_size {} must be divisible by alignment {}",
        page_size,
        input_tensor.buffer()->alignment());
    TT_FATAL(operation_attributes.cluster_axis.has_value(), "cluster_axis must be set");
    TT_FATAL(operation_attributes.dim == 3, "dim must be 3");
    TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "input_tensor must be on device");
    TT_FATAL(input_tensor.buffer() != nullptr, "input_tensor must have a buffer");
    TT_FATAL(operation_attributes.num_links > 0, "num_links must be greater than 0");
    TT_FATAL(
        operation_attributes.topology == ::ttnn::ccl::Topology::Ring ||
            operation_attributes.topology == ::ttnn::ccl::Topology::Linear,
        "topology must be Ring or Linear");
    TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "input_tensor must be on device");
    TT_FATAL(input_tensor.buffer() != nullptr, "input_tensor must have a buffer");

    if (input_tensor.memory_config().memory_layout() == TensorMemoryLayout::BLOCK_SHARDED) {
        TT_FATAL(input_tensor.memory_config().buffer_type() == BufferType::L1, "DRAM block sharding is not supported");
    }

    uint32_t axis = operation_attributes.cluster_axis.value();
    log_debug(tt::LogOp, "axis: {}", axis);
    TT_FATAL(axis == 0 || axis == 1, "axis must be 0 or 1");
    auto mesh_view = input_tensor.device()->get_view();
    uint32_t reduction_devices = axis == 0 ? mesh_view.num_rows() : mesh_view.num_cols();
    log_debug(tt::LogOp, "reduction_devices: {}", reduction_devices);

    if (tensor_args.optional_output_tensor.has_value()) {
        const auto& output_tensor = tensor_args.optional_output_tensor.value();

        TT_FATAL(
            output_tensor.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED ||
                output_tensor.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED ||
                output_tensor.memory_config().memory_layout() == TensorMemoryLayout::BLOCK_SHARDED ||
                output_tensor.memory_config().memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED,
            "Unsupported output tensor memory layout {}.",
            output_tensor.memory_config().memory_layout());

        TT_FATAL(
            output_tensor.storage_type() == StorageType::DEVICE,
            "Operands to reduce_scatter_minimal_async need to be on device!");
        TT_FATAL(
            output_tensor.layout() == input_tensor.layout(),
            "Error, Output tensor layout should be same as input tensor layout but has {}",
            output_tensor.layout());
        TT_FATAL(
            output_tensor.dtype() == input_tensor.dtype(),
            "Error, Output tensor dtype should be same as input tensor dtype but has {}",
            output_tensor.dtype());
        TT_FATAL(
            output_tensor.tensor_spec().page_config() == input_tensor.tensor_spec().page_config(),
            "Error, Output tensor page config should be same as input tensor page config but has {}",
            output_tensor.tensor_spec().page_config());

        // check the output tensor size
        auto output_shape = output_tensor.padded_shape();
        auto input_shape = input_tensor.padded_shape();
        TT_FATAL(
            output_shape.size() == input_shape.size(),
            "Error, Output tensor shape should have same number of dimensions as input tensor but has {}",
            output_shape.size());
        for (size_t i = 0; i < input_shape.size(); ++i) {
            if (i == operation_attributes.dim) {
                TT_FATAL(
                    output_shape[i] == input_shape[i] / reduction_devices,
                    "Error, Output tensor shape at dimension {} should be {} but has {}",
                    i,
                    input_shape[i] / reduction_devices,
                    output_shape[i]);
            } else {
                TT_FATAL(
                    output_shape[i] == input_shape[i],
                    "Error, Output tensor shape at dimension {} should be {} but has {}",
                    i,
                    input_shape[i],
                    output_shape[i]);
            }
        }

        // Don't support DRAM block sharding
        if (output_tensor.memory_config().memory_layout() == TensorMemoryLayout::BLOCK_SHARDED) {
            TT_FATAL(
                output_tensor.memory_config().buffer_type() == BufferType::L1,
                "We don't support output DRAM block sharding");
        }
    }
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
    auto mesh_device = input_tensor.device();
    auto mesh_view = mesh_device->get_view();
    auto inter_shape = input_tensor.tensor_spec().logical_shape();

    if (operation_attributes.topology == ::ttnn::ccl::Topology::Linear) {
        inter_shape[0] *= 2;
    }

    auto output_shape = input_tensor.logical_shape();
    uint32_t reduction_devices = mesh_view.num_devices();
    if (operation_attributes.cluster_axis.has_value()) {
        uint32_t axis = operation_attributes.cluster_axis.value();
        log_debug(tt::LogOp, "axis: {}", axis);
        TT_FATAL(axis == 0 || axis == 1, "axis must be 0 or 1");
        reduction_devices = axis == 0 ? mesh_view.num_rows() : mesh_view.num_cols();
    }
    output_shape[operation_attributes.dim] /= reduction_devices;
    // For now default to tt::tt_metal::BufferType::DRAM to prevent CB overflows.
    // TODO: add L1 estimation similar to the one in all_to_all_dispatch and choose to use L1 as an intermediate buffer
    // if enough space is available. L1 estimation has to be done outside the program cache
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
    auto mesh_device = input_tensor.device();
    auto sd_id = subdevice_id.value_or(mesh_device->get_sub_device_ids().at(0));
    auto subdevice_core_range_set = mesh_device->worker_cores(tt::tt_metal::HalProgrammableCoreType::TENSIX, sd_id);
    return tt::tt_metal::operation::hash_operation<ReduceScatterDeviceOperation>(
        operation_attributes.dim,
        operation_attributes.num_links,
        operation_attributes.cluster_axis,
        operation_attributes.memory_config,
        subdevice_core_range_set,
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
