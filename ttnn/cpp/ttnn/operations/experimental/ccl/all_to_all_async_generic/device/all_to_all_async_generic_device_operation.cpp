// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "all_to_all_async_generic_device_operation.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

namespace ttnn::operations::experimental::ccl {

AllToAllAsyncGenericDeviceOperation::program_factory_t AllToAllAsyncGenericDeviceOperation::select_program_factory(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& /*tensor_args*/) {
    return AllToAllAsyncGenericProgram{};
}

void AllToAllAsyncGenericDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate_on_program_cache_hit(operation_attributes, tensor_args);

    const auto& input_tensor = tensor_args.input_tensor;
    const auto& page_size = input_tensor.buffer()->page_size();
    const auto& input_shape = input_tensor.logical_shape();
    auto rank = input_shape.rank();

    TT_FATAL(operation_attributes.in_dim >= 0 && operation_attributes.in_dim < rank, "in_dim out of range");
    TT_FATAL(operation_attributes.out_dim >= 0 && operation_attributes.out_dim < rank, "out_dim out of range");

    TT_FATAL(page_size % input_tensor.buffer()->alignment() == 0, "AllToAllAsync currently requires aligned pages");

    TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "Operands to all_to_all_async must be on device");
    TT_FATAL(input_tensor.buffer() != nullptr, "Operands to all_to_all_async must be allocated in buffers on device");
    TT_FATAL(operation_attributes.num_links == 1, "num_links must be 1, but is {}", operation_attributes.num_links);

    TT_FATAL(
        input_shape[operation_attributes.out_dim] % operation_attributes.num_devices == 0,
        "AllToAllAsync: input tensor dimension {} must be divisible by num_devices {}",
        input_shape[operation_attributes.out_dim],
        operation_attributes.num_devices);
    TT_FATAL(input_tensor.layout() == Layout::TILE, "Unsupported input layout {}.", input_tensor.layout());

    // recreate output shape to cover optional output buffer
    auto output_shape = input_tensor.logical_shape();
    output_shape[operation_attributes.in_dim] *= operation_attributes.num_devices;
    output_shape[operation_attributes.out_dim] /= operation_attributes.num_devices;

    // Check padding support, currently supported only on height
    auto last_dim = rank - 1;
    auto second_last_dim = rank - 2;
    TT_FATAL(
        operation_attributes.in_dim != second_last_dim || input_shape[operation_attributes.in_dim] % 16 == 0,
        "{} dimension support only 0 or 16 padding, so must be divisible by 16. Input tensor shape {} , but has {} "
        "padding",
        operation_attributes.in_dim,
        input_shape,
        input_shape[operation_attributes.in_dim] % 32);
    TT_FATAL(
        operation_attributes.out_dim != second_last_dim || output_shape[operation_attributes.out_dim] % 16 == 0,
        "{} dimension support only 0 or 16 padding, so must be divisible by 16. Output tensor shape {} , but has {} "
        "padding",
        operation_attributes.out_dim,
        output_shape,
        output_shape[operation_attributes.out_dim] % 32);
    TT_FATAL(
        operation_attributes.in_dim != last_dim || input_shape[operation_attributes.in_dim] % 32 == 0,
        "{} dimension doesn't support padding, so must be divisible by 32. Input tensor shape {} , but has {} padding",
        operation_attributes.in_dim,
        input_shape,
        input_shape[operation_attributes.in_dim] % 32);
    TT_FATAL(
        operation_attributes.out_dim != last_dim || output_shape[operation_attributes.out_dim] % 32 == 0,
        "{} dimension doesn't support padding, so must be divisible by 32. Output tensor shape {} , but has {} padding",
        operation_attributes.out_dim,
        output_shape,
        output_shape[operation_attributes.out_dim] % 32);
}

void AllToAllAsyncGenericDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input_tensor;
    const auto& persistent_output_buffer = tensor_args.persistent_output_buffer;

    if (persistent_output_buffer.has_value()) {
        const auto& output_tensor = persistent_output_buffer.value();

        TT_FATAL(
            output_tensor.storage_type() == StorageType::DEVICE,
            "Output tensor for all_to_all_async must be on device");
        TT_FATAL(
            output_tensor.buffer()->buffer_type() == BufferType::DRAM,
            "Output tensor for all_to_all_async must be in DRAM, but is in {}",
            output_tensor.buffer()->buffer_type());
        TT_FATAL(output_tensor.layout() == Layout::TILE, "Unsupported output layout {}.", output_tensor.layout());

        TT_FATAL(output_tensor.dtype() == input_tensor.dtype(), "Output tensor dtype must match input tensor dtype");
        TT_FATAL(
            output_tensor.memory_config() == operation_attributes.output_mem_config,
            "Output tensor memory config must match specified output_mem_config");

        const auto& output_shape = output_tensor.logical_shape();
        auto expected_output_shape = input_tensor.logical_shape();
        expected_output_shape[operation_attributes.in_dim] *= operation_attributes.num_devices;
        expected_output_shape[operation_attributes.out_dim] /= operation_attributes.num_devices;
        TT_FATAL(
            output_shape == expected_output_shape,
            "Output tensor shape {} must match expected output tensor shape {} for AllToAllAsync",
            output_shape,
            expected_output_shape);
    }
}

AllToAllAsyncGenericDeviceOperation::spec_return_value_t AllToAllAsyncGenericDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    if (tensor_args.persistent_output_buffer.has_value()) {
        return tensor_args.persistent_output_buffer->tensor_spec();
    }

    const auto& input_tensor = tensor_args.input_tensor;
    auto shape = input_tensor.logical_shape();
    shape[operation_attributes.in_dim] *= operation_attributes.num_devices;
    shape[operation_attributes.out_dim] /= operation_attributes.num_devices;
    return TensorSpec(
        shape,
        tt::tt_metal::TensorLayout(
            input_tensor.dtype(), input_tensor.tensor_spec().page_config(), operation_attributes.output_mem_config));
}

AllToAllAsyncGenericDeviceOperation::tensor_return_value_t AllToAllAsyncGenericDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    if (tensor_args.persistent_output_buffer.has_value()) {
        return tensor_args.persistent_output_buffer.value();
    }
    return create_device_tensor(
        compute_output_specs(operation_attributes, tensor_args), tensor_args.input_tensor.device());
}

tt::stl::hash::hash_t AllToAllAsyncGenericDeviceOperation::compute_program_hash(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    log_trace(tt::LogOp, "AllToAllAsyncGenericDeviceOperation::compute_program_hash is called");

    auto subdevice_id = operation_attributes.sub_device_id;
    auto* mesh_device = tensor_args.input_tensor.device();
    auto sd_id = subdevice_id.value_or(mesh_device->get_sub_device_ids().at(0));
    auto subdevice_core_range_set = mesh_device->worker_cores(tt::tt_metal::HalProgrammableCoreType::TENSIX, sd_id);

    auto program_factory = select_program_factory(operation_attributes, tensor_args);

    return tt::tt_metal::operation::hash_operation<AllToAllAsyncGenericDeviceOperation>(
        operation_attributes.in_dim,
        operation_attributes.out_dim,
        operation_attributes.num_links,
        operation_attributes.num_devices,
        operation_attributes.output_mem_config,
        operation_attributes.topology,
        operation_attributes.cluster_axis,
        subdevice_core_range_set,
        tensor_args,
        program_factory.index());
}

}  // namespace ttnn::operations::experimental::ccl

namespace ttnn::prim {

ttnn::operations::experimental::ccl::AllToAllAsyncGenericDeviceOperation::tensor_return_value_t
all_to_all_async_generic(
    const ttnn::Tensor& input_tensor,
    const std::optional<Tensor>& persistent_output_buffer,
    int32_t in_dim,
    int32_t out_dim,
    uint32_t num_links,
    const std::optional<MemoryConfig>& memory_config,
    ttnn::ccl::Topology topology,
    std::optional<tt::tt_metal::SubDeviceId> sub_device_id,
    std::optional<uint32_t> cluster_axis) {
    using OperationType = ttnn::operations::experimental::ccl::AllToAllAsyncGenericDeviceOperation;
    uint32_t num_devices = ttnn::ccl::get_topological_dimension(input_tensor, cluster_axis);
    TT_FATAL(
        num_devices > 1,
        "all_to_all_async is a collective operation and requires more than 1 device, but has {}",
        num_devices);

    auto operation_attributes = OperationType::operation_attributes_t{
        .in_dim = static_cast<uint32_t>(in_dim),
        .out_dim = static_cast<uint32_t>(out_dim),
        .num_links = num_links,
        .num_devices = num_devices,
        .output_mem_config = memory_config.value_or(input_tensor.memory_config()),
        .topology = topology,
        .sub_device_id = sub_device_id,
        .cluster_axis = cluster_axis};
    auto tensor_args = OperationType::tensor_args_t{
        .input_tensor = input_tensor, .persistent_output_buffer = persistent_output_buffer};

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
