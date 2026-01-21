// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "all_to_all_async_device_operation.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/tensor/tensor_utils.hpp"

namespace ttnn::experimental::prim {

AllToAllAsyncDeviceOperation::program_factory_t AllToAllAsyncDeviceOperation::select_program_factory(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& /*tensor_args*/) {
    return AllToAllAsyncProgram{};
}

void AllToAllAsyncDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate_on_program_cache_hit(operation_attributes, tensor_args);

    const auto& input_tensor = tensor_args.input_tensor;
    const auto& page_size = input_tensor.buffer()->page_size();
    const auto& input_shape = input_tensor.logical_shape();
    auto rank = input_shape.rank();

    TT_FATAL(
        operation_attributes.in_dim >= 0 && operation_attributes.in_dim < rank,
        "AllToAllAsync: in_dim must be in range [0, {}), but is {}",
        rank,
        operation_attributes.in_dim);
    TT_FATAL(
        operation_attributes.out_dim >= 0 && operation_attributes.out_dim < rank,
        "AllToAllAsync: out_dim must be in range [0, {}), but is {}",
        rank,
        operation_attributes.out_dim);

    TT_FATAL(page_size % input_tensor.buffer()->alignment() == 0, "AllToAllAsync currently requires aligned pages");

    TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "Operands to all_to_all_async must be on device");
    TT_FATAL(input_tensor.buffer() != nullptr, "Operands to all_to_all_async must be allocated in buffers on device");
    TT_FATAL(
        operation_attributes.num_links > 0,
        "Number of links must be greater than 0, but is {}",
        operation_attributes.num_links);
    TT_FATAL(
        operation_attributes.num_links <= input_tensor.device()->compute_with_storage_grid_size().y,
        "Worker cores used by links are parallelized over rows, num_links ({}) exceeds available rows ({})",
        operation_attributes.num_links,
        input_tensor.device()->compute_with_storage_grid_size().y);

    TT_FATAL(
        input_tensor.buffer()->buffer_type() == BufferType::DRAM,
        "AllToAllAsync: Input tensor must be in DRAM, but is in {}",
        input_tensor.buffer()->buffer_type());
    TT_FATAL(input_tensor.layout() == Layout::TILE, "Unsupported input layout {}.", input_tensor.layout());
    TT_FATAL(
        input_tensor.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "Unsupported input memory layout {}.",
        input_tensor.memory_config().memory_layout());

    TT_FATAL(
        operation_attributes.in_dim == 2 || operation_attributes.in_dim == 3,
        "AllToAllAsync: in_dim must be 2 or 3, but is {}",
        operation_attributes.in_dim);
    TT_FATAL(
        operation_attributes.out_dim == 2 || operation_attributes.out_dim == 3,
        "AllToAllAsync: out_dim must be 2 or 3, but is {}",
        operation_attributes.out_dim);
    TT_FATAL(
        operation_attributes.in_dim != operation_attributes.out_dim,
        "AllToAllAsync: in_dim and out_dim must be different, but are both {}",
        operation_attributes.in_dim);
    TT_FATAL(input_tensor.padded_shape().size() == 4, "AllToAllAsync: input tensor must have 4 dimensions");

    TT_FATAL(
        input_tensor.padded_shape()[operation_attributes.out_dim] % operation_attributes.ring_size == 0,
        "AllToAllAsync: input tensor dimension {} must be divisible by ring_size {}",
        input_tensor.padded_shape()[operation_attributes.out_dim],
        operation_attributes.ring_size);

    // Validate output buffers
    const auto& intermediate_buffer = tensor_args.persistent_intermediate_buffer;
    const auto& output_buffer = tensor_args.persistent_output_buffer;

    TT_FATAL(
        intermediate_buffer.storage_type() == StorageType::DEVICE,
        "Intermediate buffer for all_to_all_async must be on device");
    TT_FATAL(
        intermediate_buffer.buffer()->buffer_type() == BufferType::DRAM,
        "Intermediate buffer for all_to_all_async must be in DRAM, but is in {}",
        intermediate_buffer.buffer()->buffer_type());
    TT_FATAL(intermediate_buffer.layout() == Layout::TILE, "Unsupported intermediate buffer layout.");
    TT_FATAL(
        intermediate_buffer.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "Unsupported intermediate buffer memory layout.");
    TT_FATAL(
        intermediate_buffer.dtype() == input_tensor.dtype(), "Intermediate buffer dtype must match input tensor dtype");

    TT_FATAL(
        output_buffer.storage_type() == StorageType::DEVICE, "Output buffer for all_to_all_async must be on device");
    TT_FATAL(
        output_buffer.buffer()->buffer_type() == BufferType::DRAM,
        "Output buffer for all_to_all_async must be in DRAM, but is in {}",
        output_buffer.buffer()->buffer_type());
    TT_FATAL(output_buffer.layout() == Layout::TILE, "Unsupported output buffer layout.");
    TT_FATAL(
        output_buffer.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "Unsupported output buffer memory layout.");
    TT_FATAL(output_buffer.dtype() == input_tensor.dtype(), "Output buffer dtype must match input tensor dtype");
    TT_FATAL(
        output_buffer.memory_config() == operation_attributes.output_mem_config,
        "Output buffer memory config must match specified output_mem_config");

    // Validate shapes
    const auto& intermediate_shape = intermediate_buffer.padded_shape();
    const auto& output_shape = output_buffer.padded_shape();
    TT_FATAL(intermediate_shape.size() == 4, "AllToAllAsync: intermediate buffer must have 4 dimensions");
    TT_FATAL(output_shape.size() == 4, "AllToAllAsync: output buffer must have 4 dimensions");

    auto expected_shape = input_tensor.padded_shape();
    expected_shape[operation_attributes.in_dim] *= operation_attributes.ring_size;
    expected_shape[operation_attributes.out_dim] /= operation_attributes.ring_size;

    TT_FATAL(
        intermediate_shape == expected_shape,
        "Intermediate buffer shape {} must match expected shape {} for AllToAllAsync",
        intermediate_shape,
        expected_shape);
    TT_FATAL(
        output_shape == expected_shape,
        "Output buffer shape {} must match expected shape {} for AllToAllAsync",
        output_shape,
        expected_shape);

    TT_FATAL(
        operation_attributes.num_links == 1,
        "AllToAllAsync: num_links must be 1, but is {}",
        operation_attributes.num_links);
}

void AllToAllAsyncDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input_tensor;
    const auto& intermediate_buffer = tensor_args.persistent_intermediate_buffer;
    const auto& output_buffer = tensor_args.persistent_output_buffer;

    TT_FATAL(
        intermediate_buffer.storage_type() == StorageType::DEVICE,
        "Intermediate buffer for all_to_all_async must be on device");
    TT_FATAL(
        output_buffer.storage_type() == StorageType::DEVICE, "Output buffer for all_to_all_async must be on device");
    TT_FATAL(
        intermediate_buffer.buffer()->buffer_type() == BufferType::DRAM,
        "Intermediate buffer for all_to_all_async must be in DRAM, but is in {}",
        intermediate_buffer.buffer()->buffer_type());
    TT_FATAL(
        output_buffer.buffer()->buffer_type() == BufferType::DRAM,
        "Output buffer for all_to_all_async must be in DRAM, but is in {}",
        output_buffer.buffer()->buffer_type());
    TT_FATAL(intermediate_buffer.layout() == Layout::TILE, "Unsupported intermediate buffer layout.");
    TT_FATAL(output_buffer.layout() == Layout::TILE, "Unsupported output buffer layout.");
    TT_FATAL(
        intermediate_buffer.dtype() == input_tensor.dtype(), "Intermediate buffer dtype must match input tensor dtype");
    TT_FATAL(output_buffer.dtype() == input_tensor.dtype(), "Output buffer dtype must match input tensor dtype");
    TT_FATAL(
        output_buffer.memory_config() == operation_attributes.output_mem_config,
        "Output buffer memory config must match specified output_mem_config");
}

AllToAllAsyncDeviceOperation::spec_return_value_t AllToAllAsyncDeviceOperation::compute_output_specs(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& tensor_args) {
    // Return spec for the output buffer (persistent_output_buffer)
    return tensor_args.persistent_output_buffer.tensor_spec();
}

AllToAllAsyncDeviceOperation::tensor_return_value_t AllToAllAsyncDeviceOperation::create_output_tensors(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& tensor_args) {
    // Return the pre-allocated output buffer
    return tensor_args.persistent_output_buffer;
}

tt::stl::hash::hash_t AllToAllAsyncDeviceOperation::compute_program_hash(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    log_trace(tt::LogOp, "AllToAllAsyncDeviceOperation::compute_program_hash is called");

    auto subdevice_id = operation_attributes.sub_device_id;
    auto* mesh_device = tensor_args.input_tensor.device();
    auto sd_id = subdevice_id.value_or(mesh_device->get_sub_device_ids().at(0));
    auto subdevice_core_range_set = mesh_device->worker_cores(tt::tt_metal::HalProgrammableCoreType::TENSIX, sd_id);

    auto program_factory = select_program_factory(operation_attributes, tensor_args);

    return tt::tt_metal::operation::hash_operation<AllToAllAsyncDeviceOperation>(
        operation_attributes.in_dim,
        operation_attributes.out_dim,
        operation_attributes.num_links,
        operation_attributes.ring_size,
        operation_attributes.output_mem_config,
        operation_attributes.topology,
        subdevice_core_range_set,
        tensor_args,
        program_factory.index());
}

Tensor all_to_all_async(
    const ttnn::Tensor& input_tensor,
    ttnn::Tensor& persistent_intermediate_buffer,
    ttnn::Tensor& persistent_output_buffer,
    int32_t in_dim,
    int32_t out_dim,
    const ttnn::GlobalSemaphore& multi_device_global_semaphore,
    uint32_t num_links,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    ttnn::ccl::Topology topology,
    std::optional<tt::tt_metal::SubDeviceId> sub_device_id) {
    using OperationType = AllToAllAsyncDeviceOperation;

    // Normalize dimensions
    int32_t rank = input_tensor.logical_shape().rank();
    int32_t norm_in_dim = (in_dim < 0) ? rank + in_dim : in_dim;
    int32_t norm_out_dim = (out_dim < 0) ? rank + out_dim : out_dim;

    TT_FATAL(norm_in_dim >= 0 && norm_in_dim < rank, "Invalid in_dim: {}", in_dim);
    TT_FATAL(norm_out_dim >= 0 && norm_out_dim < rank, "Invalid out_dim: {}", out_dim);

    // Compute ring_size from number of devices
    std::vector<IDevice*> devices = ttnn::ccl::get_active_physical_devices(input_tensor);
    uint32_t num_devices = devices.size();
    TT_FATAL(num_devices > 0, "all_to_all_async requires at least one device, but has {}", num_devices);

    ttnn::ccl::Topology ccl_topology = topology;
    if (num_devices == 1) {
        TT_THROW("all_to_all_async is a collective operation and requires more than 1 device.");
    }
    if (num_devices == 2 && topology == ttnn::ccl::Topology::Ring) {
        log_warning(tt::LogOp, "Using Linear topology for AllToAllAsync with 2 devices instead of Ring.");
        ccl_topology = ttnn::ccl::Topology::Linear;
    }

    auto operation_attributes = OperationType::operation_attributes_t(
        static_cast<uint32_t>(norm_in_dim),
        static_cast<uint32_t>(norm_out_dim),
        num_links,
        num_devices,
        memory_config.value_or(input_tensor.memory_config()),
        ccl_topology,
        multi_device_global_semaphore,
        sub_device_id);
    auto tensor_args = OperationType::tensor_args_t{
        .input_tensor = input_tensor,
        .persistent_intermediate_buffer = persistent_intermediate_buffer,
        .persistent_output_buffer = persistent_output_buffer};

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::experimental::prim
