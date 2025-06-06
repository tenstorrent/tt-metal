// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "all_to_all_async_op.hpp"
#include "ttnn/operations/functions.hpp"
#include "ttnn/operations/math.hpp"
#include "ttnn/global_semaphore.hpp"

#include "ttnn/tensor/tensor_utils.hpp"

namespace ttnn {

// Implementation of AllToAllAsync methods

void AllToAllAsync::validate_with_output_tensors(
    const std::vector<Tensor>& input_tensors, const std::vector<std::optional<Tensor>>& output_tensors) const {
    TT_FATAL(input_tensors.size() == 1, "AllToAllAsync: Input tensor size must be 1, but is {}", input_tensors.size());
    const auto& input_tensor = input_tensors[0];
    const auto& layout = input_tensor.layout();
    const auto& dtype = input_tensor.dtype();
    const auto& page_size = input_tensor.buffer()->page_size();
    TT_FATAL(page_size % input_tensor.buffer()->alignment() == 0, "AllToAllAsync currently requires aligned pages");

    TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "Operands to all_to_all_async must be on device");
    TT_FATAL(input_tensor.buffer() != nullptr, "Operands to all_to_all_async must be allocated in buffers on device");
    TT_FATAL(this->num_links > 0, "Number of links must be greater than 0, but is {}", this->num_links);
    TT_FATAL(
        this->num_links <= input_tensor.device()->compute_with_storage_grid_size().y,
        "Worker cores used by links are parallelized over rows, num_links ({}) exceeds available rows ({})",
        this->num_links,
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

    TT_FATAL(this->in_dim == 2 || this->in_dim == 3, "AllToAllAsync: in_dim must be 2 or 3, but is {}", this->in_dim);
    TT_FATAL(
        this->out_dim == 2 || this->out_dim == 3, "AllToAllAsync: out_dim must be 2 or 3, but is {}", this->out_dim);
    TT_FATAL(
        this->in_dim != this->out_dim,
        "AllToAllAsync: in_dim and out_dim must be different, but are both {}",
        this->in_dim);
    TT_FATAL(input_tensor.padded_shape().size() == 4, "AllToAllAsync: input tensor must have 4 dimensions");

    TT_FATAL(
        input_tensor.padded_shape()[this->out_dim] % this->ring_size == 0,
        "AllToAllAsync: input tensor dimension {} must be divisible by ring_size {}",
        input_tensor.padded_shape()[this->out_dim],
        this->ring_size);

    // Output tensor validation
    TT_FATAL(
        output_tensors.size() == 2,
        "AllToAllAsync: Number of output tensors must be 2, but is {}",
        output_tensors.size());

    for (const auto& maybe_output_tensor : output_tensors) {
        TT_FATAL(maybe_output_tensor.has_value(), "Output tensor must be provided");
        const auto& output_tensor = maybe_output_tensor.value();
        TT_FATAL(
            output_tensor.storage_type() == StorageType::DEVICE,
            "Output tensor for all_to_all_async must be on device");
        TT_FATAL(
            output_tensor.buffer()->buffer_type() == BufferType::DRAM,
            "Output tensor for all_to_all_async must be in DRAM, but is in {}",
            output_tensor.buffer()->buffer_type());
        TT_FATAL(output_tensor.layout() == Layout::TILE, "Unsupported output layout {}.", output_tensor.layout());
        TT_FATAL(
            output_tensor.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
            "Unsupported output memory layout {}.",
            output_tensor.memory_config().memory_layout());
        TT_FATAL(output_tensor.dtype() == dtype, "Output tensor dtype must match input tensor dtype");
        TT_FATAL(
            output_tensor.memory_config() == this->output_mem_config,
            "Output tensor memory config must match specified output_mem_config");

        // For AllToAll, the shape of the *local* tensor shard should typically be the same.
        // Global logical shape also remains the same.
        auto output_shape = output_tensor.padded_shape();
        TT_FATAL(output_shape.size() == 4, "AllToAllAsync: output tensor must have 4 dimensions");
        auto input_shape = input_tensor.padded_shape();
        input_shape[this->in_dim] *= this->ring_size;
        input_shape[this->out_dim] /= this->ring_size;
        TT_FATAL(
            output_shape == input_shape,
            "Output tensor shape {} must match input tensor shape {} for AllToAllAsync",
            output_shape,
            input_shape);
    }

    TT_FATAL(this->num_links == 1, "AllToAllAsync: num_links must be 1, but is {}", this->num_links);
}

std::vector<ttnn::TensorSpec> AllToAllAsync::compute_output_specs(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors[0];
    auto shape = input_tensor.padded_shape();
    shape[this->in_dim] *= this->ring_size;
    shape[this->out_dim] /= this->ring_size;
    auto tensor_spec = TensorSpec(
        shape, TensorLayout(input_tensor.dtype(), input_tensor.tensor_spec().page_config(), output_mem_config));
    return {tensor_spec, tensor_spec};
}

tt::tt_metal::operation::MeshWorkloadWithCallbacks AllToAllAsync::create_mesh_workload(
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const std::vector<Tensor>& input_tensors,
    std::vector<Tensor>& output_tensors) const {
    return ccl::create_mesh_workload_from_programs(
        tensor_coords, input_tensors, output_tensors, [&, this](const ttnn::MeshCoordinate& coord) {
            return create_program_at(coord, input_tensors, output_tensors);
        });
}

tt::tt_metal::operation::ProgramWithCallbacks AllToAllAsync::create_program_at(
    const MeshCoordinate& coord, const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const {
    log_debug(tt::LogOp, "DEBUG: create_program_at is called");
    auto mesh_device = input_tensors[0].mesh_device();
    IDevice* target_device = mesh_device ? mesh_device->get_device(coord) : input_tensors[0].device();

    std::optional<IDevice*> forward_device = std::nullopt;
    std::optional<IDevice*> backward_device = std::nullopt;
    uint32_t device_index = this->ring_size;  // Initialize device index

    TT_FATAL(this->topology == ttnn::ccl::Topology::Ring, "DEBUG: topology: {}", this->topology);

    std::vector<IDevice*> devices_to_use = input_tensors[0].mesh_device()->get_view().get_ring_devices();

    for (uint32_t i = 0; i < this->ring_size; ++i) {
        if (devices_to_use.at(i) == target_device) {
            device_index = i;
            if (i != 0) {
                backward_device = devices_to_use.at(i - 1);
            } else if (topology == ttnn::ccl::Topology::Ring) {
                backward_device = devices_to_use.at(this->ring_size - 1);
            }
            if (i != this->ring_size - 1) {
                forward_device = devices_to_use.at(i + 1);
            } else if (topology == ttnn::ccl::Topology::Ring) {
                forward_device = devices_to_use.at(0);
            }
        }
    }

    TT_FATAL(device_index < this->ring_size, "DEBUG: device_index: {}", device_index);
    TT_FATAL(
        forward_device.value()->id() != backward_device.value()->id(),
        "DEBUG: forward and backward devices are the same: {}, {}",
        forward_device.value()->id(),
        backward_device.value()->id());
    TT_FATAL(
        forward_device.value()->id() != target_device->id(),
        "DEBUG: forward device is the same as target device: {}, {}",
        forward_device.value()->id(),
        target_device->id());
    TT_FATAL(
        backward_device.value()->id() != target_device->id(),
        "DEBUG: backward device is the same as target device: {}, {}",
        backward_device.value()->id(),
        target_device->id());

    return all_to_all_async_minimal(
        input_tensors[0],
        output_tensors.at(0),
        output_tensors.at(1),
        target_device,
        forward_device,
        backward_device,
        this->in_dim,
        this->out_dim,
        this->num_links,
        this->ring_size,
        device_index,
        this->topology,
        this->semaphore,
        this->sub_device_id);
}

tt::tt_metal::operation::Hash AllToAllAsync::compute_program_hash(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors[0];
    auto input_shape = input_tensor.padded_shape();
    auto input_memory_layout = input_tensor.layout();
    auto input_dtype = input_tensor.dtype();
    auto input_memory_config = input_tensor.memory_config();

    return tt::tt_metal::operation::hash_operation<AllToAllAsync>(
        this->in_dim,
        this->out_dim,
        this->num_links,
        this->ring_size,
        this->output_mem_config,
        this->topology,
        input_shape,
        input_memory_layout,
        input_dtype,
        input_memory_config);
}

namespace operations {
namespace experimental {
namespace ccl {

// Top-level API function for AllToAllAsync
Tensor all_to_all_async(
    const Tensor& input_tensor,
    Tensor& persistent_intermediate_buffer,
    Tensor& persistent_output_buffer,
    const int32_t in_dim,   // Changed from dim
    const int32_t out_dim,  // Added
    const GlobalSemaphore& multi_device_global_semaphore,
    const uint32_t num_links,
    const std::optional<MemoryConfig>& memory_config,
    const ttnn::ccl::Topology topology,
    std::optional<tt::tt_metal::SubDeviceId> sub_device_id) {
    TT_FATAL(
        std::getenv("TT_METAL_SLOW_DISPATCH_MODE") == nullptr,
        "all_to_all_async op is only supported for Fast Dispatch");

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

    std::vector<std::optional<Tensor>> optional_output_tensors = {
        persistent_intermediate_buffer, persistent_output_buffer};

    // Normalizing dims here before passing to the struct/op implementation
    int32_t rank = input_tensor.logical_shape().rank();
    int32_t norm_in_dim = (in_dim < 0) ? rank + in_dim : in_dim;
    int32_t norm_out_dim = (out_dim < 0) ? rank + out_dim : out_dim;

    TT_FATAL(norm_in_dim >= 0 && norm_in_dim < rank, "Invalid in_dim: {}", in_dim);
    TT_FATAL(norm_out_dim >= 0 && norm_out_dim < rank, "Invalid out_dim: {}", out_dim);

    return tt::tt_metal::operation::run(
               ttnn::AllToAllAsync(
                   devices,
                   norm_in_dim,
                   norm_out_dim,
                   num_links,
                   num_devices,
                   memory_config.value_or(input_tensor.memory_config()),
                   ccl_topology,
                   multi_device_global_semaphore,
                   sub_device_id),
               {input_tensor},
               {},
               optional_output_tensors)
        .at(1);
}

}  // namespace ccl
}  // namespace experimental
}  // namespace operations
}  // namespace ttnn
