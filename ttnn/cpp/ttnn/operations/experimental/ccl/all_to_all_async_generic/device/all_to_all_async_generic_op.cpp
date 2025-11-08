// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "all_to_all_async_generic_op.hpp"
#include "ttnn/operations/functions.hpp"
#include "ttnn/operations/math.hpp"
#include "ttnn/global_semaphore.hpp"

#include "tt-metalium/tensor/tensor_utils.hpp"

namespace ttnn {

// Implementation of AllToAllAsyncGeneric methods

void AllToAllAsyncGeneric::validate_with_output_tensors(
    const std::vector<Tensor>& input_tensors, const std::vector<std::optional<Tensor>>& output_tensors) const {
    TT_FATAL(
        input_tensors.size() == 1,
        "AllToAllAsyncGeneric: Input tensor size must be 1, but is {}",
        input_tensors.size());
    const auto& input_tensor = input_tensors[0];
    const auto& page_size = input_tensor.buffer()->page_size();
    const auto& input_shape = input_tensor.logical_shape();
    auto rank = input_shape.rank();

    TT_FATAL(this->in_dim >= 0 && this->in_dim < rank, "in_dim out of range");
    TT_FATAL(this->out_dim >= 0 && this->out_dim < rank, "out_dim out of range");

    TT_FATAL(page_size % input_tensor.buffer()->alignment() == 0, "AllToAllAsync currently requires aligned pages");

    TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "Operands to all_to_all_async must be on device");
    TT_FATAL(input_tensor.buffer() != nullptr, "Operands to all_to_all_async must be allocated in buffers on device");
    TT_FATAL(this->num_links == 1, "num_links must be 1, but is {}", this->num_links);

    TT_FATAL(
        input_shape[this->out_dim] % this->num_devices == 0,
        "AllToAllAsync: input tensor dimension {} must be divisible by num_devices {}",
        input_shape[this->out_dim],
        this->num_devices);
    TT_FATAL(
        input_tensor.buffer()->buffer_type() == BufferType::DRAM,
        "AllToAllAsync: Input tensor must be in DRAM, but is in {}",
        input_tensor.buffer()->buffer_type());
    TT_FATAL(input_tensor.layout() == Layout::TILE, "Unsupported input layout {}.", input_tensor.layout());

    const auto& maybe_output_tensor = output_tensors[0];
    if (maybe_output_tensor.has_value()) {
        const auto& output_tensor = maybe_output_tensor.value();

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
            output_tensor.memory_config() == this->output_mem_config,
            "Output tensor memory config must match specified output_mem_config");

        const auto& output_shape = output_tensor.logical_shape();
        auto expected_output_shape = input_tensor.logical_shape();
        expected_output_shape[this->in_dim] *= this->num_devices;
        expected_output_shape[this->out_dim] /= this->num_devices;
        TT_FATAL(
            output_shape == expected_output_shape,
            "Output tensor shape {} must match expected output tensor shape {} for AllToAllAsync",
            output_shape,
            expected_output_shape);
    }

    // recreate output shape to cover optional output buffer
    auto output_shape = input_tensor.logical_shape();
    output_shape[this->in_dim] *= this->num_devices;
    output_shape[this->out_dim] /= this->num_devices;

    // Check padding support, currently supported only on height
    auto last_dim = rank - 1;
    auto second_last_dim = rank - 2;
    TT_FATAL(
        in_dim != second_last_dim || input_shape[in_dim] % 16 == 0,
        "{} dimension support only 0 or 16 padding, so must be divisible by 16. Input tensor shape {} , but has {} "
        "padding",
        in_dim,
        input_shape,
        input_shape[in_dim] % 32);
    TT_FATAL(
        out_dim != second_last_dim || output_shape[out_dim] % 16 == 0,
        "{} dimension support only 0 or 16 padding, so must be divisible by 16. Output tensor shape {} , but has {} "
        "padding",
        out_dim,
        output_shape,
        output_shape[out_dim] % 32);
    TT_FATAL(
        in_dim != last_dim || input_shape[in_dim] % 32 == 0,
        "{} dimension doesnt support padding, so must be divisible by 32. Input tensor shape {} , but has {} padding",
        in_dim,
        input_shape,
        input_shape[in_dim] % 32);
    TT_FATAL(
        out_dim != last_dim || output_shape[out_dim] % 32 == 0,
        "{} dimension doesnt support padding, so must be divisible by 32. Output tensor shape {} , but has {} padding",
        out_dim,
        output_shape,
        output_shape[out_dim] % 32);
}

std::vector<ttnn::TensorSpec> AllToAllAsyncGeneric::compute_output_specs(
    const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors[0];
    auto shape = input_tensor.logical_shape();
    shape[this->in_dim] *= this->num_devices;
    shape[this->out_dim] /= this->num_devices;
    auto tensor_spec = TensorSpec(
        shape, TensorLayout(input_tensor.dtype(), input_tensor.tensor_spec().page_config(), output_mem_config));
    return {tensor_spec};
}

tt::tt_metal::operation::MeshWorkloadWithCallbacks AllToAllAsyncGeneric::create_mesh_workload(
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const std::vector<Tensor>& input_tensors,
    std::vector<Tensor>& output_tensors) const {
    auto mesh_device = input_tensors[0].device();
    auto sub_device_id = this->sub_device_id;

    auto subdevice = sub_device_id.has_value() ? *sub_device_id : mesh_device->get_sub_device_ids().at(0);
    const auto available_cores = mesh_device->worker_cores(tt::tt_metal::HalProgrammableCoreType::TENSIX, subdevice);
    auto subdevices = {subdevice};

    auto init_barrier_semaphore = ttnn::global_semaphore::create_global_semaphore(mesh_device, available_cores, 0);
    auto final_barrier_semaphore = ttnn::global_semaphore::create_global_semaphore(mesh_device, available_cores, 0);
    tt::tt_metal::distributed::Synchronize(mesh_device, std::nullopt, subdevices);

    return ccl::create_mesh_workload_from_programs(
        tensor_coords, input_tensors, output_tensors, [&, this](const ttnn::MeshCoordinate& coord) {
            return create_program_at(
                coord, input_tensors, output_tensors, init_barrier_semaphore, final_barrier_semaphore);
        });
}

tt::tt_metal::operation::ProgramWithCallbacks AllToAllAsyncGeneric::create_program_at(
    const MeshCoordinate& coord,
    const std::vector<Tensor>& input_tensors,
    std::vector<Tensor>& output_tensors,
    const GlobalSemaphore& init_barrier_semaphore,
    const GlobalSemaphore& final_barrier_semaphore) const {
    log_debug(tt::LogOp, "DEBUG: create_program_at is called");

    uint32_t device_index = ttnn::ccl::get_linearized_index_from_physical_coord(input_tensors[0], coord, cluster_axis);

    const std::optional<MeshCoordinate> forward_coord =
        ttnn::ccl::get_physical_neighbor_from_physical_coord(input_tensors[0], coord, 1, topology, cluster_axis);
    const std::optional<MeshCoordinate> backward_coord =
        ttnn::ccl::get_physical_neighbor_from_physical_coord(input_tensors[0], coord, -1, topology, cluster_axis);

    TT_FATAL(device_index < this->num_devices, "DEBUG: device_index: {}", device_index);

    return all_to_all_async_generic_program(
        input_tensors[0],
        output_tensors.at(0),
        coord,
        forward_coord,
        backward_coord,
        this->in_dim,
        this->out_dim,
        this->num_links,
        this->num_devices,
        device_index,
        this->topology,
        init_barrier_semaphore,
        final_barrier_semaphore,
        this->sub_device_id);
}

tt::tt_metal::operation::Hash AllToAllAsyncGeneric::compute_program_hash(
    const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors[0];
    const auto& input_shape = input_tensor.padded_shape();
    auto input_memory_layout = input_tensor.layout();
    auto input_dtype = input_tensor.dtype();
    const auto& input_memory_config = input_tensor.memory_config();

    return tt::tt_metal::operation::hash_operation<AllToAllAsyncGeneric>(
        this->in_dim,
        this->out_dim,
        this->num_links,
        this->num_devices,
        this->output_mem_config,
        this->topology,
        input_shape,
        input_memory_layout,
        input_dtype,
        input_memory_config);
}

std::vector<Tensor> AllToAllAsyncGeneric::create_output_tensors(
    const std::vector<Tensor>& input_tensors, const std::vector<std::optional<Tensor>>& optional_output_tensors) const {
    auto tensor_specs = compute_output_specs(input_tensors);

    ttnn::Tensor output_buffer = optional_output_tensors.at(0).has_value()
                                     ? optional_output_tensors.at(0).value()
                                     : create_device_tensor(tensor_specs[0], input_tensors.at(0).device());

    return {output_buffer};
}

namespace operations {
namespace experimental {
namespace ccl {

// Top-level API function for AllToAllAsyncGeneric
Tensor all_to_all_async_generic(
    const Tensor& input_tensor,
    const std::optional<Tensor>& persistent_output_buffer,
    const int32_t in_dim,
    const int32_t out_dim,
    const uint32_t num_links,
    const std::optional<MemoryConfig>& memory_config,
    const ttnn::ccl::Topology topology,
    std::optional<tt::tt_metal::SubDeviceId> sub_device_id,
    std::optional<uint32_t> cluster_axis) {
    uint32_t num_devices = ttnn::ccl::get_topological_dimension(input_tensor, cluster_axis);
    TT_FATAL(
        num_devices > 1,
        "all_to_all_async is a collective operation and requires more than 1 device, but has {}",
        num_devices);

    std::vector<std::optional<Tensor>> optional_output_tensors = {persistent_output_buffer};

    return tt::tt_metal::operation::run(
               ttnn::AllToAllAsyncGeneric(
                   in_dim,
                   out_dim,
                   num_links,
                   num_devices,
                   memory_config.value_or(input_tensor.memory_config()),
                   topology,
                   sub_device_id,
                   cluster_axis),
               {input_tensor},
               {},
               optional_output_tensors)
        .at(0);
}

}  // namespace ccl
}  // namespace experimental
}  // namespace operations
}  // namespace ttnn
