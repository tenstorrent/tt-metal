// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "all_to_all_async_generic_op.hpp"
#include "ttnn/operations/functions.hpp"
#include "ttnn/operations/math.hpp"
#include "ttnn/global_semaphore.hpp"

#include "ttnn/tensor/tensor_utils.hpp"

namespace ttnn {

// Implementation of AllToAllAsyncGeneric methods

void AllToAllAsyncGeneric::validate_with_output_tensors(
    const std::vector<Tensor>& input_tensors, const std::vector<std::optional<Tensor>>& output_tensors) const {
    TT_FATAL(
        input_tensors.size() == 1,
        "AllToAllAsyncGeneric: Input tensor size must be 1, but is {}",
        input_tensors.size());
}

std::vector<ttnn::TensorSpec> AllToAllAsyncGeneric::compute_output_specs(
    const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors[0];
    auto shape = input_tensor.padded_shape();
    shape[this->in_dim] *= this->num_devices;
    shape[this->out_dim] /= this->num_devices;
    auto tensor_spec = TensorSpec(
        shape, TensorLayout(input_tensor.dtype(), input_tensor.tensor_spec().page_config(), output_mem_config));
    return {tensor_spec, tensor_spec};
}

tt::tt_metal::operation::MeshWorkloadWithCallbacks AllToAllAsyncGeneric::create_mesh_workload(
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const std::vector<Tensor>& input_tensors,
    std::vector<Tensor>& output_tensors) const {
    return ccl::create_mesh_workload_from_programs(
        tensor_coords, input_tensors, output_tensors, [&, this](const ttnn::MeshCoordinate& coord) {
            return create_program_at(coord, input_tensors, output_tensors);
        });
}

tt::tt_metal::operation::ProgramWithCallbacks AllToAllAsyncGeneric::create_program_at(
    const MeshCoordinate& coord, const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const {
    log_debug(tt::LogOp, "DEBUG: create_program_at is called");
    auto mesh_device = input_tensors[0].device();
    IDevice* target_device = mesh_device ? mesh_device->get_device(coord) : input_tensors[0].device();

    std::optional<IDevice*> forward_device = std::nullopt;
    std::optional<IDevice*> backward_device = std::nullopt;
    uint32_t device_index = this->num_devices;  // Initialize device index

    std::vector<IDevice*> devices_to_use = input_tensors[0].device()->get_view().get_line_devices();

    for (uint32_t i = 0; i < this->num_devices; ++i) {
        if (devices_to_use.at(i) == target_device) {
            device_index = i;
            if (i != 0) {
                backward_device = devices_to_use.at(i - 1);
            } else if (topology == ttnn::ccl::Topology::Ring) {
                backward_device = devices_to_use.at(this->num_devices - 1);
            }
            if (i != this->num_devices - 1) {
                forward_device = devices_to_use.at(i + 1);
            } else if (topology == ttnn::ccl::Topology::Ring) {
                forward_device = devices_to_use.at(0);
            }
        }
    }

    TT_FATAL(device_index < this->num_devices, "DEBUG: device_index: {}", device_index);
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

    return all_to_all_async_generic_program(
        input_tensors[0],
        output_tensors.at(0),
        output_tensors.at(1),
        target_device,
        forward_device,
        backward_device,
        this->in_dim,
        this->out_dim,
        this->num_links,
        this->num_devices,
        device_index,
        this->topology,
        this->semaphore,
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

namespace operations {
namespace experimental {
namespace ccl {

// Top-level API function for AllToAllAsyncGeneric
Tensor all_to_all_async_generic(
    const Tensor& input_tensor,
    Tensor& persistent_intermediate_buffer,
    Tensor& persistent_output_buffer,
    const int32_t in_dim,
    const int32_t out_dim,
    const GlobalSemaphore& multi_device_global_semaphore,
    const uint32_t num_links,
    const std::optional<MemoryConfig>& memory_config,
    const ttnn::ccl::Topology topology,
    std::optional<tt::tt_metal::SubDeviceId> sub_device_id) {
    std::vector<IDevice*> devices = ttnn::ccl::get_active_physical_devices(input_tensor);

    uint32_t num_devices = devices.size();
    TT_FATAL(
        num_devices > 1,
        "all_to_all_async is a collective operation and requires more than 1 device, but has {}",
        num_devices);

    ttnn::ccl::Topology ccl_topology = topology;

    std::vector<std::optional<Tensor>> optional_output_tensors = {
        persistent_intermediate_buffer, persistent_output_buffer};

    return tt::tt_metal::operation::run(
               ttnn::AllToAllAsyncGeneric(
                   devices,
                   in_dim,
                   out_dim,
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
