// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "all_gather_command_processor_async_op.hpp"
#include <tt-metalium/fabric.hpp>
#include "ttnn/operations/functions.hpp"
#include "ttnn/operations/math.hpp"
#include "ttnn/global_semaphore.hpp"

#include "ttnn/tensor/tensor_utils.hpp"

namespace ttnn {

void AllGatherCommandProcessorAsync::validate_with_output_tensors(
    const std::vector<Tensor>& input_tensors, const std::vector<std::optional<Tensor>>& output_tensors) const {
    TT_FATAL(input_tensors.size() == 1, "Error, Input tensor size should be 1 but has {}", input_tensors.size());
    const auto& input_tensor = input_tensors[0];
    const auto& layout = input_tensors[0].layout();
    const auto& dtype = input_tensors[0].dtype();
    const auto& page_size = input_tensors[0].buffer()->page_size();
    TT_FATAL(page_size % input_tensors[0].buffer()->alignment() == 0, "All Gather currently requires aligned pages");

    TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "Operands to all_gather need to be on device!");
    TT_FATAL(input_tensor.buffer() != nullptr, "Operands to all_gather need to be allocated in buffers on device!");
    TT_FATAL(this->num_links > 0, "Error, num_links should be more than 0 but has {}", this->num_links);
    TT_FATAL(
        this->num_links <= input_tensor.device()->compute_with_storage_grid_size().y,
        "Worker cores used by links are parallelizaed over rows");

    TT_FATAL(
        input_tensor.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED ||
            input_tensor.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED ||
            input_tensor.memory_config().memory_layout() == TensorMemoryLayout::BLOCK_SHARDED ||
            input_tensor.memory_config().memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED,
        "Unsupported input tensor memory layout {}.",
        input_tensor.memory_config().memory_layout());

    if (output_tensors.size() > 0 and output_tensors[0].has_value()) {
        TT_FATAL(
            output_tensors.size() <= 1,
            "Error, Number of output tensors should be at most 1 but has {}",
            output_tensors.size());
        const auto& output_tensor = output_tensors[0];

        TT_FATAL(
            output_tensor.value().storage_type() == StorageType::DEVICE,
            "Operands to all_gather need to be on device!");
        TT_FATAL(
            output_tensor.value().layout() == layout,
            "Error, Output tensor layout should be same as input tensor layout but has {}",
            output_tensor.value().layout());
        TT_FATAL(
            output_tensor.value().dtype() == dtype,
            "Error, Output tensor dtype should be same as input tensor dtype but has {}",
            output_tensor.value().dtype());
        TT_FATAL(
            output_tensor.value().tensor_spec().page_config() == input_tensor.tensor_spec().page_config(),
            "Error, Output tensor page config should be same as input tensor page config but has {}",
            output_tensor.value().tensor_spec().page_config());
        TT_FATAL(
            output_tensor.value().memory_config() == this->output_memory_config,
            "Error, Output tensor memory config should be same as output_memory_config but has {}",
            output_tensor.value().memory_config());

        TT_FATAL(
            output_tensor.value().memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED ||
                output_tensor.value().memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED ||
                output_tensor.value().memory_config().memory_layout() == TensorMemoryLayout::BLOCK_SHARDED ||
                output_tensor.value().memory_config().memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED,
            "Unsupported output tensor memory layout {}.",
            output_tensor.value().memory_config().memory_layout());

        // check the output tensor size
        auto output_shape = output_tensor.value().padded_shape();
        auto input_shape = input_tensor.padded_shape();
        TT_FATAL(
            output_shape.size() == input_shape.size(),
            "Error, Output tensor shape should have same number of dimensions as input tensor but has {}",
            output_shape.size());
        for (size_t i = 0; i < input_shape.size(); ++i) {
            if (i == this->dim) {
                TT_FATAL(
                    output_shape[i] <= input_shape[i] * this->ring_size,
                    "Error, Output tensor shape at dimension {} should be {} but has {}",
                    i,
                    input_shape[i] * this->ring_size,
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

        TT_FATAL(
            output_tensor.value().memory_config().memory_layout() == input_tensor.memory_config().memory_layout(),
            "Error, Output tensor memory layout should be same as input tensor memory layout but has {}",
            output_tensor.value().memory_config().memory_layout());
    }

    TT_FATAL(
        tt::tt_fabric::is_1d_fabric_config(tt::tt_fabric::GetFabricConfig()), "Only 1D fabric config is supported");
}

std::vector<ttnn::TensorSpec> AllGatherCommandProcessorAsync::compute_output_specs(
    const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors[0];
    auto shape = input_tensor.logical_shape();
    shape[this->dim] *= this->ring_size;
    return {TensorSpec(
        shape, TensorLayout(input_tensor.dtype(), input_tensor.tensor_spec().page_config(), output_memory_config))};
}

std::vector<Tensor> AllGatherCommandProcessorAsync::create_output_tensors(
    const std::vector<Tensor>& input_tensors, const std::vector<std::optional<Tensor>>& optional_output_tensors) const {
    return tt::tt_metal::operation::default_create_output_tensors(*this, input_tensors, optional_output_tensors);
}

tt::tt_metal::operation::MeshWorkloadWithCallbacks AllGatherCommandProcessorAsync::create_mesh_workload(
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const std::vector<Tensor>& input_tensors,
    std::vector<Tensor>& output_tensors) const {
    return ccl::create_mesh_workload_from_programs(
        tensor_coords, input_tensors, output_tensors, [&, this](const ttnn::MeshCoordinate& coord) {
            return create_program_at(coord, input_tensors, output_tensors);
        });
}

tt::tt_metal::operation::ProgramWithCallbacks AllGatherCommandProcessorAsync::create_program_at(
    const MeshCoordinate& coord, const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const {
    log_debug(tt::LogOp, "DEBUG: create_program_at is called");
    auto mesh_device = input_tensors[0].device();
    IDevice* target_device = mesh_device ? mesh_device->get_device(coord) : input_tensors[0].device();
    std::vector<IDevice*> devices_to_use = {};
    if (this->cluster_axis.has_value()) {
        const auto& mesh_view = input_tensors[0].device()->get_view();
        // User specified the cluster-axis. Derive devices based on the current coordinate
        // and the cluster-axis.
        devices_to_use = (this->cluster_axis.value() == 0) ? mesh_view.get_devices_on_column(coord[1])
                                                           : mesh_view.get_devices_on_row(coord[0]);
    } else {
        devices_to_use = devices;
    }
    uint32_t target_ring_size = devices_to_use.size();

    std::optional<IDevice*> forward_device = std::nullopt;
    std::optional<IDevice*> backward_device = std::nullopt;
    uint32_t device_index = 0;  // Initialize device index
    for (uint32_t i = 0; i < target_ring_size; ++i) {
        if (devices_to_use.at(i) == target_device) {
            device_index = i;
            if (i != 0) {
                backward_device = devices_to_use.at(i - 1);
            } else if (topology == ttnn::ccl::Topology::Ring) {
                backward_device = devices_to_use.at(target_ring_size - 1);
            }
            if (i != target_ring_size - 1) {
                forward_device = devices_to_use.at(i + 1);
            } else if (topology == ttnn::ccl::Topology::Ring) {
                forward_device = devices_to_use.at(0);
            }
        }
    }

    return all_gather_command_processor_async_multi_core_with_workers(
        input_tensors[0],
        target_device,
        forward_device,
        backward_device,
        output_tensors[0],
        target_ring_size,
        device_index,
        this->dim,
        this->semaphore,
        this->num_links,
        this->topology,
        this->sub_device_id);
}

tt::tt_metal::operation::Hash AllGatherCommandProcessorAsync::compute_program_hash(
    const std::vector<Tensor>& input_tensors) const {
    log_trace(tt::LogOp, "compute_program_hash is called");
    auto input_shape = input_tensors[0].padded_shape();
    auto input_memory_layout = input_tensors[0].layout();
    auto input_dtype = input_tensors[0].dtype();
    auto input_memory_config = input_tensors[0].memory_config();

    return tt::tt_metal::operation::hash_operation<AllGatherCommandProcessorAsync>(
        this->ring_size,
        this->dim,
        this->num_links,
        this->output_memory_config,
        this->topology,
        this->cluster_axis,
        this->sub_device_id.has_value(),
        this->sub_device_id.has_value()
            ? input_tensors[0].device()->worker_cores(
                  tt::tt_metal::HalProgrammableCoreType::TENSIX, this->sub_device_id.value())
            : CoreRangeSet(CoreRange({0, 0}, {0, 0})),
        input_shape,
        input_memory_layout,
        input_dtype,
        input_memory_config);
}

namespace operations {
namespace experimental {
namespace ccl {

namespace {
Tensor all_gather_command_processor_async_impl(
    const ttnn::Tensor& input_tensor,
    int32_t dim,
    const GlobalSemaphore& multi_device_global_semaphore,
    const std::optional<ttnn::Tensor>& persistent_output_buffer,
    uint32_t num_links,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    ttnn::ccl::Topology topology,
    std::optional<uint32_t> cluster_axis,
    std::optional<tt::tt_metal::SubDeviceId> sub_device_id,
    const std::vector<IDevice*>& devices) {
    TT_FATAL(
        std::getenv("TT_METAL_SLOW_DISPATCH_MODE") == nullptr,
        "all_gather_command_processor_async op is only supported for Fast Dispatch");

    uint32_t num_devices;
    if (cluster_axis.has_value()) {
        auto mesh_device = input_tensor.device();
        TT_FATAL(mesh_device != nullptr, "Mesh device is required when cluster_axis is set");
        const auto& mesh_view = mesh_device->get_view();
        // Use the mesh dimensions to determine the ring size
        num_devices = (cluster_axis.value() == 0) ? mesh_view.num_rows() : mesh_view.num_cols();
    } else {
        num_devices = devices.size();
    }

    ttnn::ccl::Topology ccl_topology = topology;
    if (num_devices == 2) {
        ccl_topology = ttnn::ccl::Topology::Linear;
    }

    std::vector<std::optional<Tensor>> optional_output_tensors = {persistent_output_buffer};
    return tt::tt_metal::operation::run(
               ttnn::AllGatherCommandProcessorAsync(
                   devices,
                   num_devices,
                   dim,
                   multi_device_global_semaphore,
                   num_links,
                   memory_config.value_or(input_tensor.memory_config()),
                   ccl_topology,
                   cluster_axis,
                   sub_device_id),
               {input_tensor},
               {},
               optional_output_tensors)
        .at(0);
}
}  // namespace

Tensor all_gather_command_processor_async(
    const ttnn::Tensor& input_tensor,
    int32_t dim,
    const GlobalSemaphore& multi_device_global_semaphore,
    const std::optional<ttnn::Tensor>& persistent_output_buffer,
    uint32_t num_links,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    ttnn::ccl::Topology topology,
    std::optional<uint32_t> cluster_axis,
    std::optional<tt::tt_metal::SubDeviceId> sub_device_id) {
    return all_gather_command_processor_async_impl(
        input_tensor,
        dim,
        multi_device_global_semaphore,
        persistent_output_buffer,
        num_links,
        memory_config,
        topology,
        cluster_axis,
        sub_device_id,
        ttnn::ccl::get_active_physical_devices(input_tensor));
}

std::vector<Tensor> all_gather_command_processor_async(
    const std::vector<Tensor>& input_tensors,
    int32_t dim,
    const std::vector<global_semaphore::MultiDeviceGlobalSemaphore>& multi_device_global_semaphore,
    const std::optional<ttnn::Tensor>& persistent_output_buffer,
    uint32_t num_links,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    ttnn::ccl::Topology topology,
    std::optional<uint32_t> cluster_axis,
    std::optional<tt::tt_metal::SubDeviceId> sub_device_id) {
    std::vector<Tensor> output_tensors;
    output_tensors.reserve(input_tensors.size());
    std::vector<GlobalSemaphore> semaphore;
    semaphore.reserve(multi_device_global_semaphore.size());
    for (size_t i = 0; i < multi_device_global_semaphore.size(); i++) {
        semaphore.push_back(multi_device_global_semaphore.at(i).global_semaphores.at(i));
    }
    for (size_t i = 0; i < input_tensors.size(); i++) {
        output_tensors.push_back(all_gather_command_processor_async_impl(
            input_tensors[i],
            dim,
            semaphore[0],
            persistent_output_buffer,
            num_links,
            memory_config,
            topology,
            cluster_axis,
            sub_device_id,
            ttnn::ccl::get_active_physical_devices(input_tensors)));
    }
    return output_tensors;
}

}  // namespace ccl
}  // namespace experimental
}  // namespace operations

}  // namespace ttnn
