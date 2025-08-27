// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "all_broadcast_async_op.hpp"
#include "ttnn/operations/functions.hpp"
#include "ttnn/operations/math.hpp"
#include "ttnn/global_semaphore.hpp"

#include "ttnn/tensor/tensor_utils.hpp"

namespace ttnn {

void AllBroadcastAsync::validate_with_output_tensors(
    const std::vector<Tensor>& input_tensors, const std::vector<std::optional<Tensor>>& output_tensors) const {
    TT_FATAL(input_tensors.size() == 1, "Error, Input tensor size should be 1 but has {}", input_tensors.size());
    const auto& input_tensor = input_tensors[0];

    TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "Operands to all_broadcast need to be on device!");
    TT_FATAL(input_tensor.buffer() != nullptr, "Operands to all_broadcast need to be allocated in buffers on device!");
    TT_FATAL(this->num_links > 0, "Error, num_links should be more than 0 but has {}", this->num_links);
    TT_FATAL(
        this->num_links <= input_tensor.device()->compute_with_storage_grid_size().y,
        "Worker cores used by links are parallelizaed over rows");

    TT_FATAL(
        input_tensor.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED ||
            input_tensor.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED ||
            input_tensor.memory_config().memory_layout() == TensorMemoryLayout::BLOCK_SHARDED ||
            input_tensor.memory_config().memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED,
        "Unsupported memory layout {}.",
        input_tensor.memory_config().memory_layout());
}

std::vector<ttnn::TensorSpec> AllBroadcastAsync::compute_output_specs(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors[0];
    const auto& shape = input_tensor.logical_shape();
    std::vector<TensorSpec> output_specs;
    output_specs.reserve(this->ring_size);
    for (uint32_t i = 0; i < this->ring_size; ++i) {
        output_specs.push_back(TensorSpec(
            shape, TensorLayout(input_tensor.dtype(), input_tensor.tensor_spec().page_config(), output_mem_config)));
    }
    return output_specs;
}

tt::tt_metal::operation::MeshWorkloadWithCallbacks AllBroadcastAsync::create_mesh_workload(
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
    log_debug(tt::LogOp, "Semaphores allocated and waiting for all devices to be ready");
    tt::tt_metal::distributed::Synchronize(mesh_device, std::nullopt, subdevices);
    log_debug(tt::LogOp, "All devices are ready, starting program execution");

    return ccl::create_mesh_workload_from_programs(
        tensor_coords, input_tensors, output_tensors, [&, this](const ttnn::MeshCoordinate& coord) {
            return create_program_at(
                coord, input_tensors, output_tensors, init_barrier_semaphore, final_barrier_semaphore);
        });
}

tt::tt_metal::operation::ProgramWithCallbacks AllBroadcastAsync::create_program_at(
    const MeshCoordinate& coord,
    const std::vector<Tensor>& input_tensors,
    std::vector<Tensor>& output_tensors,
    const GlobalSemaphore& init_barrier_semaphore,
    const GlobalSemaphore& final_barrier_semaphore) const {
    log_debug(tt::LogOp, "DEBUG: create_program_at is called");
    auto mesh_device = input_tensors[0].device();
    IDevice* target_device = mesh_device ? mesh_device->get_device(coord) : input_tensors[0].device();
    std::vector<IDevice*> devices_to_use = {};
    if (this->cluster_axis.has_value()) {
        // User specified the cluster-axis. Derive devices based on the current coordinate
        // and the cluster-axis.
        const auto& mesh_view = input_tensors[0].device()->get_view();
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

    return all_broadcast_async_multicore(
        input_tensors[0],
        target_device,
        forward_device,
        backward_device,
        output_tensors,
        this->num_links,
        target_ring_size,
        device_index,
        this->topology,
        final_barrier_semaphore,
        init_barrier_semaphore,
        this->sub_device_id);
}

tt::tt_metal::operation::Hash AllBroadcastAsync::compute_program_hash(const std::vector<Tensor>& input_tensors) const {
    log_trace(tt::LogOp, "compute_program_hash is called");
    auto input_shape = input_tensors[0].padded_shape();
    auto input_memory_layout = input_tensors[0].layout();
    auto input_dtype = input_tensors[0].dtype();
    auto input_memory_config = input_tensors[0].memory_config();
    return tt::tt_metal::operation::hash_operation<AllBroadcastAsync>(
        this->num_links,
        this->ring_size,
        this->output_mem_config,
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

namespace operations::experimental::ccl {

std::vector<Tensor> all_broadcast_async_impl(
    const Tensor& input_tensor,
    const uint32_t num_links,
    const std::optional<MemoryConfig>& memory_config,
    const ttnn::ccl::Topology topology,
    std::optional<uint32_t> cluster_axis,
    std::optional<tt::tt_metal::SubDeviceId> sub_device_id,
    const std::vector<IDevice*>& devices) {
    TT_FATAL(
        std::getenv("TT_METAL_SLOW_DISPATCH_MODE") == nullptr,
        "all_broadcast_async op is only supported for Fast Dispatch");

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

    TT_FATAL(num_devices > 1, "all_broadcast_async op will only work for num_devices > 1, but has {}", num_devices);

    ttnn::ccl::Topology ccl_topology = topology;
    if (num_devices == 2) {
        ccl_topology = ttnn::ccl::Topology::Linear;
    }
    log_debug(tt::LogOp, "DEBUG: creating line_fabric with num devices: {}, num links: {}", devices.size(), num_links);
    log_debug(tt::LogOp, "DEBUG: line_fabric is created");

    return tt::tt_metal::operation::run(
        ttnn::AllBroadcastAsync(
            devices,
            num_links,
            num_devices,
            memory_config.value_or(input_tensor.memory_config()),
            ccl_topology,
            sub_device_id,
            cluster_axis),
        {input_tensor});
}

std::vector<Tensor> all_broadcast_async(
    const Tensor& input_tensor,
    const uint32_t num_links,
    const std::optional<MemoryConfig>& memory_config,
    const ttnn::ccl::Topology topology,
    std::optional<uint32_t> cluster_axis,
    std::optional<tt::tt_metal::SubDeviceId> sub_device_id) {
    return all_broadcast_async_impl(
        input_tensor,
        num_links,
        memory_config,
        topology,
        cluster_axis,
        sub_device_id,
        ttnn::ccl::get_active_physical_devices(input_tensor));
}

}  // namespace operations::experimental::ccl

}  // namespace ttnn
