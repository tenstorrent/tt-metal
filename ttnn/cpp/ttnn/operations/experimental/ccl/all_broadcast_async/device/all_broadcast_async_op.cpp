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

std::optional<MeshCoordinate> get_tensor_topology_neighbor(
    const tt::tt_metal::distributed::MeshShape& shape,
    const MeshCoordinate& coord,
    int offset,
    ttnn::ccl::Topology topology,
    const std::optional<uint32_t>& cluster_axis) {
    auto boundary_mode = topology == ttnn::ccl::Topology::Ring
                             ? tt::tt_metal::distributed::MeshCoordinate::BoundaryMode::WRAP
                             : tt::tt_metal::distributed::MeshCoordinate::BoundaryMode::NONE;
    if (cluster_axis.has_value()) {
        TT_FATAL(cluster_axis.value() == 0 || cluster_axis.value() == 1, "Cluster axis must be 0 or 1");
        return coord.get_neighbor(shape, offset, cluster_axis.value(), boundary_mode);
    } else {
        for (int i = shape.dims() - 1; i >= 0; i--) {
            if (shape[i] > 1) {
                return coord.get_neighbor(shape, offset, i, boundary_mode);
            }
        }
        return std::nullopt;
    }
}

uint32_t get_tensor_topology_linearized_index(
    const tt::tt_metal::distributed::MeshShape& shape,
    const MeshCoordinate& coord,
    const std::optional<uint32_t>& cluster_axis) {
    if (cluster_axis.has_value()) {
        return coord[cluster_axis.value()];
    } else {
        return coord.to_linear_index(shape);
    }
}

uint32_t get_tensor_topology_dimension(
    const tt::tt_metal::distributed::MeshShape& shape, const std::optional<uint32_t>& cluster_axis) {
    if (cluster_axis.has_value()) {
        return shape[cluster_axis.value()];
    } else {
        return shape.mesh_size();
    }
}

tt::tt_metal::operation::ProgramWithCallbacks AllBroadcastAsync::create_program_at(
    const MeshCoordinate& coord,
    const std::vector<Tensor>& input_tensors,
    std::vector<Tensor>& output_tensors,
    const GlobalSemaphore& init_barrier_semaphore,
    const GlobalSemaphore& final_barrier_semaphore) const {
    log_debug(tt::LogOp, "DEBUG: create_program_at is called");
    auto target_device_coord = coord;

    auto tensor_topology_shape = input_tensors[0].tensor_topology().distribution_shape();
    uint32_t target_ring_size = get_tensor_topology_dimension(tensor_topology_shape, this->cluster_axis);
    std::optional<MeshCoordinate> backward_coord =
        get_tensor_topology_neighbor(tensor_topology_shape, coord, -1, this->topology, this->cluster_axis);
    std::optional<MeshCoordinate> forward_coord =
        get_tensor_topology_neighbor(tensor_topology_shape, coord, 1, this->topology, this->cluster_axis);
    uint32_t device_index = get_tensor_topology_linearized_index(tensor_topology_shape, coord, this->cluster_axis);

    return all_broadcast_async_multicore(
        input_tensors[0],
        target_device_coord,
        forward_coord,
        backward_coord,
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
    auto tensor_topology_shape = input_tensor.tensor_topology().distribution_shape();
    TT_FATAL(
        std::getenv("TT_METAL_SLOW_DISPATCH_MODE") == nullptr,
        "all_broadcast_async op is only supported for Fast Dispatch");

    if (!cluster_axis.has_value()) {
        bool all_other_dims_are_one = true;
        for (int i = 0; i < tensor_topology_shape.dims(); i++) {
            if (tensor_topology_shape[i] > 1 && !all_other_dims_are_one) {
                TT_THROW(
                    "all_broadcast_async op is only supported for cluster_axis=None when all but one mesh dimensions "
                    "are 1");
            }
            if (tensor_topology_shape[i] > 1) {
                all_other_dims_are_one = false;
            }
        }
    }

    uint32_t num_devices = get_tensor_topology_dimension(tensor_topology_shape, cluster_axis);

    TT_FATAL(num_devices > 1, "all_broadcast_async op will only work for num_devices > 1, but has {}", num_devices);

    ttnn::ccl::Topology ccl_topology = topology;
    if (num_devices == 2) {
        ccl_topology = ttnn::ccl::Topology::Linear;
    }
    log_debug(tt::LogOp, "DEBUG: creating line_fabric with num devices: {}, num links: {}", devices.size(), num_links);
    log_debug(tt::LogOp, "DEBUG: line_fabric is created");

    return tt::tt_metal::operation::run(
        ttnn::AllBroadcastAsync(
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
