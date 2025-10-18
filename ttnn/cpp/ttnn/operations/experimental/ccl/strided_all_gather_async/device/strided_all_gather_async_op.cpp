// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "strided_all_gather_async_op.hpp"
#include <tt-metalium/fabric.hpp>
#include "ttnn/operations/functions.hpp"
#include "ttnn/operations/math.hpp"
#include "ttnn/global_semaphore.hpp"

#include "ttnn/tensor/tensor_utils.hpp"

namespace ttnn {

void StridedAllGatherAsync::validate_with_output_tensors(
    const std::vector<Tensor>& input_tensors, const std::vector<std::optional<Tensor>>& output_tensors) const {}

std::vector<ttnn::TensorSpec> StridedAllGatherAsync::compute_output_specs(
    const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors[0];
    auto shape = input_tensor.logical_shape();  // TODO: Replace with logical_shape()
    shape[this->dim] *= this->ring_size;
    return {TensorSpec(
        shape, TensorLayout(input_tensor.dtype(), input_tensor.tensor_spec().page_config(), output_mem_config))};
}

std::vector<Tensor> StridedAllGatherAsync::create_output_tensors(
    const std::vector<Tensor>& input_tensors, const std::vector<std::optional<Tensor>>& optional_output_tensors) const {
    return tt::tt_metal::operation::default_create_output_tensors(*this, input_tensors, optional_output_tensors);
}

tt::tt_metal::operation::MeshWorkloadWithCallbacks StridedAllGatherAsync::create_mesh_workload(
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const std::vector<Tensor>& input_tensors,
    std::vector<Tensor>& output_tensors) const {
    return ccl::create_mesh_workload_from_programs(
        tensor_coords, input_tensors, output_tensors, [&, this](const ttnn::MeshCoordinate& coord) {
            return create_program_at(coord, input_tensors, output_tensors);
        });
}

tt::tt_metal::operation::ProgramWithCallbacks StridedAllGatherAsync::create_program_at(
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

    log_trace(tt::LogOp, "Detected all gather specialized shape. strided_all_gather_async_minimal_default is called");
    return strided_all_gather_async_minimal_default(
        input_tensors[0],
        target_device,
        forward_device,
        backward_device,
        output_tensors[0],
        this->dim,
        this->num_links,
        target_ring_size,
        device_index,
        this->topology,
        this->semaphore,
        this->barrier_semaphore,
        this->using_persistent_buffers,
        this->sub_device_id,
        this->chunks_per_sync,
        this->num_workers_per_link,
        this->num_buffers_per_channel);
}

tt::tt_metal::operation::Hash StridedAllGatherAsync::compute_program_hash(
    const std::vector<Tensor>& input_tensors) const {
    log_trace(tt::LogOp, "compute_program_hash is called");
    auto input_shape = input_tensors[0].padded_shape();
    auto input_memory_layout = input_tensors[0].layout();
    auto input_dtype = input_tensors[0].dtype();
    auto input_memory_config = input_tensors[0].memory_config();
    return tt::tt_metal::operation::hash_operation<StridedAllGatherAsync>(
        this->dim,
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
        this->barrier_semaphore.has_value(),
        this->using_persistent_buffers,
        this->chunks_per_sync,
        this->num_workers_per_link,
        this->num_buffers_per_channel,
        input_shape,
        input_memory_layout,
        input_dtype,
        input_memory_config);
}

namespace operations {
namespace experimental {
namespace ccl {

namespace {
Tensor strided_all_gather_async_impl(
    const Tensor& input_tensor,
    const std::optional<ttnn::Tensor>& persistent_output_buffer,
    const uint32_t dim,
    const std::vector<GlobalSemaphore>& multi_device_global_semaphore,
    const uint32_t num_links,
    const std::optional<MemoryConfig>& memory_config,
    const ttnn::ccl::Topology topology,
    std::optional<tt::tt_metal::SubDeviceId> sub_device_id,
    const std::vector<IDevice*>& devices,
    const std::optional<uint32_t>& cluster_axis,
    const std::optional<GlobalSemaphore>& barrier_semaphore,
    const std::optional<uint32_t>& chunks_per_sync,
    const std::optional<uint32_t>& num_workers_per_link,
    const std::optional<uint32_t>& num_buffers_per_channel) {
    TT_FATAL(
        std::getenv("TT_METAL_SLOW_DISPATCH_MODE") == nullptr,
        "strided_all_gather_async op is only supported for Fast Dispatch");

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

    TT_FATAL(
        num_devices > 1, "strided_all_gather_async op will only work for num_devices > 1, but has {}", num_devices);
    ttnn::ccl::Topology ccl_topology = topology;

    if (num_devices == 2) {
        ccl_topology = ttnn::ccl::Topology::Linear;
    }
    log_debug(tt::LogOp, "DEBUG: creating line_fabric with num devices: {}, num links: {}", devices.size(), num_links);
    log_debug(tt::LogOp, "DEBUG: line_fabric is created");

    bool using_persistent_buffers = persistent_output_buffer.has_value();

    std::vector<std::optional<Tensor>> optional_output_tensors = {persistent_output_buffer};

    return tt::tt_metal::operation::run(
               ttnn::StridedAllGatherAsync(
                   devices,
                   dim,
                   num_links,
                   num_devices,
                   memory_config.value_or(input_tensor.memory_config()),
                   ccl_topology,
                   multi_device_global_semaphore,
                   sub_device_id,
                   cluster_axis,
                   barrier_semaphore,
                   using_persistent_buffers,
                   chunks_per_sync,
                   num_workers_per_link,
                   num_buffers_per_channel),
               {input_tensor},
               {},
               optional_output_tensors)
        .at(0);
}
}  // namespace

Tensor strided_all_gather_async(
    const Tensor& input_tensor,
    const std::optional<ttnn::Tensor>& persistent_output_buffer,
    const uint32_t dim,
    const std::vector<GlobalSemaphore>& multi_device_global_semaphore,
    const uint32_t num_links,
    const std::optional<MemoryConfig>& memory_config,
    const ttnn::ccl::Topology topology,
    std::optional<tt::tt_metal::SubDeviceId> sub_device_id,
    std::optional<uint32_t> cluster_axis,
    const std::optional<GlobalSemaphore>& barrier_semaphore,
    std::optional<uint32_t> chunks_per_sync,
    std::optional<uint32_t> num_workers_per_link,
    std::optional<uint32_t> num_buffers_per_channel) {
    return strided_all_gather_async_impl(
        input_tensor,
        persistent_output_buffer,
        dim,
        multi_device_global_semaphore,
        num_links,
        memory_config,
        topology,
        sub_device_id,
        ttnn::ccl::get_active_physical_devices(input_tensor),
        cluster_axis,
        barrier_semaphore,
        chunks_per_sync,
        num_workers_per_link,
        num_buffers_per_channel);
}

}  // namespace ccl
}  // namespace experimental
}  // namespace operations

}  // namespace ttnn
