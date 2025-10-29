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
    uint32_t device_index = ccl::get_linearized_index_from_physical_coord(input_tensors[0], coord, this->cluster_axis);

    std::optional<MeshCoordinate> forward_coord =
        ccl::get_physical_neighbor_from_physical_coord(input_tensors[0], coord, 1, this->topology, this->cluster_axis);

    std::optional<MeshCoordinate> backward_coord =
        ccl::get_physical_neighbor_from_physical_coord(input_tensors[0], coord, -1, this->topology, this->cluster_axis);
    TT_FATAL(forward_coord.has_value() || backward_coord.has_value(), "DEBUG: forward_coord or backward_coord is null");

    return strided_all_gather_async_minimal_default(
        input_tensors[0],
        coord,
        forward_coord,
        backward_coord,
        output_tensors[0],
        this->dim,
        this->num_links,
        this->ring_size,
        device_index,
        this->topology,
        this->semaphore,
        this->barrier_semaphore,
        this->using_persistent_buffers,
        this->sub_device_id,
        this->tiles_per_chunk,
        this->num_workers_per_link,
        this->num_buffers_per_channel,
        this->mm_cores_y,
        this->mm_block_h,
        this->mm_block_w);
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
        this->tiles_per_chunk,
        this->num_workers_per_link,
        this->num_buffers_per_channel,
        this->mm_cores_y,
        this->mm_block_h,
        this->mm_block_w,
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
    const std::optional<uint32_t>& tiles_per_chunk,
    const std::optional<uint32_t>& num_workers_per_link,
    const std::optional<uint32_t>& num_buffers_per_channel,
    const std::optional<uint32_t>& mm_cores_y,
    const std::optional<uint32_t>& mm_block_h,
    const std::optional<uint32_t>& mm_block_w) {
    TT_FATAL(
        std::getenv("TT_METAL_SLOW_DISPATCH_MODE") == nullptr,
        "strided_all_gather_async op is only supported for Fast Dispatch");

    uint32_t num_devices = ::ttnn::ccl::get_topological_dimension(input_tensor, std::nullopt);

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
                   tiles_per_chunk,
                   num_workers_per_link,
                   num_buffers_per_channel,
                   mm_cores_y,
                   mm_block_h,
                   mm_block_w),
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
    std::optional<uint32_t> tiles_per_chunk,
    std::optional<uint32_t> num_workers_per_link,
    std::optional<uint32_t> num_buffers_per_channel,
    std::optional<uint32_t> mm_cores_y,
    std::optional<uint32_t> mm_block_h,
    std::optional<uint32_t> mm_block_w) {
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
        tiles_per_chunk,
        num_workers_per_link,
        num_buffers_per_channel,
        mm_cores_y,
        mm_block_h,
        mm_block_w);
}

}  // namespace ccl
}  // namespace experimental
}  // namespace operations

}  // namespace ttnn
