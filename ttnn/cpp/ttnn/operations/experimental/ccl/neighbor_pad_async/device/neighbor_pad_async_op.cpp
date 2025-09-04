// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "neighbor_pad_async_op.hpp"
#include <tt-metalium/fabric.hpp>
#include "ttnn/operations/functions.hpp"
#include "ttnn/operations/math.hpp"
#include "ttnn/global_semaphore.hpp"

#include "ttnn/tensor/tensor_utils.hpp"

namespace ttnn {

void NeighborPadAsync::validate_with_output_tensors(
    const std::vector<Tensor>& input_tensors, const std::vector<std::optional<Tensor>>& output_tensors) const {
    TT_FATAL(this->dim < 3, "Error, neighbor pad currently only supports padding non last dim, provided {}", this->dim);
}

std::vector<ttnn::TensorSpec> NeighborPadAsync::compute_output_specs(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors[0];
    auto shape = input_tensor.logical_shape();
    shape[this->dim] += (this->padding_left + this->padding_right);
    return {TensorSpec(
        shape, TensorLayout(input_tensor.dtype(), input_tensor.tensor_spec().page_config(), output_mem_config))};
}

std::vector<Tensor> NeighborPadAsync::create_output_tensors(
    const std::vector<Tensor>& input_tensors, const std::vector<std::optional<Tensor>>& optional_output_tensors) const {
    return tt::tt_metal::operation::default_create_output_tensors(*this, input_tensors, optional_output_tensors);
}

tt::tt_metal::operation::MeshWorkloadWithCallbacks NeighborPadAsync::create_mesh_workload(
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const std::vector<Tensor>& input_tensors,
    std::vector<Tensor>& output_tensors) const {
    return ccl::create_mesh_workload_from_programs(
        tensor_coords, input_tensors, output_tensors, [&, this](const ttnn::MeshCoordinate& coord) {
            return create_program_at(coord, input_tensors, output_tensors);
        });
}

tt::tt_metal::operation::ProgramWithCallbacks NeighborPadAsync::create_program_at(
    const MeshCoordinate& coord, const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const {
    auto mesh_device = input_tensors[0].device();
    IDevice* target_device = mesh_device ? mesh_device->get_device(coord) : input_tensors[0].device();
    std::vector<IDevice*> devices_to_use = {};
    const auto& mesh_view = input_tensors[0].device()->get_view();
    // User specified the cluster-axis. Derive devices based on the current coordinate
    // and the cluster-axis.
    devices_to_use =
        (this->cluster_axis == 0) ? mesh_view.get_devices_on_column(coord[1]) : mesh_view.get_devices_on_row(coord[0]);
    uint32_t target_ring_size = devices_to_use.size();

    std::optional<IDevice*> forward_device = std::nullopt;
    std::optional<IDevice*> backward_device = std::nullopt;
    uint32_t device_index = 0;  // Initialize device index
    for (uint32_t i = 0; i < target_ring_size; ++i) {
        if (devices_to_use.at(i) == target_device) {
            device_index = i;
            if (i != 0) {
                backward_device = devices_to_use.at(i - 1);
            }
            if (i != target_ring_size - 1) {
                forward_device = devices_to_use.at(i + 1);
            }
        }
    }

    return neighbor_pad_async_minimal(
        input_tensors[0],
        target_device,
        forward_device,
        backward_device,
        output_tensors[0],
        this->dim,
        this->padding_left,
        this->padding_right,
        this->padding_mode,
        this->final_semaphore,
        this->barrier_semaphore,
        this->num_links,
        this->topology,
        target_ring_size,
        device_index);
}

tt::tt_metal::operation::Hash NeighborPadAsync::compute_program_hash(const std::vector<Tensor>& input_tensors) const {
    auto input_shape = input_tensors[0].padded_shape();
    auto input_memory_layout = input_tensors[0].layout();
    auto input_dtype = input_tensors[0].dtype();
    auto input_memory_config = input_tensors[0].memory_config();
    uint32_t semaphore_address = this->final_semaphore.address();
    uint32_t barrier_semaphore_address = this->barrier_semaphore.address();
    return tt::tt_metal::operation::hash_operation<NeighborPadAsync>(
        this->dim,
        this->padding_left,
        this->padding_right,
        this->padding_mode,
        this->num_links,
        this->output_mem_config,
        this->topology,
        this->cluster_axis,
        this->ring_size,
        input_shape,
        input_memory_layout,
        input_dtype,
        input_memory_config);
}

namespace operations {
namespace experimental {
namespace ccl {

namespace {
Tensor neighbor_pad_async_impl(
    const Tensor& input_tensor,
    const int32_t dim,
    const uint32_t padding_left,
    const uint32_t padding_right,
    const std::string& padding_mode,
    const uint32_t cluster_axis,
    const GlobalSemaphore& final_semaphore,
    const GlobalSemaphore& barrier_semaphore,
    const MeshDevice& mesh_device,
    const std::vector<IDevice*>& devices,
    const std::optional<size_t> num_preferred_links,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<ttnn::ccl::Topology> topology) {
    TT_FATAL(
        std::getenv("TT_METAL_SLOW_DISPATCH_MODE") == nullptr,
        "neighbor_pad_async op is only supported for Fast Dispatch");

    uint32_t num_devices;
    const auto& mesh_view = mesh_device.get_view();
    // Use the mesh dimensions to determine the ring size
    num_devices = (cluster_axis == 0) ? mesh_view.num_rows() : mesh_view.num_cols();

    TT_FATAL(num_devices > 1, "neighbor_pad_async op will only work for num_devices > 1, but has {}", num_devices);
    ttnn::ccl::Topology ccl_topology = topology.value_or(ttnn::ccl::Topology::Linear);

    CoreCoord grid_size = devices[0]->compute_with_storage_grid_size();
    auto core_grid = CoreRange({0, 0}, {grid_size.x - 1, grid_size.y - 1});

    return tt::tt_metal::operation::run(
               ttnn::NeighborPadAsync(
                   devices,
                   dim,
                   padding_left,
                   padding_right,
                   padding_mode,
                   cluster_axis,
                   final_semaphore,
                   barrier_semaphore,
                   num_preferred_links.value_or(1),
                   memory_config.value_or(input_tensor.memory_config()),
                   ccl_topology,
                   num_devices),
               {input_tensor},
               {},
               {})
        .at(0);
}
}  // namespace

Tensor neighbor_pad_async(
    const Tensor& input_tensor,
    const int32_t dim,
    const uint32_t padding_left,
    const uint32_t padding_right,
    const std::string& padding_mode,
    const uint32_t cluster_axis,
    const GlobalSemaphore& final_semaphore,
    const GlobalSemaphore& barrier_semaphore,
    const MeshDevice& mesh_device,
    const std::optional<size_t> num_preferred_links,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<ttnn::ccl::Topology> topology) {
    std::vector<IDevice*> devices = ttnn::ccl::get_active_physical_devices(input_tensor);

    return neighbor_pad_async_impl(
        input_tensor,
        dim,
        padding_left,
        padding_right,
        padding_mode,
        cluster_axis,
        final_semaphore,
        barrier_semaphore,
        mesh_device,
        devices,
        num_preferred_links,
        memory_config,
        topology);
}

}  // namespace ccl
}  // namespace experimental
}  // namespace operations

}  // namespace ttnn
