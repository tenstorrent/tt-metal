// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "neighbor_pad_async.hpp"
#include <utility>
#include "ttnn/operations/experimental/ccl/neighbor_pad_async/device/neighbor_pad_async_op.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/data_movement/concat/concat.hpp"
#include "ttnn/operations/copy/typecast/typecast.hpp"
#include "ttnn/distributed/types.hpp"
#include "ttnn/global_semaphore.hpp"

namespace ttnn::operations::experimental::ccl {

ttnn::Tensor ExecuteNeighborPadAsync::invoke(
    const ttnn::Tensor& input_tensor,
    int32_t dim,
    uint32_t padding_left,
    uint32_t padding_right,
    const std::string& padding_mode,
    uint32_t cluster_axis,
    const GlobalSemaphore& final_semaphore,
    const GlobalSemaphore& barrier_semaphore,
    std::optional<size_t> num_preferred_links,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<ttnn::ccl::Topology> topology,
    std::optional<uint32_t> secondary_cluster_axis,
    const std::optional<std::vector<uint32_t>>& secondary_mesh_shape) {
    TT_FATAL(
        std::getenv("TT_METAL_SLOW_DISPATCH_MODE") == nullptr,
        "neighbor_pad_async op is only supported for Fast Dispatch");
    std::vector<IDevice*> devices = ttnn::ccl::get_active_physical_devices(input_tensor);

    auto mesh_device = input_tensor.device();
    uint32_t num_devices;
    const auto& mesh_view = mesh_device->get_view();
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
                   num_devices,
                   secondary_cluster_axis,
                   secondary_mesh_shape),
               {input_tensor},
               {},
               {})
        .at(0);
}

}  // namespace ttnn::operations::experimental::ccl
