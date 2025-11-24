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
    std::vector<IDevice*> devices = ttnn::ccl::get_active_physical_devices(input_tensor);

    auto mesh_device = input_tensor.device();
    uint32_t num_devices;
    const auto& mesh_view = mesh_device->get_view();
    // Use the mesh dimensions to determine the ring size
    num_devices = (cluster_axis == 0) ? mesh_view.num_rows() : mesh_view.num_cols();

    TT_FATAL(num_devices > 1, "neighbor_pad_async op will only work for num_devices > 1, but has {}", num_devices);

    tt::tt_fabric::Topology topology_ = ::ttnn::ccl::get_usable_topology(input_tensor, topology, cluster_axis);

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
                   topology_,
                   num_devices,
                   secondary_cluster_axis,
                   secondary_mesh_shape),
               {input_tensor},
               {},
               {})
        .at(0);
}

}  // namespace ttnn::operations::experimental::ccl
