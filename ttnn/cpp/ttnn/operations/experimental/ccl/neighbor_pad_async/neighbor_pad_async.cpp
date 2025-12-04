// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "neighbor_pad_async.hpp"
#include "ttnn/operations/experimental/ccl/neighbor_pad_async/device/neighbor_pad_async_device_operation.hpp"

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
    return ttnn::prim::neighbor_pad_async(
        input_tensor,
        dim,
        padding_left,
        padding_right,
        padding_mode,
        cluster_axis,
        final_semaphore,
        barrier_semaphore,
        num_preferred_links,
        memory_config,
        topology,
        secondary_cluster_axis,
        secondary_mesh_shape);
}

}  // namespace ttnn::operations::experimental::ccl
