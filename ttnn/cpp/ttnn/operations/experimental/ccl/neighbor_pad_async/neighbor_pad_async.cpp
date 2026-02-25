// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "neighbor_pad_async.hpp"
#include "ttnn/operations/experimental/ccl/neighbor_pad_async/device/neighbor_pad_async_device_operation.hpp"

namespace ttnn::operations::experimental::ccl {

ttnn::Tensor ExecuteNeighborPadAsync::invoke(
    const ttnn::Tensor& input_tensor,
    std::vector<int32_t> dim,
    std::vector<uint32_t> padding_left,
    std::vector<uint32_t> padding_right,
    const std::string& padding_mode,
    std::vector<uint32_t> cluster_axis,
    std::vector<GlobalSemaphore> neighbor_semaphore,
    std::vector<GlobalSemaphore> barrier_semaphore,
    std::optional<std::vector<size_t>> num_preferred_links,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<ttnn::ccl::Topology> topology,
    std::optional<uint32_t> secondary_cluster_axis,
    const std::optional<std::vector<uint32_t>>& secondary_mesh_shape) {
    TT_FATAL(dim.size() >= 1 && dim.size() <= 2, "dim must have 1 or 2 elements, got {}", dim.size());

    std::vector<size_t> links = num_preferred_links.value_or(std::vector<size_t>(dim.size(), 1));

    // neighbor_semaphore[0] is always the H (primary) neighbor semaphore.
    // neighbor_semaphore[1] is the W (secondary) neighbor semaphore for 2D padding.
    // For 1D padding, reuse [0] as the W semaphore (it won't be used).
    const auto& h_neighbor_sem = neighbor_semaphore[0];
    const auto& w_neighbor_sem = neighbor_semaphore.size() >= 2 ? neighbor_semaphore[1] : neighbor_semaphore[0];

    // Unpack secondary dimension if present
    std::optional<uint32_t> pad_dim2;
    uint32_t pad2_left = 0;
    uint32_t pad2_right = 0;
    std::optional<uint32_t> pad2_cluster_axis;
    std::optional<size_t> pad2_num_links;

    if (dim.size() == 2) {
        pad_dim2 = static_cast<uint32_t>(dim[1]);
        pad2_left = padding_left[1];
        pad2_right = padding_right[1];
        pad2_cluster_axis = cluster_axis[1];
        pad2_num_links = links.size() >= 2 ? links[1] : 1;
    }

    return ttnn::prim::neighbor_pad_async(
        input_tensor,
        dim[0],
        padding_left[0],
        padding_right[0],
        padding_mode,
        cluster_axis[0],
        h_neighbor_sem,
        w_neighbor_sem,
        barrier_semaphore[0],
        links[0],
        memory_config,
        topology,
        secondary_cluster_axis,
        secondary_mesh_shape,
        pad_dim2,
        pad2_left,
        pad2_right,
        pad2_cluster_axis,
        pad2_num_links);
}

}  // namespace ttnn::operations::experimental::ccl
