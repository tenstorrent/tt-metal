// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <vector>

#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/global_semaphore.hpp"

namespace ttnn {
namespace experimental {

ttnn::Tensor neighbor_pad_async(
    const ttnn::Tensor& input_tensor,
    int32_t dim,
    uint32_t padding_left,
    uint32_t padding_right,
    const std::string& padding_mode,
    uint32_t cluster_axis,
    const GlobalSemaphore& final_semaphore,
    const GlobalSemaphore& barrier_semaphore,
    std::optional<size_t> num_preferred_links = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    std::optional<ttnn::ccl::Topology> topology = std::nullopt,
    std::optional<uint32_t> secondary_cluster_axis = std::nullopt,
    const std::optional<std::vector<uint32_t>>& secondary_mesh_shape = std::nullopt);

}  // namespace experimental
}  // namespace ttnn
