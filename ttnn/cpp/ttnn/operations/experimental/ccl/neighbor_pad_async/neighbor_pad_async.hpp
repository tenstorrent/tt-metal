// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/global_semaphore.hpp"

namespace ttnn {
namespace operations::experimental::ccl {

struct ExecuteNeighborPadAsync {
    static ttnn::Tensor invoke(
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
};

}  // namespace operations::experimental::ccl

namespace experimental {

constexpr auto neighbor_pad_async = ttnn::register_operation<
    "ttnn::experimental::neighbor_pad_async",
    ttnn::operations::experimental::ccl::ExecuteNeighborPadAsync>();

}  // namespace experimental
}  // namespace ttnn
