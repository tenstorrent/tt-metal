// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
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
        const ttnn::Tensor& input_tensors,
        int32_t dim,
        uint32_t padding,
        const std::string& padding_mode,
        bool direction,
        uint32_t cluster_axis,
        const GlobalSemaphore& final_semaphore,
        const GlobalSemaphore& barrier_semaphore,
        const MeshDevice& mesh_device,
        std::optional<size_t> num_preferred_links = std::nullopt,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        std::optional<ttnn::ccl::Topology> topology = std::nullopt);
};

}  // namespace operations::experimental::ccl

namespace experimental {

constexpr auto neighbor_pad_async = ttnn::register_operation<
    "ttnn::experimental::neighbor_pad_async",
    ttnn::operations::experimental::ccl::ExecuteNeighborPadAsync>();

}  // namespace experimental
}  // namespace ttnn
