// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/global_semaphore.hpp"

namespace ttnn {
namespace operations::experimental::ccl {

struct ExecuteSliceReshardAsync {
    static ttnn::Tensor invoke(
        const ttnn::Tensor& input_tensors,
        int32_t dim,
        uint32_t output_dim_offset,
        uint32_t output_dim_shape,
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

constexpr auto slice_reshard_async = ttnn::register_operation<
    "ttnn::experimental::slice_reshard_async",
    ttnn::operations::experimental::ccl::ExecuteSliceReshardAsync>();

}  // namespace experimental
}  // namespace ttnn
