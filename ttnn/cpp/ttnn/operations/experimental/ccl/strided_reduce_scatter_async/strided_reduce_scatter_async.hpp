// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/global_semaphore.hpp"
#include "ttnn/operations/experimental/ccl/composite_common.hpp"

namespace ttnn {
namespace operations::experimental::ccl {

struct ExecuteStridedReduceScatterAsync {
    static ttnn::Tensor invoke(
        const ttnn::Tensor& input_tensor,
        const std::optional<std::vector<ttnn::Tensor>>& persistent_output_buffers,
        int32_t dim,
        const std::vector<GlobalSemaphore>& multi_device_global_semaphore,
        const std::optional<GlobalSemaphore>& barrier_semaphore = std::nullopt,
        uint32_t num_links = 1,
        const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
        const std::optional<ttnn::MemoryConfig>& intermediate_memory_config = std::nullopt,
        ttnn::ccl::Topology topology = ttnn::ccl::Topology::Ring,
        std::optional<tt::tt_metal::SubDeviceId> sub_device_id = std::nullopt,
        std::optional<uint32_t> cluster_axis = std::nullopt,
        std::optional<uint32_t> chunks_per_sync = std::nullopt,
        std::optional<uint32_t> num_workers_per_link = std::nullopt,
        std::optional<uint32_t> num_buffers_per_channel = std::nullopt,
        std::optional<uint32_t> mm_cores_y = std::nullopt,
        std::optional<uint32_t> mm_block_ht = std::nullopt,
        std::optional<uint32_t> mm_block_wt = std::nullopt,
        std::optional<uint32_t> mm_M_block_ht = std::nullopt,
        std::optional<uint32_t> mm_N_block_wt = std::nullopt,
        std::optional<uint32_t> chunk_width_in_mm_blocks = std::nullopt);
};

}  // namespace operations::experimental::ccl

namespace experimental {

constexpr auto strided_reduce_scatter_async = ttnn::register_operation<
    "ttnn::experimental::strided_reduce_scatter_async",
    ttnn::operations::experimental::ccl::ExecuteStridedReduceScatterAsync>();

}  // namespace experimental
}  // namespace ttnn
