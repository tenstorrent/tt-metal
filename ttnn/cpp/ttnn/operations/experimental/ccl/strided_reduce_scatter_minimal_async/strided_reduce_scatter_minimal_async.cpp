// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "strided_reduce_scatter_minimal_async.hpp"
#include "ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/reduce_scatter_minimal_async.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"

namespace ttnn::operations::experimental::ccl {

ttnn::Tensor ExecuteStridedReduceScatterMinimalAsync::invoke(
    const ttnn::Tensor& input_tensor,
    const std::optional<std::vector<ttnn::Tensor>>& persistent_output_buffers,
    const int32_t dim,
    const std::vector<GlobalSemaphore>& multi_device_global_semaphore,
    const std::optional<GlobalSemaphore>& barrier_semaphore,
    const uint32_t num_links,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    const std::optional<ttnn::MemoryConfig>& intermediate_memory_config,
    const ttnn::ccl::Topology topology,
    std::optional<tt::tt_metal::SubDeviceId> sub_device_id,
    std::optional<uint32_t> cluster_axis,
    std::optional<uint32_t> chunks_per_sync,
    std::optional<uint32_t> num_workers_per_link,
    std::optional<uint32_t> num_buffers_per_channel,
    std::optional<uint32_t> tiles_per_chunk,
    std::optional<uint32_t> mm_cores_y,
    std::optional<uint32_t> mm_block_ht,
    std::optional<uint32_t> mm_block_wt) {
    // TODO: Implement proper strided reduce scatter minimal async
    // For now, delegate to the regular reduce_scatter_minimal_async implementation
    log_debug(tt::LogOp, "strided_reduce_scatter_minimal_async: delegating to reduce_scatter_minimal_async");

    return ExecuteReduceScatterMinimalAsync::invoke(
        input_tensor,
        persistent_output_buffers,
        dim,
        multi_device_global_semaphore,
        barrier_semaphore,
        num_links,
        memory_config,
        intermediate_memory_config,
        topology,
        sub_device_id,
        cluster_axis,
        chunks_per_sync,
        num_workers_per_link,
        num_buffers_per_channel);
}

}  // namespace ttnn::operations::experimental::ccl
