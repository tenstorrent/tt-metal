// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "strided_all_gather_async.hpp"
#include <utility>
#include "ttnn/operations/experimental/ccl/strided_all_gather_async/device/strided_all_gather_async_op.hpp"
#include "ttnn/operations/experimental/ccl/all_broadcast_async/device/all_broadcast_async_op.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/data_movement/concat/concat.hpp"
#include "ttnn/operations/copy/typecast/typecast.hpp"
#include "ttnn/distributed/types.hpp"
#include "ttnn/global_semaphore.hpp"

namespace ttnn::operations::experimental::ccl {

ttnn::Tensor ExecuteStridedAllGatherAsync::invoke(
    const ttnn::Tensor& input_tensor,
    const std::optional<ttnn::Tensor>& persistent_output_buffer,
    const int32_t dim,
    const std::vector<GlobalSemaphore>& multi_device_global_semaphore,
    const uint32_t num_links,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    const ttnn::ccl::Topology topology,
    std::optional<tt::tt_metal::SubDeviceId> subdevice_id,
    std::optional<uint32_t> cluster_axis,
    const std::optional<GlobalSemaphore>& barrier_semaphore,
    std::optional<uint32_t> chunks_per_sync,
    std::optional<uint32_t> num_workers_per_link,
    std::optional<uint32_t> num_buffers_per_channel) {
    return ttnn::operations::experimental::ccl::strided_all_gather_async(
        input_tensor,
        persistent_output_buffer,
        dim,
        multi_device_global_semaphore,
        num_links,
        memory_config,
        topology,
        subdevice_id,
        cluster_axis,
        barrier_semaphore,
        chunks_per_sync,
        num_workers_per_link,
        num_buffers_per_channel);
}
}  // namespace ttnn::operations::experimental::ccl
