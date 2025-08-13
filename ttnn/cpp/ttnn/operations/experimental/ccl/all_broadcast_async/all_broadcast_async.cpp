// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "all_broadcast_async.hpp"
#include <utility>
#include "ttnn/operations/experimental/ccl/all_broadcast_async/device/all_broadcast_async_op.hpp"
#include "ttnn/distributed/types.hpp"
#include "ttnn/global_semaphore.hpp"

namespace ttnn::operations::experimental::ccl {

std::vector<ttnn::Tensor> ExecuteAllBroadcastAsync::invoke(
    const ttnn::Tensor& input_tensor,
    const GlobalSemaphore& multi_device_global_semaphore,
    const GlobalSemaphore& barrier_semaphore,
    const uint32_t num_links,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    const ttnn::ccl::Topology topology,
    std::optional<uint32_t> cluster_axis,
    std::optional<tt::tt_metal::SubDeviceId> subdevice_id) {
    return ttnn::operations::experimental::ccl::all_broadcast_async(
        input_tensor,
        multi_device_global_semaphore,
        barrier_semaphore,
        num_links,
        memory_config,
        topology,
        cluster_axis,
        subdevice_id);
}

}  // namespace ttnn::operations::experimental::ccl
