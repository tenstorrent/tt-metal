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
    const uint32_t num_links,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    const ttnn::ccl::Topology topology,
    std::optional<tt::tt_metal::SubDeviceId> subdevice_id) {
    return ttnn::operations::experimental::ccl::all_broadcast_async(
        input_tensor, multi_device_global_semaphore, num_links, memory_config, topology, subdevice_id);
}

std::vector<ttnn::Tensor> ExecuteAllBroadcastAsync::invoke(
    const ttnn::Tensor& input_tensor,
    const uint32_t cluster_axis,
    const MeshDevice& mesh_device,
    const ttnn::ccl::Topology topology,
    const GlobalSemaphore& multi_device_global_semaphore,
    const std::optional<ttnn::Tensor>& persistent_output_tensor,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<size_t> num_preferred_links,
    std::optional<tt::tt_metal::SubDeviceId> subdevice_id) {
    return ttnn::operations::experimental::ccl::all_broadcast_async(
        input_tensor,
        cluster_axis,
        mesh_device,
        topology,
        multi_device_global_semaphore,
        persistent_output_tensor,
        memory_config,
        num_preferred_links,
        subdevice_id);
}

}  // namespace ttnn::operations::experimental::ccl
