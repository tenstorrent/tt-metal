// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "all_broadcast.hpp"
#include <utility>
#include "ttnn/operations/ccl/all_broadcast/device/all_broadcast_op.hpp"
#include "ttnn/distributed/types.hpp"
#include "ttnn/global_semaphore.hpp"

namespace ttnn::operations::ccl {

std::vector<ttnn::Tensor> ExecuteAllBroadcast::invoke(
    const ttnn::Tensor& input_tensor,
    std::optional<uint32_t> cluster_axis,
    const std::optional<tt::tt_metal::SubDeviceId>& subdevice_id,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    std::optional<uint32_t> num_links,
    std::optional<ttnn::ccl::Topology> topology) {
    // Default values for num_links and topology
    uint32_t num_links_ = num_links.value_or(1);
    ttnn::ccl::Topology topology_ = topology.value_or(ttnn::ccl::Topology::Linear);

    return ttnn::operations::ccl::all_broadcast(
        input_tensor, cluster_axis, subdevice_id, memory_config, num_links_, topology_);
}

}  // namespace ttnn::operations::ccl
