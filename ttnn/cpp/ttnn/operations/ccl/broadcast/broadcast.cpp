// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "broadcast.hpp"
#include <utility>
#include "ttnn/operations/ccl/broadcast/device/broadcast_op.hpp"
#include "ttnn/distributed/types.hpp"
#include "ttnn/global_semaphore.hpp"

namespace ttnn::operations::ccl {

ttnn::Tensor ExecuteBroadcast::invoke(
    const ttnn::Tensor& input_tensor,
    const MeshCoordinate& sender_coord,
    const uint32_t num_links,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    const ttnn::ccl::Topology topology,
    std::optional<uint32_t> cluster_axis,
    std::optional<tt::tt_metal::SubDeviceId> subdevice_id) {
    tt::tt_fabric::Topology topology_ = ::ttnn::ccl::get_usable_topology(input_tensor, topology, cluster_axis);
    return ttnn::operations::ccl::broadcast(
        input_tensor, sender_coord, num_links, memory_config, topology_, cluster_axis, subdevice_id);
}

}  // namespace ttnn::operations::ccl
