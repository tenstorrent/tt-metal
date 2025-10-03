// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "broadcast.hpp"
#include <utility>
#include "ttnn/operations/experimental/ccl/broadcast/device/broadcast_op.hpp"
#include "ttnn/distributed/types.hpp"
#include "ttnn/global_semaphore.hpp"

namespace ttnn::operations::experimental::ccl {

ttnn::Tensor ExecuteBroadcast::invoke(
    const ttnn::Tensor& input_tensor,
    const MeshCoordinate& sender_coord,
    const uint32_t num_links,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    const ttnn::ccl::Topology topology,
    std::optional<uint32_t> cluster_axis,
    std::optional<tt::tt_metal::SubDeviceId> subdevice_id) {
    return ttnn::operations::experimental::ccl::broadcast(
        input_tensor, sender_coord, num_links, memory_config, topology, cluster_axis, subdevice_id);
}

}  // namespace ttnn::operations::experimental::ccl
