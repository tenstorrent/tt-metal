// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include <tt-metalium/sub_device.hpp>

namespace ttnn {

// Broadcast the shard at `sender_ring_index` (along `cluster_axis`) to every device on that ring line,
// via manual per-hop unicast relay over FABRIC_1D / FABRIC_1D_RING. The op runs independently on each
// line parallel to the orthogonal axis, so a tp-sharded input broadcasts each orthogonal row's own data
// (the per-line semantics ttnn.broadcast lacks). Output has the same shape/spec as the input; every
// device on the line ends holding the sender shard's data.
//
// v1 constraints: single sender; one-way around the ring (requires the wrap link -> Ring topology).
ttnn::Tensor broadcast_ring(
    const ttnn::Tensor& input_tensor,
    uint32_t sender_ring_index,
    uint32_t cluster_axis,
    std::optional<uint32_t> num_links = std::nullopt,
    const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
    ttnn::ccl::Topology topology = ttnn::ccl::Topology::Ring,
    std::optional<tt::tt_metal::SubDeviceId> subdevice_id = std::nullopt);

}  // namespace ttnn
