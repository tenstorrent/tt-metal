// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "broadcast_ring.hpp"
#include "ttnn/operations/experimental/ccl/broadcast_ring/device/broadcast_ring_device_operation.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/ccl/common/host/moe_utils.hpp"

namespace ttnn {

ttnn::Tensor broadcast_ring(
    const ttnn::Tensor& input_tensor,
    uint32_t sender_ring_index,
    uint32_t cluster_axis,
    std::optional<uint32_t> num_links,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    ttnn::ccl::Topology topology,
    std::optional<tt::tt_metal::SubDeviceId> subdevice_id,
    uint32_t chunk_size_tiles,
    uint32_t broadcast_offset_tiles,
    uint32_t broadcast_num_tiles,
    bool use_l1_relay) {
    auto* mesh_device = input_tensor.device();
    TT_FATAL(mesh_device != nullptr, "broadcast_ring requires a mesh device");
    uint32_t num_links_ = num_links.value_or(ttnn::operations::ccl::common::get_num_links(*mesh_device, cluster_axis));
    tt::tt_fabric::Topology topology_ =
        ::ttnn::ccl::get_usable_topology(input_tensor, std::optional<tt::tt_fabric::Topology>(topology), cluster_axis);
    return ttnn::prim::broadcast_ring(
        input_tensor,
        sender_ring_index,
        cluster_axis,
        num_links_,
        memory_config,
        topology_,
        subdevice_id,
        chunk_size_tiles,
        broadcast_offset_tiles,
        broadcast_num_tiles,
        use_l1_relay);
}

}  // namespace ttnn
