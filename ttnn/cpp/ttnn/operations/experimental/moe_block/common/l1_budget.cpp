// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "l1_budget.hpp"

#include <algorithm>

#include <tt-metalium/allocator.hpp>
#include <tt-metalium/hal_types.hpp>
#include <tt-metalium/mesh_device.hpp>

#include "moe_block_resources.hpp"

namespace ttnn::operations::experimental::moe_block {

const BlockL1Plan* block_l1_plan(const MoEBlockResources& resources) {
    return resources.find_plan<BlockL1Plan>(kBlockL1Plan);
}

uint32_t free_cb_l1_bytes(
    const tt::tt_metal::distributed::MeshDevice& mesh_device, uint32_t margin_bytes, uint32_t cap_bytes) {
    const uint64_t base = mesh_device.allocator()->get_base_allocator_addr(tt::tt_metal::HalMemType::L1);
    const uint64_t top =
        mesh_device.lowest_occupied_compute_l1_address().value_or(base + mesh_device.l1_size_per_core());
    const uint64_t free = top > base ? top - base : 0;
    const uint64_t usable = free > margin_bytes ? (free - margin_bytes) / 64 * 64 : 0;
    return static_cast<uint32_t>(std::min<uint64_t>(cap_bytes, usable));
}

}  // namespace ttnn::operations::experimental::moe_block
