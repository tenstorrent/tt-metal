// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The one rule every MoE block op uses to size the L1 it turns into program CBs (an arena, an engine's pools): the
// span from the allocator's L1 base up to the lowest occupied L1 address on any compute core (the persistent words /
// tensors sit at the top of the allocator, so the lowest of them bounds the CBs), less a margin for L1 the block
// allocates after the op is planned.

#pragma once

#include <cstdint>
#include <limits>

namespace tt::tt_metal::distributed {
class MeshDevice;
}

namespace ttnn::operations::experimental::moe_block {

class MoEBlockResources;

// The block's L1 plan (block/moe_block_arena.hpp moe_block_plan), kept in the resources under kBlockL1Plan: what the
// block's ops read when they run as one block around a shared arena. Absent: an op on its own.
struct BlockL1Plan {
    uint32_t dispatch_group = 0;  // the dispatch's row group
    // the static CBs of the block's programs that run in the arena after the head (the flat experts, the dispatch):
    // the head's arena stays above them
    uint32_t head_cb_floor = 0;
    // the head's own static CBs (its ports' packet CB): the combine's arena stays above them
    uint32_t head_port_cb = 0;
};
inline constexpr const char* kBlockL1Plan = "moe_block/l1";
// The resources' block L1 plan, or nullptr.
const BlockL1Plan* block_l1_plan(const MoEBlockResources& resources);

// L1 the block takes after an op is planned (the fused tail's kKeptLater).
inline constexpr uint32_t kL1KeptLater = 44 * 1024;

// Bytes of L1 left for program CBs on every compute core: lowest occupied L1 address - L1 base - margin_bytes, rounded
// down to 64 B, at most cap_bytes (0 if the margin does not fit). With nothing allocated, the top is the base plus
// the L1 size per core.
uint32_t free_cb_l1_bytes(
    const tt::tt_metal::distributed::MeshDevice& mesh_device,
    uint32_t margin_bytes = kL1KeptLater,
    uint32_t cap_bytes = std::numeric_limits<uint32_t>::max());

}  // namespace ttnn::operations::experimental::moe_block
