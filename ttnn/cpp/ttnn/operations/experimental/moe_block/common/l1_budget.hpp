// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The rule the MoE head and the MoE tail measure the L1 they may turn into program CBs (an arena, an engine's
// pools) by: free_l1_span_bytes, the span from the allocator's L1 base up to the lowest occupied L1 address on any
// compute core (the persistent words / tensors sit at the top of the allocator, so the lowest of them bounds the CBs),
// or the whole allocatable L1 bank with nothing allocated. Each takes its own margin off it for L1 the block allocates
// after the op is planned: the head (moe_head.cpp moe_head_l1_free) kL1Margin, the tail (free_cb_l1_bytes)
// kL1KeptLater.

#pragma once

#include <cstdint>
#include <limits>
#include <optional>

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
// The resources' block L1 plan (a copy), or nullopt.
std::optional<BlockL1Plan> block_l1_plan(const MoEBlockResources& resources);

// The MoE tail's margin: L1 the block took above the tail's arena after the tail was planned, measured on the
// Python reference block (the exchange's and the head's persistent words and the head's holes, all in L1 there: 36 KB
// for Kimi-K3 768 rows on a 4x4, whose 1395 KB arena then no longer fit the bank) and rounded up to 44 KB. With the
// persistent words in L1_SMALL (MoEBlockConfig.persist_l1_small, as the models run) the block takes less L1 later; the
// margin is kept so the tail's plans (its ring slots) stay as they were tuned.
inline constexpr uint32_t kL1KeptLater = 44 * 1024;

// The L1 a compute core has free for program CBs now: the lowest occupied L1 address on any compute core less the
// allocator's L1 base - with nothing allocated in L1, the allocatable L1 bank (allocator get_bank_size(L1)), which
// ends below L1_SMALL.
uint64_t free_l1_span_bytes(const tt::tt_metal::distributed::MeshDevice& mesh_device);

// Bytes of L1 left for program CBs on every compute core: free_l1_span_bytes - margin_bytes, rounded down to 64 B, at
// most cap_bytes (0 if the margin does not fit).
uint32_t free_cb_l1_bytes(
    const tt::tt_metal::distributed::MeshDevice& mesh_device,
    uint32_t margin_bytes = kL1KeptLater,
    uint32_t cap_bytes = std::numeric_limits<uint32_t>::max());

}  // namespace ttnn::operations::experimental::moe_block
