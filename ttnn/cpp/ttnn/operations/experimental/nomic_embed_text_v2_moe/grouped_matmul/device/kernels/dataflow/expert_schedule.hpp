// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The order every kernel of grouped_matmul walks the blocks in. Each one derives it from
// expert_counts on its own, so they must agree to the block: a kernel that waits for a block another
// never sends hangs the device.

#pragma once

#include <cstdint>

namespace grouped_matmul {

constexpr uint32_t kBlockRows = 32;

// The compute kernel's schedule word for an expert with no rows: no weight slice is sent for it.
constexpr uint32_t kSkipExpert = 0xFFFFFFFF;

// routing_metadata's layout: expert e owns blocks [first_block[e], first_block[e] + num_blocks[e]),
// the regions in expert order, each its count rounded up to whole 32-row blocks. Returns the number
// of blocks in use; blocks past it hold no slot and are neither computed nor written. The regions
// stop at input_blocks, the blocks the input holds, so counts that do not match the input never
// move a read or a write past its rows or the output's.
template <uint32_t num_experts>
uint32_t expert_regions(
    volatile const uint32_t* counts,
    uint32_t input_blocks,
    uint32_t (&first_block)[num_experts],
    uint32_t (&num_blocks)[num_experts]) {
    uint32_t next_block = 0;
    for (uint32_t expert = 0; expert < num_experts; ++expert) {
        const uint32_t count = counts[expert];
        const uint32_t blocks = count / kBlockRows + (count % kBlockRows != 0 ? 1 : 0);
        const uint32_t room = input_blocks - next_block;
        first_block[expert] = next_block;
        num_blocks[expert] = blocks < room ? blocks : room;
        next_block += num_blocks[expert];
    }
    return next_block;
}

// The blocks of one expert's region a lane multiplies: count of them, from first, stride apart.
struct LaneBlocks {
    uint32_t first;
    uint32_t stride;
    uint32_t count;
};

// The shared schedule deals the blocks round-robin to the lanes, lane l taking blocks l, l +
// num_lanes, ..., so that one weight slice serves all lanes at once. With per-expert lanes, lane l
// multiplies all of expert l's blocks and none of the others'. Every kernel walks the blocks
// through this function, so the two schedules cannot disagree between kernels.
template <uint32_t num_lanes, bool per_expert_lanes>
LaneBlocks lane_blocks(uint32_t expert, uint32_t first_block, uint32_t num_blocks, uint32_t lane) {
    if constexpr (per_expert_lanes) {
        return {first_block, 1, expert == lane ? num_blocks : 0};
    } else {
        const uint32_t offset = first_block % num_lanes;
        const uint32_t first = first_block - offset + lane + (lane < offset ? num_lanes : 0);
        const uint32_t end = first_block + num_blocks;
        return {first, num_lanes, first < end ? (end - 1 - first) / num_lanes + 1 : 0};
    }
}

}  // namespace grouped_matmul
