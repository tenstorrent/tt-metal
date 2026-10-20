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

// Blocks are dealt round-robin to the lanes: lane l takes blocks l, l + num_lanes, ... Every lane
// then walks the experts in order, so one weight slice serves all lanes at once.
template <uint32_t num_lanes>
uint32_t first_lane_block(uint32_t begin, uint32_t lane) {
    const uint32_t offset = begin % num_lanes;
    return begin - offset + lane + (lane < offset ? num_lanes : 0);
}

// How many of blocks [begin, end) lane `lane` takes.
template <uint32_t num_lanes>
uint32_t lane_block_count(uint32_t begin, uint32_t end, uint32_t lane) {
    const uint32_t first = first_lane_block<num_lanes>(begin, lane);
    return first < end ? (end - 1 - first) / num_lanes + 1 : 0;
}

}  // namespace grouped_matmul
