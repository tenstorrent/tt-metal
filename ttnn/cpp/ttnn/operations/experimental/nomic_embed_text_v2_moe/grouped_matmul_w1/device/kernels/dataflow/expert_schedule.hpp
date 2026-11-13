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

// The order every kernel walks the experts in: most blocks first (stable), so the expert computed after the last
// weight slice lands is the smallest.
template <uint32_t num_experts>
void expert_order(const uint32_t (&num_blocks)[num_experts], uint32_t (&order)[num_experts]) {
    for (uint32_t position = 0; position < num_experts; ++position) {
        order[position] = position;
    }
    for (uint32_t position = 1; position < num_experts; ++position) {
        const uint32_t expert = order[position];
        uint32_t slot = position;
        while (slot > 0 && num_blocks[order[slot - 1]] < num_blocks[expert]) {
            order[slot] = order[slot - 1];
            --slot;
        }
        order[slot] = expert;
    }
}

// The tail split of a one-expert call. Of its blocks, the first full = (blocks / lanes) * lanes are dealt
// round-robin; the lanes are spread over the remaining tail blocks as evenly as they go (tail block t takes lanes
// [ceil(t * lanes / tail), ceil((t + 1) * lanes / tail))), and each takes a contiguous run of its block's subblocks,
// [position * subblocks_max / lanes_on_block, (position + 1) * subblocks_max / lanes_on_block), subblocks_max a block
// (the widest share's). A share with fewer subblocks skips the subblocks past its own.
constexpr uint32_t kNoUnit = 0xFFFFFFFF;

struct LaneWork {
    uint32_t full_count;           // full blocks: lane, lane + lanes, ...
    uint32_t tail_block;           // the tail block the lane takes subblocks of, or kNoUnit
    uint32_t tail_subblock_begin;  // the lane's subblocks of it, [begin, end), or kNoUnit for none
    uint32_t tail_subblock_end;
};

inline LaneWork lane_work(uint32_t blocks, uint32_t lanes, uint32_t lane, uint32_t subblocks_max) {
    const uint32_t full = blocks / lanes * lanes;
    const uint32_t tail = blocks - full;
    if (tail == 0) {
        return {full / lanes, kNoUnit, kNoUnit, kNoUnit};
    }
    const uint32_t block = lane * tail / lanes;
    const uint32_t first_lane = (block * lanes + tail - 1) / tail;
    const uint32_t lanes_on_block = ((block + 1) * lanes + tail - 1) / tail - first_lane;
    const uint32_t position = lane - first_lane;
    const uint32_t begin = position * subblocks_max / lanes_on_block;
    const uint32_t end = (position + 1) * subblocks_max / lanes_on_block;
    if (begin == end) {
        return {full / lanes, kNoUnit, kNoUnit, kNoUnit};
    }
    return {full / lanes, full + block, begin, end};
}

}  // namespace grouped_matmul
