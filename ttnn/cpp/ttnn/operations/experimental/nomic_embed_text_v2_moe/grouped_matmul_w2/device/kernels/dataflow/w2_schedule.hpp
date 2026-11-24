// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The walk every kernel of grouped_matmul_w2 derives from expert_counts: the regions of expert_schedule.hpp, walked
// in the alternating order below, and the weight pieces of each slice. Kernels that disagree hang the device.

#pragma once

#include <cstdint>

#include "ttnn/operations/experimental/nomic_embed_text_v2_moe/grouped_matmul_w1/device/kernels/dataflow/expert_schedule.hpp"

namespace grouped_matmul_w2 {

// The experts with blocks, largest first, alternating with the smallest from the other end, the smallest kept for
// last; the experts without blocks after them. Every expert's slice streams while another computes, and a ring of
// two slots stalls the stream whenever the expert computing is shorter than the next slice takes to arrive: with
// most blocks first those short experts come one after another at the end, alternating each sits behind a long one.
template <uint32_t num_experts>
void expert_order(const uint32_t (&num_blocks)[num_experts], uint32_t (&order)[num_experts]) {
    uint32_t ranked[num_experts];
    grouped_matmul::expert_order<num_experts>(num_blocks, ranked);
    uint32_t used = 0;
    while (used < num_experts && num_blocks[ranked[used]] > 0) {
        ++used;
    }
    uint32_t low = 0;
    uint32_t high = used > 1 ? used - 2 : 0;
    for (uint32_t position = 0; position + 1 < used; ++position) {
        order[position] = ranked[position % 2 == 0 ? low++ : high--];
    }
    for (uint32_t index = used > 0 ? used - 1 : 0; index < num_experts; ++index) {
        order[index] = ranked[index];
    }
}

// The regions and walk order, from the counts page. Blocks go to lanes by region index (lane_blocks), as H's
// lane-local layout places them.
template <uint32_t num_experts>
struct Walk {
    uint32_t first_block[num_experts];
    uint32_t num_blocks[num_experts];
    uint32_t order[num_experts];
    uint32_t used_blocks;

    Walk(volatile const uint32_t* counts, uint32_t input_blocks) {
        used_blocks = grouped_matmul::expert_regions<num_experts>(counts, input_blocks, first_block, num_blocks);
        expert_order<num_experts>(num_blocks, order);
    }

    // The blocks of the expert at `position` that `lane` multiplies.
    template <uint32_t num_lanes>
    grouped_matmul::LaneBlocks lane_blocks(uint32_t position, uint32_t lane) const {
        const uint32_t expert = order[position];
        return grouped_matmul::lane_blocks<num_lanes, false>(expert, first_block[expert], num_blocks[expert], lane);
    }
};

// A lane's blocks in the walk order, one at a time.
template <uint32_t num_experts, uint32_t num_lanes>
struct LaneWalk {
    const Walk<num_experts>& walk;
    uint32_t lane;
    uint32_t position = 0;
    uint32_t step = 0;

    bool next(uint32_t& block) {
        while (position < num_experts) {
            const auto blocks = walk.template lane_blocks<num_lanes>(position, lane);
            if (step < blocks.count) {
                block = blocks.first + step++ * blocks.stride;
                return true;
            }
            ++position;
            step = 0;
        }
        return false;
    }
};

}  // namespace grouped_matmul_w2
