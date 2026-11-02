// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Ragged K-block schedule: up to two small leading "ramp" K-blocks (ramp0, ramp1; 0 = absent), then
// uniform k_block blocks, the last one ragged. Every block occupies a full k_block slot in the CBs (slot
// strides stay k_block); only the first size(b) K tiles of a slot are valid. Shrinks the pipeline fill:
// compute starts on ramp0 K tiles instead of k_block.
#pragma once

#include <cstdint>

namespace k_schedule {

template <uint32_t k_block, uint32_t k_total, uint32_t ramp0, uint32_t ramp1>
struct Schedule {
    static_assert(ramp0 <= k_block && ramp1 <= k_block);
    static_assert(ramp1 == 0 || ramp0 != 0);
    static constexpr uint32_t ramp_len = (ramp0 ? 1u : 0u) + (ramp1 ? 1u : 0u);
    static constexpr uint32_t ramp_sum = ramp0 + ramp1;
    static_assert(ramp_sum < k_total);
    static constexpr uint32_t num_blocks = ramp_len + (k_total - ramp_sum + k_block - 1) / k_block;

    static constexpr uint32_t first(uint32_t block) {
        if (block < ramp_len) {
            return block == 0 ? 0 : ramp0;
        }
        return ramp_sum + (block - ramp_len) * k_block;
    }
    static constexpr uint32_t size(uint32_t block) {
        if (block < ramp_len) {
            return block == 0 ? ramp0 : ramp1;
        }
        const uint32_t rest = k_total - first(block);
        return rest < k_block ? rest : k_block;
    }
};

}  // namespace k_schedule
