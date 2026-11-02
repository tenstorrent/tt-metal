// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// fill_v2: K-block schedule. FV_KSIZES (comma list) = explicit per-block K sizes, each <= k_block (the slot
// size; slot strides stay k_block); otherwise k_schedule's ramp (FV_RAMP0 / FV_RAMP1, default uniform).
#pragma once

#include <cstdint>
#include "k_schedule.hpp"

#ifndef FV_RAMP0
#define FV_RAMP0 0
#endif
#ifndef FV_RAMP1
#define FV_RAMP1 0
#endif

namespace fv_schedule {

#ifdef FV_KSIZES
constexpr uint32_t sizes[] = {FV_KSIZES};
template <uint32_t k_block, uint32_t k_total>
struct Schedule {
    static constexpr uint32_t num_blocks = sizeof(sizes) / sizeof(sizes[0]);
    static constexpr uint32_t first(uint32_t block) {
        uint32_t sum = 0;
        for (uint32_t b = 0; b < block; ++b) {
            sum += sizes[b];
        }
        return sum;
    }
    static constexpr uint32_t size(uint32_t block) { return sizes[block]; }
    static_assert(first(num_blocks) == k_total);
};
#else
template <uint32_t k_block, uint32_t k_total>
using Schedule = k_schedule::Schedule<k_block, k_total, FV_RAMP0, FV_RAMP1>;
#endif

}  // namespace fv_schedule
