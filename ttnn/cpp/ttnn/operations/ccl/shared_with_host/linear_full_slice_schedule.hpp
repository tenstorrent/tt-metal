// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>

namespace ttnn::ccl {
struct LinearFullSlice {
    uint32_t source;
    uint32_t direction;
    uint32_t threshold;
};

// Local first, then increasing distance, lower rank before higher. No wraparound.
// dir0 publishes r-1,r-2,...; dir1 publishes local,r+1,r+2,... .
constexpr LinearFullSlice linear_full_slice(uint32_t size, uint32_t rank, uint32_t ordinal) {
    if (ordinal == 0) {
        return {rank, 1, 1};
    }
    uint32_t index = 1;
    for (uint32_t distance = 1; distance < size; ++distance) {
        if (distance <= rank && index++ == ordinal) {
            return {rank - distance, 0, distance};
        }
        if (rank + distance < size && index++ == ordinal) {
            return {rank + distance, 1, distance + 1};
        }
    }
    return {size, 0, 0};  // Invalid ordinal; host/device callers validate the bound.
}
}  // namespace ttnn::ccl
