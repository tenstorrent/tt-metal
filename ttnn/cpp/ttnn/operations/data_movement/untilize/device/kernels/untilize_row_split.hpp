// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cstdint>
#include <utility>

// Per-core tile-row range of the interleaved multi-core untilize, derived on device from the common
// row count; shared by reader, compute and writer. The first `rows % cores` cores take one extra row
// so the work is balanced. Cores past the end of a short input get an empty range.
inline std::pair<uint32_t, uint32_t> untilize_row_range(uint32_t core_idx, uint32_t num_tile_rows, uint32_t num_cores) {
    if (num_cores == 0 || core_idx >= num_cores) {
        return {0, 0};
    }
    const uint32_t base = num_tile_rows / num_cores;
    const uint32_t extra = num_tile_rows % num_cores;
    const uint32_t start = core_idx * base + std::min(core_idx, extra);
    const uint32_t count = base + (core_idx < extra ? 1u : 0u);
    return {start, start + count};
}
