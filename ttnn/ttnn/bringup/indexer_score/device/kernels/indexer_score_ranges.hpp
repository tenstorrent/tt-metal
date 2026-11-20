// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Bring-up fork (DeepSeek-V4.1 PB.7C, operation_attributes_t::lane_ranges): a device's query-row ranges, read by the
// reader, compute and writer kernels from their common runtime args at a base the factory passes as the named
// compile-time arg "ranges_base" (only under the INDEXER_SCORE_RANGES define). Pure integer code: host and kernels.
//
// Table (kTableWords words): [n, (q_tile_lo, q_tile_hi, buffer, chunk_start_tiles, kv_len_tiles) x kRangesMax]; the
// reader's table is followed by the gathered-key buffer addresses [kBuffersMax] and the local-key buffer addresses
// [kBuffersMax]. chunk_start_tiles is the diagonal tile of q-tile-row 0 of the device (mod 2^32: a range starting at
// position p at row r has p / 32 - r / 32; causal_diag_tile adds the row back), so a range's rows see exactly the
// geometry of a natural_query_lane call starting at that range's first row.

#include <cstdint>

namespace indexer_ranges {

inline constexpr uint32_t kRangesMax = 8;
inline constexpr uint32_t kRangeWords = 5;
inline constexpr uint32_t kTableWords = 1 + kRangesMax * kRangeWords;
inline constexpr uint32_t kBuffersMax = 8;

struct Range {
    uint32_t buffer;
    uint32_t chunk_start_tiles;
    uint32_t kv_len_tiles;
};

// The range holding q-tile-row `q_tile` (kv_len 0 when none: the group's cells are all empty).
template <typename Get>
inline Range find(Get get, uint32_t base, uint32_t q_tile) {
    const uint32_t n = get(base);
    for (uint32_t e = 0; e < n && e < kRangesMax; ++e) {
        const uint32_t w = base + 1 + e * kRangeWords;
        if (q_tile >= get(w) && q_tile < get(w + 1)) {
            return {get(w + 2), get(w + 3), get(w + 4)};
        }
    }
    return {0u, 0u, 0u};
}

}  // namespace indexer_ranges
