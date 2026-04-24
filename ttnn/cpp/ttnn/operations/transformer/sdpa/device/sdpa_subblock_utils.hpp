// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstdint>
#include <utility>

#include "kernels/sdpa_streaming_qktv.hpp"

namespace ttnn::prim::detail {

// Determine the largest subblock size for matmul that:
// 1. Has subblock_h * subblock_w <= dst_size
// 2. subblock_h divides block_height and subblock_w divides block_width
// 3. subblock_h <= max_subblock_h (omit or pass UINT32_MAX to leave unconstrained)
//
// Candidates are ordered by total volume (h*w) descending to maximize
// dest register utilization.
static inline std::pair<uint32_t, uint32_t> determine_largest_subblock_size(
    uint32_t block_height, uint32_t block_width, uint32_t dst_size, uint32_t max_subblock_h = UINT32_MAX) {
    constexpr std::array<std::pair<uint32_t, uint32_t>, 20> subblocks = {{
        {2, 4}, {4, 2}, {1, 8}, {8, 1}, {1, 7}, {7, 1}, {2, 3}, {3, 2}, {1, 6}, {6, 1},
        {1, 5}, {5, 1}, {2, 2}, {1, 4}, {4, 1}, {1, 3}, {3, 1}, {1, 2}, {2, 1}, {1, 1},
    }};

    for (auto [subblock_height, subblock_width] : subblocks) {
        if (subblock_height * subblock_width > dst_size) {
            continue;
        }
        if (subblock_height > max_subblock_h) {
            continue;
        }
        if ((block_height % subblock_height != 0) || (block_width % subblock_width != 0)) {
            continue;
        }
        return {subblock_height, subblock_width};
    }
    return {1, 1};
}

// Find the largest granularity value that:
// 1. Is <= max_granularity (typically dst_size or dst_size/2)
// 2. Evenly divides tile_count (so no tiles are dropped in the kernel loop)
//
// This is critical for correctness: the kernel processes dst_tiles per iteration
// for granularity iterations, totaling dst_tiles * granularity tiles. If the
// granularity doesn't divide tile_count, trailing tiles are silently dropped.
static inline uint32_t find_valid_granularity(uint32_t tile_count, uint32_t max_granularity) {
    uint32_t granularity = std::min(tile_count, max_granularity);
    while (granularity > 1 && tile_count % granularity != 0) {
        granularity--;
    }
    return granularity;
}

// Streaming cb_out sizing: 2-slot ping-pong of matmul row-groups (pending SALAD + matmul
// in-flight). qktv_h comes from the shared helper in sdpa_streaming_qktv.hpp so host and
// kernel can't drift — see that header for the formula and rationale.
static inline uint32_t streaming_cb_out_tiles(
    uint32_t out_out_subblock_h, uint32_t out_out_subblock_w, uint32_t dst_size, uint32_t Sq_chunk_t, uint32_t vDHt) {
    const uint32_t qktv_h =
        ttnn::transformer::sdpa::streaming_qktv_h(out_out_subblock_h, out_out_subblock_w, dst_size, Sq_chunk_t);
    return 2u * qktv_h * vDHt;
}

}  // namespace ttnn::prim::detail
