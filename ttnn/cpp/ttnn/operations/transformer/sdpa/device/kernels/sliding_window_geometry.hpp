// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

// Number of diagonal edge tiles a sliding window contributes to the lightweight mask CB:
// trailing_primary, leading_prev, leading_current, trailing_next.
constexpr uint32_t kSlidingWindowEdgeTiles = 4;

/**
 * Shared sliding-window K-loop bound geometry.
 *
 * The reader and the streaming compute kernel both narrow the per-Q K-chunk loop
 * to the window. They MUST compute identical bounds or the lightweight masks and
 * output positions desync. This is the single source for that math.
 *
 * Token counts:
 *   - causal window:          left = window - 1 tokens behind the diagonal, no right reach.
 *   - non-causal centered:    left = right = window / 2 tokens around the Q position.
 *
 * Tile counts use ceil division so the loop bound never clips a partially-covered K tile.
 *
 * NOTE: these are *loop-bound* tiles (ceil). They are intentionally distinct from the
 * per-row stamp geometry in apply_lightweight_mask_streaming / generate_lightweight_mask_tiles,
 * which uses floor division + remainder and (for causal) a `window` rather than `window - 1`
 * reach. Do not unify the two — they answer different questions.
 *
 * @tparam sliding_window_size  Window size in tokens (0 = no sliding window).
 * @tparam is_causal            Causal (left-only) vs non-causal centered window.
 * @tparam tile_height          TILE_HEIGHT for the current kernel domain (compute vs dataflow
 *                              expose it under different symbols; both equal 32).
 */
template <uint32_t sliding_window_size, bool is_causal, uint32_t tile_height>
struct SlidingWindowLoopGeometry {
    static constexpr bool has_sliding_window = sliding_window_size > 0;
    static constexpr uint32_t half_window = sliding_window_size / 2;
    static constexpr uint32_t left_window_tokens =
        has_sliding_window ? (is_causal ? (sliding_window_size - 1) : half_window) : 0;
    static constexpr uint32_t right_window_tokens = (has_sliding_window && !is_causal) ? half_window : 0;
    static constexpr uint32_t left_window_tiles = (left_window_tokens + tile_height - 1) / tile_height;
    static constexpr uint32_t right_window_tiles = (right_window_tokens + tile_height - 1) / tile_height;
};
