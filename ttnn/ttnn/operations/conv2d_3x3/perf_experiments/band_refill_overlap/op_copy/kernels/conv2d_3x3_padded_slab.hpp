// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Padded slab of the tap-direct N-slice prelude (the descriptor's PADDED_SLAB). Pure integer arithmetic, shared by the
// reader (gap zeroing) and compute (untilize destinations, tilize sources); no dataflow or compute API.
//
// The fill streams the same input tile-rows in the same order as the TILE slab plan (tile_slab_plan in
// conv2d_3x3_slab_dataflow.hpp; this file mirrors its band / segment walk). Compute untilizes each image row the slab
// holds into its own row slot instead of packing the tile-rows back to back: slot r is S = W + g pixels, g = d zero
// pixels then the W pixels of the row, and one more g-pixel gap follows the last slot. A row's left padding (w < 0) is
// then the g zeros in front of it and its right padding (w >= W) the gap of the next slot, so with stride 1 and
// p == d every tap block of a tile-row [32 pixels, c_in_p] inside the image row is ONE contiguous run of the slab:
// kw = 0 at w0 = 0 starts in the gap, kw = 2 at w0 = W - 32 ends in the next one. Nothing is gathered or copied;
// a tap whose kernel row lies outside the image is the 32-pixel zero block. The gaps are never written by the untilize,
// so the reader zeroes them once (slots_max + 1 gaps at fixed offsets).
//
// Slots: image rows are ranked in fill order (the fill's tile-rows increase strictly, so the rows do too); a segment
// that starts in the image row the previous segment ended in shares that row's slot. Every band's rows are contiguous,
// so band kh maps image row h to slot h + band_row_off[kh]. Tile-rows past the plan (a last batch's unread tail) are
// not untilized.

#pragma once

#include <cstdint>

namespace conv2d_3x3::padded_slab {

struct Plan {
    uint32_t num_segs;
    uint32_t seg_first[3];    // first input tile-row of segment s
    uint32_t seg_count[3];    // its tile-rows
    int32_t seg_row_off[3];   // slot of image row h in segment s: h + seg_row_off[s]
    int32_t band_row_off[3];  // slot of image row h of band kh (an in-image band): h + band_row_off[kh]
    uint32_t rows;            // tile-rows the plan streams (the fill's slab rows before the unread tail)
    uint32_t slots;           // row slots used
};

// Plan of output pixels [o_first, o_end) (stride 1, the non-face-direct band, as block_footprint / tile_slab_plan).
template <uint32_t output_width, int32_t input_width, int32_t padding, int32_t dilation, int32_t num_in_px>
inline __attribute__((always_inline)) void plan(uint32_t o_first, uint32_t o_end, Plan& p) {
    constexpr int32_t TILE_DIM = 32;
    // Compiled on every path (W < 32 must not divide by zero); padded slabs only run with W % 32 == 0.
    constexpr int32_t tpr = input_width >= TILE_DIM ? input_width / TILE_DIM : 1;
    const int32_t ho_first = o_first / output_width;
    const int32_t wo_first = o_first % output_width;
    const int32_t ho_last = (o_end - 1) / output_width;
    const int32_t wo_last = (o_end - 1) % output_width;
    const int32_t f = (ho_first - padding) * input_width + wo_first - padding;
    const int32_t band_len = (ho_last - ho_first) * input_width + (wo_last - wo_first) + 2 * dilation + 1;
    p.num_segs = 0;
    p.rows = 0;
    bool have_prev = false;
    int32_t prev_t_hi = 0;
    int32_t prev_band_off = 0;
    int32_t last_row = -1;
    int32_t last_slot = -1;
    for (int32_t kh = 0; kh < 3; ++kh) {
        const int32_t v_lo = f + kh * dilation * input_width;
        const int32_t lo = v_lo > 0 ? v_lo : 0;
        const int32_t hi = v_lo + band_len < num_in_px ? v_lo + band_len : num_in_px;
        p.band_row_off[kh] = 0;
        if (lo >= hi) {
            continue;
        }
        const int32_t t_lo = lo / TILE_DIM;
        const int32_t t_hi = (hi + TILE_DIM - 1) / TILE_DIM;
        const bool shares = have_prev && t_lo < prev_t_hi;
        const int32_t first_new = shares ? prev_t_hi : t_lo;
        int32_t seg_off = prev_band_off;
        if (t_hi > first_new) {
            const int32_t r0 = first_new / tpr;
            const int32_t r1 = (t_hi - 1) / tpr;
            const int32_t s0 = (last_row >= 0 && r0 == last_row) ? last_slot : last_slot + 1;
            seg_off = s0 - r0;
            p.seg_first[p.num_segs] = first_new;
            p.seg_count[p.num_segs] = t_hi - first_new;
            p.seg_row_off[p.num_segs] = seg_off;
            ++p.num_segs;
            p.rows += t_hi - first_new;
            last_row = r1;
            last_slot = s0 + (r1 - r0);
        }
        p.band_row_off[kh] = shares ? prev_band_off : seg_off;
        prev_band_off = p.band_row_off[kh];
        prev_t_hi = t_hi;
        have_prev = true;
    }
    p.slots = static_cast<uint32_t>(last_slot + 1);
}

// Slab pixel of the k-th tile-row the fill streams (row 0 of that tile-row), k < p.rows.
template <int32_t input_width, int32_t gap, uint32_t slots_max>
inline __attribute__((always_inline)) uint32_t tile_row_px(const Plan& p, uint32_t k) {
    constexpr int32_t TILE_DIM = 32;
    constexpr int32_t tpr = input_width >= TILE_DIM ? input_width / TILE_DIM : 1;
    constexpr int32_t stride = input_width + gap;
    uint32_t cum = 0;
    for (uint32_t s = 0; s < p.num_segs; ++s) {
        if (k < cum + p.seg_count[s]) {
            const int32_t i = static_cast<int32_t>(p.seg_first[s] + (k - cum));
            const int32_t h = i / tpr;
            return static_cast<uint32_t>((h + p.seg_row_off[s]) * stride + gap + (i - h * tpr) * TILE_DIM);
        }
        cum += p.seg_count[s];
    }
    return slots_max * stride;  // not reached for k < p.rows
}

}  // namespace conv2d_3x3::padded_slab
