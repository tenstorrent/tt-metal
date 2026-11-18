// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// ROW_MAJOR halo slab plan of one M-block, shared by the reader and the writer so both place every input pixel
// at the same slab offset when they split the fill between them.

#pragma once

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

namespace conv2d_3x3 {

struct SlabFootprint {
    int32_t f;         // flat input index of tap (0, 0) of the block's first output pixel; negative in the padding
    int32_t band_len;  // input pixels one kernel row of the block touches
};

// Footprint of output pixels [o_first, o_end): from tap (kh, 0) of o_first to tap (kh, band_last_tap) of the last.
template <
    uint32_t output_width,
    int32_t input_width,
    int32_t stride,
    int32_t padding,
    int32_t dilation,
    int32_t band_last_tap>
FORCE_INLINE SlabFootprint block_footprint(uint32_t o_first, uint32_t o_end) {
    const int32_t ho_first = o_first / output_width;
    const int32_t wo_first = o_first % output_width;
    const int32_t ho_last = (o_end - 1) / output_width;
    const int32_t wo_last = (o_end - 1) % output_width;
    return {
        (ho_first * stride - padding) * input_width + wo_first * stride - padding,
        (ho_last - ho_first) * stride * input_width + (wo_last - wo_first) * stride + band_last_tap * dilation + 1};
}

// Band kh holds flat pixels [f + kh * d * W, + band_len) from slab pixel kh * band_stride_px, so pixel v of band kh
// sits at slab pixel v + slab_x_base[kh]. With band_stride_px == d * W overlapping bands coincide, so a band adds
// only the slab pixels past the previous band's end. Calls band_pixels(kh, x_lo, x_hi, v_minus_x) with the
// in-image slab range [x_lo, x_hi) band kh adds (flat pixel = slab pixel + v_minus_x), in band order; the range
// may be empty.
template <int32_t input_width, int32_t num_in_px, int32_t dilation, int32_t band_stride_px, typename Fn>
FORCE_INLINE void slab_bands(const SlabFootprint& fp, int32_t (&slab_x_base)[3], Fn&& band_pixels) {
    for (int32_t kh = 0; kh < 3; ++kh) {
        const int32_t prev_end = (kh - 1) * band_stride_px + fp.band_len;
        const int32_t seg_lo = kh == 0 ? 0 : (prev_end > kh * band_stride_px ? prev_end : kh * band_stride_px);
        const int32_t seg_hi = kh * band_stride_px + fp.band_len;
        const int32_t v_minus_x = fp.f + kh * dilation * input_width - kh * band_stride_px;
        slab_x_base[kh] = -v_minus_x;
        const int32_t x_lo = seg_lo > -v_minus_x ? seg_lo : -v_minus_x;
        const int32_t x_hi = seg_hi < num_in_px - v_minus_x ? seg_hi : num_in_px - v_minus_x;
        band_pixels(kh, x_lo, x_hi, v_minus_x);
    }
}

// TILE slab plan of one M-block, shared by the reader (which streams it) and the writer's split row gather (which
// needs slab_x_base). Band kh is its in-image pixels rounded out to whole 32-pixel input tile-rows; a band that starts
// inside the previous band's tile-rows shares them, otherwise it is placed after them.
struct TileSlabPlan {
    uint32_t seg_first[3];
    uint32_t seg_count[3];
    uint32_t num_segs;
    uint32_t ready_pages[3];  // slab pages that hold band kh once compute pushed them
};

template <int32_t input_width, int32_t num_in_px, int32_t dilation, uint32_t c_in_tiles>
FORCE_INLINE void tile_slab_plan(const SlabFootprint& fp, int32_t (&slab_x_base)[3], TileSlabPlan& plan) {
    constexpr int32_t TILE_DIM = 32;
    plan.num_segs = 0;
    int32_t next_free = 0;
    bool have_prev = false;
    int32_t prev_t_lo = 0, prev_t_hi = 0, prev_row = 0;
    for (int32_t kh = 0; kh < 3; ++kh) {
        const int32_t v_lo = fp.f + kh * dilation * input_width;
        const int32_t lo = v_lo > 0 ? v_lo : 0;
        const int32_t hi = v_lo + fp.band_len < num_in_px ? v_lo + fp.band_len : num_in_px;
        if (lo >= hi) {
            slab_x_base[kh] = 0;
            plan.ready_pages[kh] = next_free * c_in_tiles;
            continue;
        }
        const int32_t t_lo = lo / TILE_DIM;
        const int32_t t_hi = (hi + TILE_DIM - 1) / TILE_DIM;
        const bool shares = have_prev && t_lo < prev_t_hi;
        const int32_t row = shares ? prev_row + (t_lo - prev_t_lo) : next_free;
        const int32_t first_new = shares ? prev_t_hi : t_lo;
        if (t_hi > first_new) {
            plan.seg_first[plan.num_segs] = first_new;
            plan.seg_count[plan.num_segs] = t_hi - first_new;
            ++plan.num_segs;
        }
        slab_x_base[kh] = TILE_DIM * (row - t_lo);
        next_free = row + (t_hi - t_lo);
        plan.ready_pages[kh] = next_free * c_in_tiles;
        have_prev = true;
        prev_t_lo = t_lo;
        prev_t_hi = t_hi;
        prev_row = row;
    }
}

}  // namespace conv2d_3x3
