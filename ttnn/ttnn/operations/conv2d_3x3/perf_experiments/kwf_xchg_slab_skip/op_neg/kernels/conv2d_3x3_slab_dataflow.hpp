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

// Ring slab (TILE, the descriptor's ring_capable): the in-image input tile-rows [lo, hi) an M-block's three bands
// touch, and each band's last row + 1 (band_hi, 0 for a band outside the image). The ring holds consecutive input
// tile-rows across M-blocks; the descriptor's ring_plan mirrors this.
struct RingWindow {
    int32_t lo;
    int32_t hi;
    int32_t band_hi[3];
};

template <int32_t input_width, int32_t num_in_px, int32_t dilation>
FORCE_INLINE RingWindow ring_window(const SlabFootprint& fp) {
    constexpr int32_t TILE_DIM = 32;
    RingWindow w{0x7FFFFFFF, 0, {0, 0, 0}};
    for (int32_t kh = 0; kh < 3; ++kh) {
        const int32_t v_lo = fp.f + kh * dilation * input_width;
        const int32_t lo = v_lo > 0 ? v_lo : 0;
        const int32_t hi = v_lo + fp.band_len < num_in_px ? v_lo + fp.band_len : num_in_px;
        if (lo >= hi) {
            continue;
        }
        const int32_t t_lo = lo / TILE_DIM;
        const int32_t t_hi = (hi + TILE_DIM - 1) / TILE_DIM;
        w.lo = t_lo < w.lo ? t_lo : w.lo;
        w.hi = t_hi > w.hi ? t_hi : w.hi;
        w.band_hi[kh] = t_hi;
    }
    return w;
}

// Ring slab: the lowest window row of M-blocks [m_first, num_blocks) of a core, i.e. the rows the ring must still hold
// (a later block's window may start lower than an earlier one's, e.g. at the image top once band 0 enters the image).
template <
    uint32_t output_width,
    int32_t input_width,
    int32_t stride,
    int32_t padding,
    int32_t dilation,
    int32_t band_last_tap,
    int32_t num_in_px,
    uint32_t num_out_px,
    uint32_t block_m_tiles>
FORCE_INLINE int32_t
ring_keep_lo(uint32_t m_tile_start, uint32_t actual_m_tiles, uint32_t m_first, uint32_t num_blocks) {
    // A block's window starts at or above its first tap's tile-row (clamped at 0), which never decreases from block to
    // block: once that bound reaches lo, no later block can lower it.
    int32_t lo = 0x7FFFFFFF;
    for (uint32_t m = m_first; m < num_blocks; ++m) {
        const uint32_t done = m * block_m_tiles;
        const uint32_t valid = actual_m_tiles - done < block_m_tiles ? actual_m_tiles - done : block_m_tiles;
        const uint32_t o_first = (m_tile_start + done) * 32;
        const uint32_t o_end_tiles = o_first + valid * 32;
        const SlabFootprint fp = block_footprint<output_width, input_width, stride, padding, dilation, band_last_tap>(
            o_first, o_end_tiles < num_out_px ? o_end_tiles : num_out_px);
        if ((fp.f > 0 ? fp.f : 0) / 32 >= lo) {
            break;
        }
        const RingWindow w = ring_window<input_width, num_in_px, dilation>(fp);
        lo = w.lo < lo ? w.lo : lo;
    }
    return lo;
}

// The input tile-rows of slab rows [batch * batch_rows, + batch_rows) in slab order: fn(row0, rows, batch_row) per
// segment piece, input tile-rows [row0, row0 + rows) landing at row batch_row of the batch. Slab rows past the plan's
// last segment are unread.
template <uint32_t batch_rows, typename Fn>
FORCE_INLINE void tile_batch_rows(const TileSlabPlan& plan, uint32_t batch, Fn&& fn) {
    const uint32_t b_lo = batch * batch_rows;
    const uint32_t b_hi = b_lo + batch_rows;
    uint32_t seg_lo = 0;
    for (uint32_t s = 0; s < plan.num_segs; ++s) {
        const uint32_t seg_hi = seg_lo + plan.seg_count[s];
        const uint32_t lo = b_lo > seg_lo ? b_lo : seg_lo;
        const uint32_t hi = b_hi < seg_hi ? b_hi : seg_hi;
        if (lo < hi) {
            fn(plan.seg_first[s] + (lo - seg_lo), hi - lo, lo - b_lo);
        }
        seg_lo = seg_hi;
    }
}

// Split TILE fill: the slab rows holding input tile-rows at or past hi_row (the higher-index owner shards') are a
// suffix of the plan, [first, rows): the plan's segments hold strictly increasing input tile-rows in slab order (band
// kh + 1 starts at or past band kh's rows, and a band that shares rows adds only those past the previous band's end).
// One O(segments) pass per plan replaces a walk of every batch's pieces, which cost the reader ~0.4 us of hand-over on
// res2 (7 batches) and the writer ~0.5 us of planning between that hand-over and its first read.
struct SlabSuffix {
    uint32_t first;  // first slab row holding an input tile-row >= hi_row (== rows when there is none)
    uint32_t rows;   // slab rows the plan reads
};

FORCE_INLINE SlabSuffix slab_rows_from(const TileSlabPlan& plan, uint32_t hi_row) {
    uint32_t seg_lo = 0;
    uint32_t first = 0xFFFFFFFF;
    for (uint32_t s = 0; s < plan.num_segs; ++s) {
        if (first == 0xFFFFFFFF && plan.seg_first[s] + plan.seg_count[s] > hi_row) {
            first = seg_lo + (hi_row > plan.seg_first[s] ? hi_row - plan.seg_first[s] : 0);
        }
        seg_lo += plan.seg_count[s];
    }
    return {first < seg_lo ? first : seg_lo, seg_lo};
}

// Does the batch hold an input tile-row at or past hi_row (a slab row of the suffix)?
template <uint32_t batch_rows>
FORCE_INLINE bool batch_has_rows_from(const SlabSuffix& sfx, uint32_t batch) {
    return batch * batch_rows < sfx.rows && (batch + 1) * batch_rows > sfx.first;
}

// kw-fold (conv2d_3x3_kw_fold.hpp): the input tile-rows compute reads of one M-block. Its walk covers tile-rows
// [lo, hi) = [t0 - ht, t_end + hb), the halo tile-row on a side only where it recomputes it (a block boundary inside
// an image row, without the halo exchange), and reads row t + (kh - 1) R of each: three bands [lo + (kh - 1) R,
// hi + (kh - 1) R). The slab plan rounds each band out by one tile-row on each side (the d-pixel tap reach), so where
// the halo is exchanged or not needed (a boundary on an image-row edge) the fill skips those rows: their slots stay
// unwritten and are never read, and the batch pushes (compute's cumulative slab waits) are unchanged.
struct KwfLiveRows {
    int32_t lo;
    int32_t hi;
    int32_t R;
    // fn(a, b) for each maximal run [a, b) of live rows inside [row0, row_end), ascending.
    template <typename Fn>
    FORCE_INLINE void pieces(int32_t row0, int32_t row_end, Fn&& fn) const {
        int32_t a = lo - R;
        int32_t b = hi - R;
        for (int32_t kh = 1; kh <= 3; ++kh) {
            if (kh < 3 && lo + (kh - 1) * R <= b) {
                b = hi + (kh - 1) * R;  // band kh touches the run: extend it
                continue;
            }
            const int32_t s = a > row0 ? a : row0;
            const int32_t e = b < row_end ? b : row_end;
            if (s < e) {
                fn(s, e);
            }
            if (kh < 3) {
                a = lo + (kh - 1) * R;
                b = hi + (kh - 1) * R;
            }
        }
    }
};

template <int32_t tile_rows_per_image_row, int32_t dilation, int32_t m_tiles, bool xchg>
FORCE_INLINE KwfLiveRows kwf_live_rows(uint32_t t0, uint32_t valid) {
    // kw_fold.hpp run(): halo_top / halo_bot (the exchange runs on one M-block per core and then replaces both).
    const int32_t t_end = static_cast<int32_t>(t0 + valid);
    const bool ht = false;  // NEGATIVE CONTROL
    const bool hb = false;  // NEGATIVE CONTROL
    return {static_cast<int32_t>(t0) - (ht ? 1 : 0), t_end + (hb ? 1 : 0), dilation * tile_rows_per_image_row};
}

// Split fill under kw-fold: the writer's slab rows are the live rows at or past hi_row, from the first one's slab row
// to one past the last live slab row (a batch of dead rows only is not handed over); {0, 0} when there is none.
FORCE_INLINE SlabSuffix kwf_live_suffix(const TileSlabPlan& plan, const KwfLiveRows& live, uint32_t hi_row) {
    uint32_t seg_lo = 0;
    uint32_t first = 0xFFFFFFFF;
    uint32_t end = 0;
    const int32_t hi = static_cast<int32_t>(hi_row);
    for (uint32_t s = 0; s < plan.num_segs; ++s) {
        const int32_t seg_first = static_cast<int32_t>(plan.seg_first[s]);
        live.pieces(hi > seg_first ? hi : seg_first, seg_first + plan.seg_count[s], [&](int32_t a, int32_t b) {
            if (first == 0xFFFFFFFF) {
                first = seg_lo + static_cast<uint32_t>(a - seg_first);
            }
            end = seg_lo + static_cast<uint32_t>(b - seg_first);
        });
        seg_lo += plan.seg_count[s];
    }
    return first == 0xFFFFFFFF ? SlabSuffix{0, 0} : SlabSuffix{first, end};
}

}  // namespace conv2d_3x3
