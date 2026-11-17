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

}  // namespace conv2d_3x3
