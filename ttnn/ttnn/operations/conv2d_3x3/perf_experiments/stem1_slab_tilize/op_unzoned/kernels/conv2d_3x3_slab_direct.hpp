// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Slab-direct face im2col (the descriptor's SLAB_DIRECT; face-direct shapes, stride * C_in_p == 16, whole-K, W_out a
// multiple of 32). Pure integer arithmetic shared by the reader / writer (the padded fill) and compute (in0 addresses);
// no dataflow or compute API.
//
// A face-direct im2col row of kernel row kh is one contiguous run of the input row (the K padding of C_in 8 reads the
// next pixel, against zero weight rows), and one output pixel steps it by exactly one 16-element face row. So the
// 32 rows of K columns [16 h, 16 h + 16) of a tile-row (32 output pixels of one output row) are ONE contiguous 1 KB
// run of the input row, and columns [16 h + 16, 16 h + 32) the same run HALF_BYTES further. Compute unpacks every
// in0 tile straight from the slab as those two runs (conv2d_3x3_compute.cpp, slab_direct): nothing is gathered.
//
// The slab is padded: slot r holds image row ih_lo + r behind GAP (= padding) zero pixels, PITCH = GAP + W pixels per
// slot, so a row's left padding is its own gap and its right padding the next slot's gap. Every slot also starts with
// ZONE zero pixels written once at kernel start (the gap plus the pixels a row's last im2col runs read past its end,
// zero weight rows only): the fill never writes a gap, and a slot whose row is outside the image keeps its zeros.
// Kernel rows outside the image read the zero block instead.

#pragma once

#include <cstdint>

namespace conv2d_3x3::slab_direct {

template <
    int32_t in_h,
    int32_t in_w,
    uint32_t out_w,
    int32_t stride,
    int32_t padding,
    int32_t dilation,
    int32_t gap,
    int32_t pitch,
    uint32_t pixel_bytes,
    uint32_t half_bytes>
struct Geom {
    // Image row of slot 0 for the M-block whose first output pixel is o_first.
    static inline __attribute__((always_inline)) int32_t row_lo(uint32_t o_first) {
        return stride * static_cast<int32_t>(o_first / out_w) - padding;
    }

    // In-image pixels of the block's three bands (kernel row kh: from tap (kh, 0) of o_first to tap
    // (kh, band_last_tap) of o_end - 1), each once, split at image rows: fn(page_lo, page_hi, slab_byte_offset).
    // Only rows of the block's slots [ih_lo, ih_last]: a band's flat range also reaches into the row before the
    // first slot (the left padding tap of o_first) and the row after the last (the taps past a row's end), which in
    // the padded layout are gap pixels, not image pixels.
    template <int32_t band_last_tap, typename Fn>
    static inline __attribute__((always_inline)) void fill_pieces(uint32_t o_first, uint32_t o_end, Fn&& fn) {
        const int32_t ho_first = o_first / out_w;
        const int32_t wo_first = o_first % out_w;
        const int32_t ho_last = (o_end - 1) / out_w;
        const int32_t wo_last = (o_end - 1) % out_w;
        const int32_t f = (ho_first * stride - padding) * in_w + wo_first * stride - padding;
        const int32_t band_len =
            (ho_last - ho_first) * stride * in_w + (wo_last - wo_first) * stride + band_last_tap * dilation + 1;
        const int32_t ih_lo = ho_first * stride - padding;
        const int32_t ih_last = ho_last * stride - padding + 2 * dilation;
        const int32_t slot_lo = (ih_lo > 0 ? ih_lo : 0) * in_w;
        const int32_t slot_hi = (ih_last + 1 < in_h ? ih_last + 1 : in_h) * in_w;
        int32_t prev_end = -0x7FFFFFFF;
        for (int32_t kh = 0; kh < 3; ++kh) {
            int32_t lo = f + kh * dilation * in_w;
            const int32_t hi_raw = lo + band_len;
            lo = lo > prev_end ? lo : prev_end;
            prev_end = hi_raw > prev_end ? hi_raw : prev_end;
            lo = lo > slot_lo ? lo : slot_lo;
            const int32_t hi = hi_raw < slot_hi ? hi_raw : slot_hi;
            for (int32_t v = lo; v < hi;) {
                const int32_t ih = v / in_w;
                const int32_t row_end = (ih + 1) * in_w;
                const int32_t e = hi < row_end ? hi : row_end;
                fn(static_cast<uint32_t>(v),
                   static_cast<uint32_t>(e),
                   static_cast<uint32_t>(((ih - ih_lo) * pitch + gap + (v - ih * in_w)) * pixel_bytes));
                v = e;
            }
        }
    }

    // L1 byte address of the first run (K columns [32 j, 32 j + 16)) of in0 K-tile j of kernel row kh for the tile-row
    // whose first output pixel is o; the second run is half_bytes further. zero: the zero block.
    static inline __attribute__((always_inline)) uint32_t
    in0_addr(uint32_t slab, uint32_t zero, int32_t ih_lo, uint32_t o, int32_t kh, uint32_t j) {
        const int32_t oh = o / out_w;
        const int32_t ow0 = o % out_w;
        const int32_t ih = stride * oh + kh * dilation - padding;
        if (ih < 0 || ih >= in_h) {
            return zero;
        }
        return slab + static_cast<uint32_t>(((ih - ih_lo) * pitch + gap + stride * ow0 - padding) * pixel_bytes) +
               j * 2 * half_bytes;
    }
};

// Fused bias (SD_FUSED_BIAS): compute adds the bias as one more K-step, in0 = the ones block (1.0 in K columns 0 and 1
// of every row) times in1 = the core's prepared bias tile (row 0 = b, every other row its zero padding). From HiFi2 on
// the multiplier takes 9 significant bits of in1, so 1.0 x b is exact. LoFi takes 5 (the hidden bit + 4 MSBs,
// truncated): 1.0 x b gives b_hi = b truncated to 5 significant bits, so the writer stores the rest, b_lo = b - b_hi
// (at most 2 significant bits on bf8_b, 3 on bf16, exact in 5), in row 1 (SD_BIAS_SPLIT) and DEST receives exactly
// b_hi + b_lo = b. Rows 16 and 17, which the ones block's second run reads, stay the tile's zero padding.
// bf8_b tile: 64 exponent bytes (face f row r at 16 f + r), then 1024 sign-magnitude bytes, face-major, value
// (-1)^s (m & 127) 2^(e - 133). bf16 tile: faces of 256 bf16.

// Four sign-magnitude bytes at once: a magnitude with its leading one at bit 6 (5) keeps its low 2 (1) bits in lo.
inline __attribute__((always_inline)) uint32_t lo_bf8x4(uint32_t v) {
    const uint32_t b6 = (v >> 6) & 0x01010101u;
    const uint32_t b5 = (v >> 5) & 0x01010101u;
    const uint32_t lomask = b6 | (b6 << 1) | (b5 & ~b6);
    const uint32_t lo_mag = v & lomask;
    const uint32_t nz = (lo_mag | (lo_mag >> 1)) & 0x01010101u;
    return lo_mag | (v & (nz << 7));
}

// Row 1 of faces 0 and 1 = b_lo, with row 0's shared exponent.
inline __attribute__((always_inline)) void split_bias_tile_bf8(uint32_t tile) {
    volatile uint32_t* w = reinterpret_cast<volatile uint32_t*>(tile);
    for (uint32_t f = 0; f < 2; ++f) {
        volatile uint32_t* row0 = w + (64 + 256 * f) / 4;
        for (uint32_t q = 0; q < 4; ++q) {
            row0[4 + q] = lo_bf8x4(row0[q]);
        }
        const uint32_t e = w[4 * f];
        w[4 * f] = (e & 0xffff00ffu) | ((e & 0xffu) << 8);
    }
}

// bf16: b_lo = the low 3 explicit mantissa bits, renormalized (0 if they are 0, for a zero / denormal / inf / NaN b,
// or if b_lo would be denormal). lead[bits] = position of the leading one of the low 3 bits.
inline __attribute__((always_inline)) uint32_t lo_bf16(uint32_t b) {
    const uint32_t bits = b & 7;
    const uint32_t e = (b >> 7) & 0xff;
    const uint32_t lead = (0xAA50u >> (2 * bits)) & 3;  // {0, 0, 1, 1, 2, 2, 2, 2}
    const int32_t le = static_cast<int32_t>(e) - 7 + static_cast<int32_t>(lead);
    if (bits == 0 || e == 0 || e == 0xff || le < 1) {
        return 0;
    }
    return (b & 0x8000) | (static_cast<uint32_t>(le) << 7) | ((bits << (7 - lead)) & 0x7f);
}

inline __attribute__((always_inline)) void split_bias_tile_bf16(uint32_t tile) {
    volatile uint32_t* w = reinterpret_cast<volatile uint32_t*>(tile);
    for (uint32_t f = 0; f < 2; ++f) {
        volatile uint32_t* row0 = w + 128 * f;  // face f row 0: 16 bf16 = 8 words; row 1 at +8
        for (uint32_t q = 0; q < 8; ++q) {
            const uint32_t v = row0[q];
            row0[8 + q] = lo_bf16(v & 0xffff) | (lo_bf16(v >> 16) << 16);
        }
    }
}

}  // namespace conv2d_3x3::slab_direct
