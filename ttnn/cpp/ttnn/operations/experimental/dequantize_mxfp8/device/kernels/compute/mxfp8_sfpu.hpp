// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "sfpi.h"
#include "llk_math_eltwise_sfpu_common.h"

namespace ckernel::sfpu {

// Exact E4M3 * 2^(E8M0 - 127), built by integer exponent arithmetic on FP32 bits.
// E4M3 has at most four significant bits, so the product is exact whenever its
// FP32 biased exponent lies in [1, 254]. Larger products become signed infinity;
// smaller ones flush to signed zero (both permitted by the contract). Inputs never
// contain NaN codes (0x7F/0xFF for E4M3, 0xFF for E8M0).
sfpi_inline sfpi::vUInt mxfp8_decode_bits(sfpi::vUInt code, sfpi::vInt scale) {
    const sfpi::vUInt magnitude = code & 0x7F;
    // FP32 bits of the unscaled E4M3 magnitude; normal codes: exponent e + 120.
    sfpi::vUInt base = (magnitude << 20) + (120u << 23);
    v_if(magnitude < 8) {
        // Subnormal E4M3: m * 2^-9 with m in 1..7 (exactly representable); zero stays zero.
        base = sfpi::as<sfpi::vUInt>(sfpi::convert<sfpi::vFloat>(sfpi::as<sfpi::vInt>(magnitude)));
        v_if(magnitude != 0) { base = base - (9u << 23); }
        v_endif;
    }
    v_endif;
    const sfpi::vInt adjust = scale - 127;
    const sfpi::vInt exponent = sfpi::as<sfpi::vInt>(base >> 23) + adjust;
    sfpi::vUInt bits = base + sfpi::as<sfpi::vUInt>(adjust << 23);
    v_if(exponent >= 255) { bits = 0x7F800000u; }
    v_elseif(exponent <= 0 || magnitude == 0) { bits = 0; }
    v_endif;
    return bits | ((code & 0x80) << 24);
}

// Decode one output plane from the packed tile in D0 into D(plane + 1).
// `scales` points at this plane's 32 E8M0 bytes: byte r is the scale of tile row r.
template <bool byte_input, bool fp32_dest>
inline void mxfp8_decode_plane(const volatile uint32_t* scales, uint32_t plane) {
    _llk_math_eltwise_sfpu_start_(0);
    // An SFPU vector holds even or odd columns of four face rows; eight lanes share
    // a row. LREG15 holds 2 * lane_id, so (id & 48) >> 1 is 8 * (row within the four).
    const sfpi::vUInt scale_shift = (sfpi::vUInt(sfpi::vConstTileId) & 48) >> 1;
    constexpr uint32_t vectors_per_face = 8;
    constexpr uint32_t vectors_per_tile = 4 * vectors_per_face;
    for (uint32_t face = 0; face < 4; ++face) {
        // Faces 0/1 cover tile rows 0-15, faces 2/3 rows 16-31.
        const uint32_t row_word = (face >> 1) * 4;
#pragma GCC unroll 8
        for (uint32_t vec = 0; vec < vectors_per_face; ++vec) {
            const uint32_t index = face * vectors_per_face + vec;
            sfpi::vUInt packed;
            if constexpr (byte_input || fp32_dest) {
                // UINT8 datacopy uses integer ELWADD and produces Integer-32 (32-bit
                // DEST only). UINT16 in FP32 DEST lands in the low half of each word.
                packed = sfpi::dst_reg[index].mode<sfpi::DataLayout::U32>();
            } else {
                packed = sfpi::dst_reg[index].mode<sfpi::DataLayout::U16>();
            }
            const sfpi::vUInt code = (packed >> (plane * 8)) & 0xFF;
            sfpi::vUInt scale = sfpi::vUInt(scales[row_word + (vec >> 1)]);
            scale = (scale >> scale_shift) & 0xFF;
            const sfpi::vFloat value = sfpi::as<sfpi::vFloat>(mxfp8_decode_bits(code, sfpi::as<sfpi::vInt>(scale)));
            // Every finite result has at most four significant bits, so the BF16
            // store's truncation is exact.
            if constexpr (fp32_dest) {
                sfpi::dst_reg[(plane + 1) * vectors_per_tile + index].mode<sfpi::DataLayout::F32>() = value;
            } else {
                sfpi::dst_reg[(plane + 1) * vectors_per_tile + index].mode<sfpi::DataLayout::F16b>() = value;
            }
        }
    }
    _llk_math_eltwise_sfpu_done_();
}

}  // namespace ckernel::sfpu
