// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "sfpi.h"
#include "sfpu/ckernel_sfpu_converter.h"
#include "llk_math_eltwise_sfpu_common.h"

namespace ckernel::sfpu {

// E2M1 magnitude table: 0, .5, 1, 1.5, 2, 3, 4, 6. All decode exactly in FP32.
sfpi_inline sfpi::vFloat nvfp4_decode_e2m1(sfpi::vUInt code) {
    sfpi::vUInt magnitude = code & 7;
    sfpi::vUInt bits = (((magnitude >> 1) + 126) << 23) | ((magnitude & 1) << 22);
    sfpi::vFloat value = sfpi::as<sfpi::vFloat>(bits);
    v_if(magnitude < 2) { value = sfpi::convert<sfpi::vFloat>(sfpi::as<sfpi::vInt>(magnitude)) * 0.5f; }
    v_endif;
    return sfpi::copysgn(value, sfpi::as<sfpi::vFloat>((code & 8) << 28));
}

sfpi_inline sfpi::vFloat nvfp4_decode_e4m3(sfpi::vUInt code) {
    // Scale contract: finite, nonnegative E4M3FN (0..126). Exponent bias is 7.
    sfpi::vUInt bits = (code << 20) + (120u << 23);
    sfpi::vFloat value = sfpi::as<sfpi::vFloat>(bits);
    v_if(code < 8) { value = sfpi::convert<sfpi::vFloat>(sfpi::as<sfpi::vInt>(code)) * 0x1p-9f; }
    v_endif;
    return value;
}

// Read the packed tile in D0 and write one expanded plane to D(plane+1). Keeping the
// integer input intact allows all planes to reuse a single unpack transaction.
template <bool byte_input, bool e2m1_scale, bool fp32_dest, bool bf16_output>
inline void nvfp4_decode_plane(const volatile uint32_t* scales, uint32_t plane, float global_scale) {
    _llk_math_eltwise_sfpu_start_(0);
    // An SFPU vector covers even/odd columns of four face rows. Eight lanes
    // share a row. LREG15 contains 2*lane_id: pick one scale byte per eight lanes.
    const sfpi::vUInt scale_shift = (sfpi::vUInt(sfpi::vConstTileId) & 48) >> 1;
    constexpr uint32_t vectors_per_face = 8;
    constexpr uint32_t vectors_per_tile = 4 * vectors_per_face;
    for (uint32_t face = 0; face < 4; ++face) {
#pragma GCC unroll 8
        for (uint32_t vec = 0; vec < vectors_per_face; ++vec) {
            const uint32_t index = face * vectors_per_face + vec;
            sfpi::vUInt packed;
            if constexpr (byte_input || fp32_dest) {
                // UINT8 datacopy uses integer ELWADD and produces Integer-32.
                // UINT16 in FP32 DEST is copied into the low half of each word.
                packed = sfpi::dst_reg[index].mode<sfpi::DataLayout::U32>();
            } else {
                packed = sfpi::dst_reg[index].mode<sfpi::DataLayout::U16>();
            }
            sfpi::vFloat value = nvfp4_decode_e2m1((packed >> (plane * 4)) & 15);
            sfpi::vUInt codes = sfpi::vUInt(scales[face * 4 + (vec >> 1)]);
            codes = (codes >> scale_shift) & 255;
            sfpi::vFloat scale;
            if constexpr (e2m1_scale) {
                scale = nvfp4_decode_e2m1(codes);
            } else {
                scale = nvfp4_decode_e4m3(codes);
            }
            value = (value * scale) * global_scale;
            if constexpr (bf16_output) {
                // Blackhole stores truncate and the packer rounds ties away.
                // Apply IEEE round-to-nearest-even explicitly in either DEST mode.
                sfpi::vUInt bits = sfpi::as<sfpi::vUInt>(value);
                bits = (bits + 0x7fffu + ((bits >> 16) & 1)) & 0xffff0000u;
                value = sfpi::as<sfpi::vFloat>(bits);
            }
            if constexpr (fp32_dest) {
                sfpi::dst_reg[(plane + 1) * vectors_per_tile + index].mode<sfpi::DataLayout::F32>() = value;
            } else {
                sfpi::dst_reg[(plane + 1) * vectors_per_tile + index].mode<sfpi::DataLayout::F16b>() = value;
            }
        }
    }
    _llk_math_eltwise_sfpu_done_();
}
// Preserve this uniform mask outside the eight LRegs used by scale expansion.
sfpi_inline void nvfp4_configure_raw_mask() { sfpi::vConstIntPrgm0 = static_cast<int32_t>(0x81c00000u); }

sfpi_inline void nvfp4_convert_begin() { _llk_math_eltwise_sfpu_start_(0); }
sfpi_inline void nvfp4_convert_end() { _llk_math_eltwise_sfpu_done_(); }

// Expand the compact 16x16 scale tile in D1 to face-broadcast vectors.
// Descending planes preserve the compact input vectors until they are read.
sfpi_inline void nvfp4_expand_scale_vectors() {
#pragma GCC unroll 4
    for (int plane = 3; plane >= 0; --plane) {
        sfpi::vFloat a = sfpi::dst_reg[32 + plane * 2].mode<sfpi::DataLayout::F16b>();
        sfpi::vFloat b = a, c = a, d = a;
        sfpi::vFloat e = sfpi::dst_reg[33 + plane * 2].mode<sfpi::DataLayout::F16b>();
        sfpi::vFloat f = e, g = e, h = e;
        sfpi::subvec_transp(a, b, c, d, e, f, g, h);
        sfpi::dst_reg[32 + plane * 8].mode<sfpi::DataLayout::F16b>() = a;
        sfpi::dst_reg[33 + plane * 8].mode<sfpi::DataLayout::F16b>() = e;
        sfpi::dst_reg[34 + plane * 8].mode<sfpi::DataLayout::F16b>() = b;
        sfpi::dst_reg[35 + plane * 8].mode<sfpi::DataLayout::F16b>() = f;
        sfpi::dst_reg[36 + plane * 8].mode<sfpi::DataLayout::F16b>() = c;
        sfpi::dst_reg[37 + plane * 8].mode<sfpi::DataLayout::F16b>() = g;
        sfpi::dst_reg[38 + plane * 8].mode<sfpi::DataLayout::F16b>() = d;
        sfpi::dst_reg[39 + plane * 8].mode<sfpi::DataLayout::F16b>() = h;
    }
}

}  // namespace ckernel::sfpu
