// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel_ops.h"

namespace ckernel::sfpu {

__attribute__((always_inline)) inline void nvfp4_init_raw() {
    // Scale expansion uses all eight LRegs, so establish decoder constants
    // afterwards. The raw decoder preserves these registers across all faces.
    constexpr uint32_t load_unsigned_immediate = 2;
    // Active LUT segments encode x - 1 for 1 <= x < 2, and x / 2 otherwise.
    TTI_SFPLOADI(1, load_unsigned_immediate, 0x0080);
    TTI_SFPLOADI(2, load_unsigned_immediate, 0x10ff);
    TTI_SFPLOADI(7, load_unsigned_immediate, 7);
}

// Raw SFPU register map: L0 magnitude (unreachable LUT segment), L1/L2
// LUT coefficients, L3 signed LUT input/next packed input, L4/L5 block scales,
// L6 pipelined output, L7 magnitude mask. This helper contains no SFPI code.
// Call all four faces of all four UINT16 planes in order, beginning at (0,0).
// The last vector preloads the next face's scale pair into L4/L5.
template <bool fpu_tensor_scale, uint32_t global_bf16_bits>
__attribute__((always_inline)) inline void nvfp4_decode_column_face_raw(uint32_t plane, uint32_t face) {
    constexpr uint32_t fmt_bf16 = 2;
    constexpr uint32_t fmt_uint16 = 6;
    constexpr uint32_t addr_mode_no_increment = 7;
    constexpr uint32_t shift_immediate_from_vc = 5;
    constexpr uint32_t use_explicit_and_source = 1;
    constexpr uint32_t add_exponent = 1;
    constexpr uint32_t retain_lut_sign = 4;
    constexpr uint32_t round_nearest = 0;
    constexpr uint32_t round_to_bf16 = 1;
    constexpr uint32_t zero_register = 9;
    constexpr bool fold_tensor_scale = global_bf16_bits >= (27u << 23) && global_bf16_bits <= (227u << 23);
    constexpr uint32_t tensor_scale = global_bf16_bits >> 16;
    constexpr uint32_t faces_per_tile = 4;
    constexpr uint32_t values_per_uint16 = 4;
    const uint32_t next_face = plane * faces_per_tile + face + 1;
    const bool has_next_face = next_face < values_per_uint16 * faces_per_tile;

    if (plane == 0 && face == 0) {
        TTI_SFPLOAD(4, fmt_bf16, addr_mode_no_increment, (32 + plane * 8 + face * 2) * 2);
        TTI_SFPLOAD(5, fmt_bf16, addr_mode_no_increment, (33 + plane * 8 + face * 2) * 2);
        if constexpr (!fpu_tensor_scale && fold_tensor_scale) {
            TTI_SFPMULI(tensor_scale, 4, 0);
            TTI_SFPMULI(tensor_scale, 5, 0);
        }
    }
    // Prefetch the first packed vector, and keep its masked magnitude in L0.
    TTI_SFPLOAD(3, fmt_uint16, addr_mode_no_increment, face * 16);
    if (plane != 0) {
        TTI_SFPSHFT((0x1000u - plane * 4) & 0xfff, 3, 3, shift_immediate_from_vc);
    }
    TTI_SFPAND(3, 7, 0, use_explicit_and_source);
#pragma GCC unroll 8
    for (uint32_t vec = 0; vec < 8; ++vec) {
        // Arithmetic is identical for every vector and plane once its nibble
        // has been extracted. The first vector records slots 0..4 while running;
        // later calls reuse them until the next packed tile's scale expansion.
        // Slots 0..4 are disjoint from transpose_dest's math slots 16..31.
        if (plane == 0 && face == 0 && vec == 0) {
            TTI_REPLAY(0, 5, 1, 1);
            TTI_SFPSHFT(22, 0, 0, shift_immediate_from_vc);
            TTI_SFPDIVP2(127, 0, 0, add_exponent);
            TTI_SFPSHFT(28, 3, 3, shift_immediate_from_vc);
            TTI_SFPSETSGN(0, 0, 3, 0);
            // Write the LUT result to L6, freeing L3 for the next input immediately.
            TTI_SFPLUT(6, retain_lut_sign, 0);
        } else {
            TTI_REPLAY(0, 5, 0, 0);
        }
        if (vec < 7) {
            // Independent load hides the LUT's two-cycle result dependency.
            TTI_SFPLOAD(3, fmt_uint16, addr_mode_no_increment, (face * 8 + vec + 1) * 2);
        }
        if (vec == 7 && has_next_face) {
            // The final vector uses odd L5, so L4 can already hold the next face.
            TTI_SFPLOAD(4, fmt_bf16, addr_mode_no_increment, (32 + next_face * 2) * 2);
        }
        TTI_SFPMUL(6, (vec & 1) ? 5 : 4, zero_register, 6, 0);
        if (vec == 7 && has_next_face) {
            // MUL has consumed the old odd scale; this load hides its result wait.
            TTI_SFPLOAD(5, fmt_bf16, addr_mode_no_increment, (33 + next_face * 2) * 2);
        }
        if (vec < 7) {
            // Nibble extraction or masking hides the scale multiply dependency.
            if (plane != 0) {
                TTI_SFPSHFT((0x1000u - plane * 4) & 0xfff, 3, 3, shift_immediate_from_vc);
            } else {
                TTI_SFPAND(3, 7, 0, use_explicit_and_source);
            }
        }
        if constexpr (!fpu_tensor_scale && !fold_tensor_scale) {
            TTI_SFPMULI(tensor_scale, 6, 0);
        }
        TTI_SFP_STOCH_RND(round_nearest, 0, zero_register, 6, 6, round_to_bf16);
        TTI_SFPSTORE(6, fmt_bf16, addr_mode_no_increment, ((plane + 2) * 32 + face * 8 + vec) * 2);
        if (vec < 7 && plane != 0) {
            TTI_SFPAND(3, 7, 0, use_explicit_and_source);
        }
        if constexpr (!fpu_tensor_scale && fold_tensor_scale) {
            if (vec == 7 && has_next_face) {
                TTI_SFPMULI(tensor_scale, 4, 0);
                TTI_SFPMULI(tensor_scale, 5, 0);
            }
        }
    }
}
}  // namespace ckernel::sfpu
