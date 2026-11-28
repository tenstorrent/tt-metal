// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// g6_typecast SFPU chain elements (Perf 1).
//
// int32 -> bfloat16, rounded exactly as torch (int32 -> fp32 RNE, then fp32 -> bf16 RNE): ONE chain
// element with ONE init for the whole kernel (Perf 1, perf_experiments/int32_bf16_fused_sfpu, variant 9).
//
// Pass 1: the LLK int32 -> fp32 SFPLOADMACRO pass (calculate_typecast_int32_to_fp32, 4 instr / DEST row).
// Pass 2: fp32 -> bf16 RNE by Veltkamp splitting with sigma = 2^16 + 1 (g = sigma*x, d = x - g, h = g + d is x
//         rounded to 8 mantissa bits, ties to even, in plain RNE float32 arithmetic), as an SFPLOADMACRO pass
//         (3 instr / row). The rounded value is stored FP32 with a zero low half, so the Float16_b pack is exact.
// Exactness: host-verified over ALL 2^32 int32 inputs; device-verified bit-exact on ~6.5M values incl. the
// double-rounding witnesses 33685505 and 2^24 + 2^16 + 1, INT32_MIN/MAX and ties around 2^24..2^31.
//
// RAW-LLK JUSTIFICATION (helper bypass, measured — do not "fix" back to two Typecast elements):
//  - no LLK / kernel_lib Typecast does int32 -> bf16 with RNE: the direct Int32 -> Float16_b arm (and the
//    Blackhole packer) round fp32 -> bf16 ties AWAY from zero (wrong on 33685505 etc.);
//  - the exact composition Typecast<Int32,Float32> -> Typecast<Float32,Float16_b> cannot keep its inits hoisted:
//    both inits write vConstIntPrgm0 (LREG12: -31 vs 1), so the chain re-inits both on every tile.
//  Measured on Q2228 [1,1,32,128] (4 tiles, 1 per core): two-element chain 1449-1484 ns, this element 1316-1320 ns.
//  Macro register use: the LLK pass owns macro 0 / templates 0-1; the rounding pass uses macros 1-2 /
//  templates 2-3, sigma in LREG13, and switches the macro misc config for its pass and back.
//  FRAGILE: the NOP delays in calculate_round_veltkamp_macro are tuned (without the first-row NOP rows 0-1 of
//  every face came out wrong); re-run the exact gate (test_g6_typecast_sources.py, Q2228) if they change.
#pragma once

#include <cstdint>

#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp"

#ifdef TRISC_MATH
#include "ckernel_sfpu_typecast.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"

namespace g6_sfpu {
using namespace ckernel;
using namespace ckernel::sfpu;

// SFPLOADMACRO Veltkamp rounding pass (runs after the LLK int32->fp32 macro pass on the same tile).
// Encoding per unit byte (see ckernel_sfpu_mul_int32.h): bit7 VB=VD (else VC=VD), bit6 VD=L16, bits5-3 delay,
// bits2-0 template 4+i or 3 = store. Uses macros 1, 2 and templates 2, 3 (the int32->fp32 pass owns macro 0 and
// templates 0, 1), so both passes' configuration lives side by side after ONE init.
//
// Row r: x_r = L[r % 4] (loaded by the macro), g_r = L[4 + r % 2]. Stream per row: LM_r, G_r, H_{r-1}.
//   G_r : g_r = x_r * sigma                  (explicit MAD)
//   D_r : [x_r] = g_r * -1 + x_r              (macro MAD, template 2 + r % 2, delay 2)
//   H_r : x_r = x_r * 1 + g_r                 (explicit MAD, one row later)
//   S_r : store [x_r] FP32                    (macro store, delay 7)
// All units count delay in issued instructions (misc 0xf00), switched per pass and restored for the LLK pass.
#ifndef VK_CONFIG_SETTLE_NOPS
#define VK_CONFIG_SETTLE_NOPS 3
#endif
constexpr std::uint32_t VK_MISC_ROUND = 0xf00 | InstrModLoadStore::FP32;
constexpr std::uint32_t VK_MISC_INT32_TO_FP32 = 0x100 | InstrModLoadStore::FP32;

inline void init_twopass_veltkamp_macro() {
    init_typecast_int32_to_fp32<false>();
    sfpi::vConstFloatPrgm1 = 65537.0f;  // LREG13 = sigma
    // Templates 2 / 3: D with g = L4 / L5.
    TTI_SFPMAD(p_sfpu::LREG4, p_sfpu::LCONST_neg1, 0, 12 + 2, 0);
    TTI_SFPMAD(p_sfpu::LREG5, p_sfpu::LCONST_neg1, 0, 12 + 3, 0);
    constexpr std::uint32_t disabled = 7 << 3;
    constexpr std::uint32_t store_bits = (7 << 3) | 3;
    {
        constexpr std::uint32_t mad_bits = (2 << 3) | (4 + 2);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mad_bits << 8) | disabled);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store_bits << 8) | disabled);
        TTI_SFPCONFIG(0, 4 + 1, 0);
    }
    {
        constexpr std::uint32_t mad_bits = (2 << 3) | (4 + 3);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mad_bits << 8) | disabled);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store_bits << 8) | disabled);
        TTI_SFPCONFIG(0, 4 + 2, 0);
    }
}

template <int ITERATIONS>
inline void calculate_round_veltkamp_macro() {
    TTI_SFPCONFIG(VK_MISC_ROUND, 8, 1);
    for (int i = 0; i < VK_CONFIG_SETTLE_NOPS; i++) {
        TTI_SFPNOP;
    }
#pragma GCC unroll 8
    for (int r = 0; r < ITERATIONS; r++) {
        const int x = r & 3;
        const int g = 4 + (r & 1);
        const int m = 1 + (r & 1);
        TT_SFPLOADMACRO((m << 2) | x, InstrModLoadStore::FP32, ADDR_MOD_6, 0);
        TT_SFPMAD(x, p_sfpu::LREG13, p_sfpu::LCONST_0, g, 0);
        if (r > 0) {
            const int xp = (r - 1) & 3;
            const int gp = 4 + ((r - 1) & 1);
            TT_SFPMAD(xp, p_sfpu::LCONST_1, gp, xp, 0);
        } else {
            TTI_SFPNOP;
        }
    }
    TTI_SFPNOP;
    TTI_SFPNOP;
    {
        constexpr int xl = (ITERATIONS - 1) & 3;
        constexpr int gl = 4 + ((ITERATIONS - 1) & 1);
        TTI_SFPMAD(xl, p_sfpu::LCONST_1, gl, xl, 0);
    }
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPCONFIG(VK_MISC_INT32_TO_FP32, 8, 1);
}
}  // namespace g6_sfpu
#endif

namespace compute_kernel_lib {

// Perf 1 sub-tile split: the vector mode that covers exactly NF faces of the DEST tile (the tiny-tile CB holds NF
// faces).
template <uint32_t NF>
constexpr VectorMode split_vector_mode() {
    static_assert(NF == 1 || NF == 2 || NF == 4);
    return NF == 4 ? VectorMode::RC : (NF == 2 ? VectorMode::R : VectorMode::None);
}

// int32 -> bf16 RNE element (header comment), SFPU passes restricted to the NF faces of the (tiny) tile.
template <uint32_t NF, Dst Slot = Dst::D0>
struct Int32ToBf16Rne : UnaryOp<Int32ToBf16Rne<NF, Slot>, Slot> {
    static ALWI void init() {
        MATH((ckernel::llk_math_eltwise_unary_sfpu_init<SfpuType::typecast>(g6_sfpu::init_twopass_veltkamp_macro)));
    }
    static ALWI void exec_impl(uint32_t slot_offset) {
        MATH((_llk_math_eltwise_unary_sfpu_params_(
            ckernel::sfpu::calculate_typecast_int32_to_fp32<false, 8>,
            to_u32(Slot) + slot_offset,
            split_vector_mode<NF>())));
        MATH((_llk_math_eltwise_unary_sfpu_params_(
            g6_sfpu::calculate_round_veltkamp_macro<8>, to_u32(Slot) + slot_offset, split_vector_mode<NF>())));
    }
};

// Typecast restricted to the NF faces of a tiny tile (Perf 1 sub-tile split). RAW-LLK JUSTIFICATION (helper
// bypass, measured): kernel_lib Typecast / typecast_tile hard-code VectorMode::RC, i.e. the SFPU always walks 4
// faces; on a 1- or 2-face tiny tile that is 2-4x the SFPU work. Measured with the split (16 cores, 1 face each):
// Q2229 bf16->int32 1262 -> ~1006 ns. NF == 4 is exactly typecast_tile. Arms without an NF-restricted call here
// fall back to typecast_tile (correct: the extra faces of DEST slot 0 are never packed). Init is the stock
// typecast_tile_init<in, out>.
template <uint32_t InDF, uint32_t OutDF, uint32_t NF, Dst Slot = Dst::D0>
struct TypecastNF : UnaryOp<TypecastNF<InDF, OutDF, NF, Slot>, Slot> {
    static ALWI void init() { typecast_tile_init<InDF, OutDF>(); }
    static ALWI void exec_impl(uint32_t slot_offset) {
        constexpr auto in_f = static_cast<DataFormat>(InDF);
        constexpr auto out_f = static_cast<DataFormat>(OutDF);
        const uint32_t d = to_u32(Slot) + slot_offset;
        if constexpr (NF == 4) {
            typecast_tile<InDF, OutDF>(d);
        } else if constexpr (in_f == DataFormat::Float16_b && out_f == DataFormat::Int32) {
            MATH((_llk_math_eltwise_unary_sfpu_params_(
                ckernel::sfpu::calculate_typecast_fp32_to_int32<APPROX, 8>, d, split_vector_mode<NF>())));
        } else if constexpr (in_f == DataFormat::Float32 && out_f == DataFormat::Float16_b) {
            MATH((_llk_math_eltwise_unary_sfpu_params_(
                ckernel::sfpu::calculate_typecast_fp32_to_fp16b<APPROX, 8>, d, split_vector_mode<NF>())));
        } else if constexpr (in_f == DataFormat::UInt16 && out_f == DataFormat::Int32) {
            MATH((_llk_math_eltwise_unary_sfpu_params_(
                ckernel::sfpu::calculate_typecast_uint16_to_uint32<APPROX, 8, DST_ACCUM_MODE>,
                d,
                split_vector_mode<NF>())));
        } else {
            // Packer-only arms (bf16 -> fp32 / bf8b ...): typecast_tile is a no-op or 4-face; keep stock.
            typecast_tile<InDF, OutDF>(d);
        }
    }
};

}  // namespace compute_kernel_lib
