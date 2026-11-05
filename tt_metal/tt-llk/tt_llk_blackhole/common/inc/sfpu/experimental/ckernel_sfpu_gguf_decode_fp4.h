// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// GGUF SFPU decoders of MXFP4 and NVFP4.

#pragma once

#include <cstdint>

#include "sfpu/experimental/ckernel_sfpu_gguf_decode_common.h"

namespace ckernel::sfpu
{
namespace gguf
{

using namespace ckernel::gguf;

// E8M0 scale times 0.5 (ggml's GGML_E8M0_TO_FP32_HALF): 2^(e - 128).
sfpi_inline vFloat e8m0_half(const vUInt e)
{
    vUInt v = (e - 1u) << 23;
    v_if (sfpi::as<vInt>(e) < 2)
    {
        v = (e + 1u) << 21;
    }
    v_endif;
    return sfpi::as<vFloat>(v);
}

// Unsigned E4M3 (low 7 bits of x) times 0.5 (ggml's ue4m3 conversion); 0 and 0x7F decode to 0.
sfpi_inline vFloat ue4m3_half(const vUInt x7)
{
    const vFloat b = sfpi::as<vFloat>((x7 << 20u) + sfpi::as<vUInt>(sfpi::vConstIntPrgm1));
    const vFloat a = sfpi::addexp(b, 1) + -0.0078125f;
    vFloat v       = sfpi::min_max(a, b).first;
    v_if (sfpi::as<vInt>(x7) == 0x7F)
    {
        v = 0.0f;
    }
    v_endif;
    return v;
}

sfpi_inline void e2m1_init()
{
    sfpi::vConstIntPrgm0 = static_cast<int>(E2M1_TABLE);
    sfpi::vConstIntPrgm1 = 0x1C;
}

// Phase 1 of MXFP4 and NVFP4. TI = tile index inside the scale page's group of K-tiles.
template <int FMT, int TI>
sfpi_inline void fp4_params()
{
    if constexpr (FMT != MXFP4 && page_layout<static_cast<Format>(FMT)>().fp16_rows == 0)
    {
        params_init();
    }
#pragma GCC unroll 2
    for (int a = 0; a < 2; a++)
    {
        if constexpr (FMT == MXFP4)
        {
            // Row 0 byte 2 * (TI % 2) + b: E8M0 of column b for tile TI of the pair.
            vUInt r0, r1, r2, r3;
            rep4<FMT>(0, a, r0, r1, r2, r3);
#pragma GCC unroll 2
            for (int b = 0; b < 2; b++)
            {
                sfpi::dst_reg[PS + pidx(a, b, 0)] = e8m0_half(bits(r0, 16 * (TI % 2) + 8 * b, 8));
            }
        }
        else if constexpr (FMT == NVFP4)
        {
            // Row 0 byte 2h + b: UE4M3 of column b, k-half h.
            vUInt r0, r1, r2, r3;
            rep4<FMT>(0, a, r0, r1, r2, r3);
#pragma GCC unroll 2
            for (int h = 0; h < 2; h++)
            {
#pragma GCC unroll 2
                for (int b = 0; b < 2; b++)
                {
                    sfpi::dst_reg[PS + pidx(a, b, h)] = ue4m3_half(bits(r0, 8 * (2 * h + b), 7));
                }
            }
        }
    }
}

// MXFP4 and NVFP4 through SFPLOADMACRO (body_macro_fp4).
template <int FMT>
constexpr bool fp4_macro_body = LOADMACRO && (FMT == MXFP4 || FMT == NVFP4);

// MXFP4 / NVFP4 instruction templates: 0 = E2M1_TABLE (LReg12) << x, 1 = x with the sign of S (LReg4) replaced by z's,
// 2 / 3 = L0 / L1 times z. Macro 0 (v) runs the table shift 2 instructions after it; macro 1 / 2 (z, v = L0 / L1) the
// sign 4, the product into LReg16 5 and its store 7 instructions after it.
inline void macro_init_fp4()
{
    TTI_SFPSHFT2(12, 0, 12, sfpi::SFPSHFT2_MOD1_SHFT_LREG);
    TTI_SFPSETSGN(0, 4, 13, 0);
    TTI_SFPMAD(0, 0, 9, 14, sfpi::SFPMAD_MOD1_OFFSET_NONE);
    TTI_SFPMAD(1, 0, 9, 15, sfpi::SFPMAD_MOD1_OFFSET_NONE);
    constexpr std::uint32_t table = (1 << 3) | 4;
    constexpr std::uint32_t sign  = 0x80 | (3 << 3) | 5;
    constexpr std::uint32_t mul0  = 0xC0 | (4 << 3) | 6;
    constexpr std::uint32_t store = 0x40 | (6 << 3) | 3;
    TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, 0);
    TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, table);
    TTI_SFPCONFIG(0, 4, 0);
    TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mul0 << 8) | sign);
    TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, store << 8);
    TTI_SFPCONFIG(0, 5, 0);
    TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, ((mul0 + 1) << 8) | sign);
    TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, store << 8);
    TTI_SFPCONFIG(0, 6, 0);
    TTI_SFPCONFIG(0xF00, 8, sfpi::SFPCONFIG_MOD1_ARG_IMM);
    TTI_SFPNOP;
}

// MXFP4 / NVFP4: body()'s arithmetic, copysgn(kv, s) S + 0 as kv (+-S) + 0 (equal bits, also for zeros). Groups
// (a, h, b) of four outputs o = 16h + 8a + 2j + b read nibbles i = 2j + b of code word 2h + a, masked to their parity
// (x, L3) so that x >> (4i - 2) is 4 code3 below zero bits: the table shift amount. Per output six instructions: macro
// 0 on v (table shift), x >> (4i - 2) into v, macro 1 / 2 on z (sign, product, store), v >> 28, x << (28 - 4i) into z
// (the sign at bit 31), the cast of v. Between groups: the code word, its mask, S (L4), after the last reads of x / S.
template <int FMT>
sfpi_inline void body_macro_fp4()
{
    sfpi::vConstIntPrgm0 = static_cast<int>(E2M1_TABLE);
    sfpi::vConstIntPrgm1 = 0x0F0F0F0F;
    sfpi::vConstIntPrgm2 = static_cast<int>(0xF0F0F0F0u);
#pragma GCC unroll 2
    for (int a = 0; a < 2; a++)
    {
#pragma GCC unroll 2
        for (int h = 0; h < 2; h++)
        {
#pragma GCC unroll 2
            for (int b = 0; b < 2; b++)
            {
                TTI_SFPLOAD(3, sfpi::SFPLOAD_MOD0_FMT_INT32, ADDR_MOD_7, 2 * (CO + 2 * h + a));
                TTI_SFPAND(0, 13 + b, 3, 0);
                TTI_SFPLOAD(4, sfpi::SFPLOAD_MOD0_FMT_SRCB, ADDR_MOD_7, 2 * (PS + pidx(a, b, traits<FMT>::has_h ? h : 0)));
#pragma GCC unroll 4
                for (int j = 0; j < 4; j++)
                {
                    const int i = 2 * j + b;
                    const int o = 16 * h + 8 * a + i;
                    const int v = j % 2;
                    TTI_SFPLOADMACRO(v, LOAD_NONE, ADDR_MOD_7, 2 * o);
                    TTI_SFPSHFT((2 - 4 * i) & 0xFFF, 3, v, SHIFT_IMM_OF_C);
                    TTI_SFPLOADMACRO(((1 + v) << 2) | 2, LOAD_NONE, ADDR_MOD_7, 2 * o);
                    TTI_SFPSHFT(-28 & 0xFFF, v, v, SHIFT_IMM_OF_C);
                    TTI_SFPSHFT(28 - 4 * i, 3, 2, SHIFT_IMM_OF_C);
                    TTI_SFPCAST(v, v, sfpi::SFPCAST_MOD1_SM32_TO_FP32_RNE);
                }
            }
        }
    }
#pragma GCC unroll 5
    for (int n = 0; n < 5; n++)
    {
        TTI_SFPNOP;
    }
}

} // namespace gguf
} // namespace ckernel::sfpu
