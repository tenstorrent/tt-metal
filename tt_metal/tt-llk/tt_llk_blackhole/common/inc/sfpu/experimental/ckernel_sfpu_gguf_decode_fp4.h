// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// GGUF SFPU decoders of MXFP4 and NVFP4, whose pages hold their codes as 4-bit block-float datums.

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

// The page's codes are BFP4 datums (sign in bit 3, magnitude code c in bits 0..2) in the [K][N] tile's face order,
// the E2M1 magnitudes permuted so that c / 4 falls on SFPLUTFP32's segments: c = 0..7 hold 0, 0.5, 4, 6, 1, 1.5,
// 2, 3. The unpacker reads them with the exponent forced to 15 (dest_srca_mm codes) into the 16-bit DEST rows that the
// output tile's rows address, Dst16b rows 64 idst + 2 base, so the SFPU loads x = +-c / 4 (FP16) where its output
// goes. The table (FP16 6-entry table 1, the sign of x kept) has slopes 4 S, 16 S, 4 S, 8 S and intercepts 0, 0,
// -2 S, -8 S on [0, .5), [.5, 1), [1, 1.5), [1.5, 2) (S = the scale times 0.5: values kv S, kv the doubled E2M1
// magnitude), and 0 from 2 on, where a code of sign 1 and magnitude 0 lands (-0). NVFP4's table holds its scale; each
// output is one SFPLOADMACRO: load x, the table into LReg16, store over the output in place. Without SFPLOADMACRO:
// load, table, store. The page (its scale row, scratch around it) is in the next DEST tile. fp4_bfp, FP4_CODE_EXP:
// experimental/ckernel_gguf_page.h.

// NVFP4: table words of group p = pidx(a, b, h), BFP_PS + 3 p + 0..2; MXFP4: S of (a, b), BFP_PS + pidx(a, b, 0).
constexpr int BFP_PS = CO + 8;

// Table word k (L0, L1, L5) of a group is (t + K[k]) | z: t = the fp16 bits of S in both halves (exact: S has four
// significant bits and lies in [2^-10, 224], so 16 S stays below 2^12), K[k] the exponent steps (and intercept signs)
// of the two coefficients; z = fp16 exponent 31 in both halves when S is 0, which the table reads as 0.
constexpr unsigned BFP_K0 = 0x10000800u; // 16 S | 4 S
constexpr unsigned BFP_K1 = 0x0C000800u; // 8 S | 4 S
constexpr unsigned BFP_K5 = 0x8C008400u; // -8 S | -2 S

// t and z of the four row groups' S (k-half h) into the page tile's entries 2 h, 2 h + 1 (rows 0..7, before the
// scale row).
template <int H>
sfpi_inline void bfp_scale(const vUInt x7)
{
    const vFloat s = ue4m3_half(x7);
    vUInt b        = (sfpi::as<vUInt>(s) >> 13u) - vUInt(112u << 10);
    vUInt z        = 0u;
    v_if (s == 0.0f)
    {
        z = vUInt(0x7C007C00u);
    }
    v_endif;
    sfpi::dst_reg[CO + 2 * H].mode<sfpi::DataLayout::U32>()     = (b << 16u) | b;
    sfpi::dst_reg[CO + 2 * H + 1].mode<sfpi::DataLayout::U32>() = z;
}

// Table word k of all eight groups: built for the four row groups of each k-half, then broadcast by one transpose.
template <int K, typename C>
sfpi_inline void bfp_words(const C& add)
{
    vUInt a0 = (ldu(CO) + add) | ldu(CO + 1);
    vUInt b0 = (ldu(CO + 2) + add) | ldu(CO + 3);
    vUInt a1 = a0, a2 = a0, a3 = a0;
    vUInt b1 = b0, b2 = b0, b3 = b0;
    sfpi::subvec_transp(a0, a1, a2, a3, b0, b1, b2, b3);
    sfpi::dst_reg[BFP_PS + 3 * pidx(0, 0, 0) + K].mode<sfpi::DataLayout::U32>() = a0;
    sfpi::dst_reg[BFP_PS + 3 * pidx(1, 0, 0) + K].mode<sfpi::DataLayout::U32>() = a1;
    sfpi::dst_reg[BFP_PS + 3 * pidx(0, 1, 0) + K].mode<sfpi::DataLayout::U32>() = a2;
    sfpi::dst_reg[BFP_PS + 3 * pidx(1, 1, 0) + K].mode<sfpi::DataLayout::U32>() = a3;
    sfpi::dst_reg[BFP_PS + 3 * pidx(0, 0, 1) + K].mode<sfpi::DataLayout::U32>() = b0;
    sfpi::dst_reg[BFP_PS + 3 * pidx(1, 0, 1) + K].mode<sfpi::DataLayout::U32>() = b1;
    sfpi::dst_reg[BFP_PS + 3 * pidx(0, 1, 1) + K].mode<sfpi::DataLayout::U32>() = b2;
    sfpi::dst_reg[BFP_PS + 3 * pidx(1, 1, 1) + K].mode<sfpi::DataLayout::U32>() = b3;
}

// Phase 1 of NVFP4: the table words of the tile's eight groups. S of (a, b, h) is byte 2h + b of the scale row, word
// parity a. A transpose gives row group g = 2b + a the word of parity a and a per-lane shift picks byte b, so one
// conversion per k-half yields the S of all four row groups; a transpose of four copies then broadcasts row group g's
// word to every lane.
sfpi_inline void fp4_bfp_params()
{
    sfpi::vConstIntPrgm0 = static_cast<int>(BFP_K0);
    sfpi::vConstIntPrgm1 = 0x3B800000; // ue4m3_half
    sfpi::vConstIntPrgm2 = static_cast<int>(BFP_K1);
    vUInt w0             = ldu(CO + scale_row<NVFP4>() / 2);
    vUInt w1             = ldu(CO + scale_row<NVFP4>() / 2 + 1);
    vUInt w2             = w0;
    vUInt w3             = w1;
    transp4(w0, w1, w2, w3);
    const vUInt sh = vUInt(25) - ((sfpi::as<vUInt>(sfpi::vConstTileId) >> 5u) << 3u);
    bfp_scale<0>((w0 << sh) >> 25u);
    bfp_scale<1>((w0 << (sh - 16u)) >> 25u);
    bfp_words<0>(sfpi::as<vUInt>(sfpi::vConstIntPrgm0));
    bfp_words<1>(sfpi::as<vUInt>(sfpi::vConstIntPrgm2));
    bfp_words<2>(vUInt(BFP_K5));
}

// Template 0 = the table into LReg16; macro 0 runs it 1 cycle after its load and stores LReg16 3 cycles after it.
// The delays count cycles and the store lands within 7 cycles of the macro, so the hardware arbitrates it against
// the unpacker's DEST writes; the macro's inputs (L0, L1, L5 and its own load) are all written before it.
inline void macro_init_fp4_bfp()
{
    if constexpr (!LOADMACRO)
    {
        return;
    }
    TTI_SFPLUTFP32(12, sfpi::SFPLUTFP32_MOD0_FP16_6ENTRY_TABLE1 | sfpi::SFPLUTFP32_MOD0_SGN_RETAIN);
    constexpr std::uint32_t lut   = 0x40 | 4;
    constexpr std::uint32_t store = 0x40 | (2 << 3) | 3;
    TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, lut << 8);
    TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, store << 8);
    TTI_SFPCONFIG(0, 4, 0);
    TTI_SFPCONFIG(0x000, 8, sfpi::SFPCONFIG_MOD1_ARG_IMM);
    TTI_SFPNOP;
}

// Phase 2 of NVFP4: per group three table words, then its four outputs.
sfpi_inline void body_fp4_bfp()
{
    TTI_SFPLOADI(2, sfpi::SFPLOADI_MOD0_LOWER, 0x7C00);
    TTI_SFPLOADI(2, sfpi::SFPLOADI_MOD0_UPPER, 0x7C00);
    TTI_SFPLOADI(4, sfpi::SFPLOADI_MOD0_LOWER, 0x7C00);
    TTI_SFPLOADI(4, sfpi::SFPLOADI_MOD0_UPPER, 0x7C00);
    TTI_SFPLOADI(6, sfpi::SFPLOADI_MOD0_LOWER, 0x7C00);
    TTI_SFPLOADI(6, sfpi::SFPLOADI_MOD0_UPPER, 0x7C00);
#pragma GCC unroll 2
    for (int a = 0; a < 2; a++)
    {
#pragma GCC unroll 2
        for (int h = 0; h < 2; h++)
        {
#pragma GCC unroll 2
            for (int b = 0; b < 2; b++)
            {
                const int p = pidx(a, b, h);
                TTI_SFPLOAD(0, sfpi::SFPLOAD_MOD0_FMT_INT32, ADDR_MOD_7, 2 * (BFP_PS + 3 * p));
                TTI_SFPLOAD(1, sfpi::SFPLOAD_MOD0_FMT_INT32, ADDR_MOD_7, 2 * (BFP_PS + 3 * p + 1));
                TTI_SFPLOAD(5, sfpi::SFPLOAD_MOD0_FMT_INT32, ADDR_MOD_7, 2 * (BFP_PS + 3 * p + 2));
#pragma GCC unroll 4
                for (int j = 0; j < 4; j++)
                {
                    const int o = 16 * h + 8 * a + 2 * j + b;
                    if constexpr (LOADMACRO)
                    {
                        TTI_SFPLOADMACRO(3, InstrModLoadStore::FP16A, ADDR_MOD_7, 2 * o);
                    }
                    else
                    {
                        TTI_SFPLOAD(3, sfpi::SFPLOAD_MOD0_FMT_FP16A, ADDR_MOD_7, 2 * o);
                        TTI_SFPLUTFP32(7, sfpi::SFPLUTFP32_MOD0_FP16_6ENTRY_TABLE1 | sfpi::SFPLUTFP32_MOD0_SGN_RETAIN);
                        TTI_SFPSTORE(7, sfpi::SFPSTORE_MOD0_FMT_FP32, ADDR_MOD_7, 2 * o);
                    }
                }
            }
        }
    }
#pragma GCC unroll 3
    for (int n = 0; n < 3; n++)
    {
        TTI_SFPNOP;
    }
}

// MXFP4: the scale, a power of two of any exponent, does not fit the table's fp16 coefficients: the table (S = 1)
// gives kv, an SFPMAD multiplies it by S (LReg7), per group (a, b) of eight outputs. Phase 1: S of (a, b) is byte
// 2 (TI % 2) + b of the scale row, word parity a, broadcast as NVFP4's.
template <int TI>
sfpi_inline void fp4_bfp_mx_params()
{
    vUInt w0 = ldu(CO + scale_row<MXFP4>() / 2);
    vUInt w1 = ldu(CO + scale_row<MXFP4>() / 2 + 1);
    vUInt w2 = w0;
    vUInt w3 = w1;
    transp4(w0, w1, w2, w3);
    const vUInt sh = vUInt(25 - 16 * (TI % 2)) - ((sfpi::as<vUInt>(sfpi::vConstTileId) >> 5u) << 3u);
    vFloat s0      = e8m0_half((w0 << (sh - 1u)) >> 24u);
    vFloat s1      = s0;
    vFloat s2      = s0;
    vFloat s3      = s0;
    transp4(s0, s1, s2, s3);
    sfpi::dst_reg[BFP_PS + pidx(0, 0, 0)] = s0;
    sfpi::dst_reg[BFP_PS + pidx(1, 0, 0)] = s1;
    sfpi::dst_reg[BFP_PS + pidx(0, 1, 0)] = s2;
    sfpi::dst_reg[BFP_PS + pidx(1, 1, 0)] = s3;
}

// Templates 0 = the table into the macro's LReg (3), 1 = LReg7 times LReg3 into LReg16. Macro 0 loads x and runs the
// table 1 cycle after; macro 1 runs the product 2 cycles after it, once the table's result is written, and stores
// LReg16 4 cycles after it. Macro 1's own load (zeros) goes to LReg2, which no table segment the codes reach reads.
inline void macro_init_fp4_bfp_mx()
{
    if constexpr (!LOADMACRO)
    {
        return;
    }
    TTI_SFPLUTFP32(12, sfpi::SFPLUTFP32_MOD0_FP16_6ENTRY_TABLE1 | sfpi::SFPLUTFP32_MOD0_SGN_RETAIN);
    TTI_SFPMAD(7, 3, 9, 13, sfpi::SFPMAD_MOD1_OFFSET_NONE);
    constexpr std::uint32_t lut   = 4;
    constexpr std::uint32_t mul   = 0x40 | (1 << 3) | 5;
    constexpr std::uint32_t store = 0x40 | (3 << 3) | 3;
    TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, lut << 8);
    TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, 0);
    TTI_SFPCONFIG(0, 4, 0);
    TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, mul << 8);
    TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, store << 8);
    TTI_SFPCONFIG(0, 5, 0);
    TTI_SFPCONFIG(0x000, 8, sfpi::SFPCONFIG_MOD1_ARG_IMM);
    TTI_SFPNOP;
}

// Phase 2 of MXFP4: two SFPLOADMACROs and an SFPNOP per output.
sfpi_inline void body_fp4_bfp_mx()
{
    // The table at S = 1: slopes 4, 16, 4, 8, intercepts 0, 0, -2, -8 on [0, 2). From 2 on (only -0, x = -131008)
    // slope 2^-15 (fp16 0, as macro 1's zero load leaves LReg2) and intercept -(4 - 2^-9): -0.
    TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, 0x4400);
    TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, 0x4C00);
    TTI_SFPLOADI(1, sfpi::SFPLOADI_MOD0_LOWER, 0x4400);
    TTI_SFPLOADI(1, sfpi::SFPLOADI_MOD0_UPPER, 0x4800);
    TTI_SFPLOADI(5, sfpi::SFPLOADI_MOD0_LOWER, 0xC000);
    TTI_SFPLOADI(5, sfpi::SFPLOADI_MOD0_UPPER, 0xC800);
    TTI_SFPLOADI(2, sfpi::SFPLOADI_MOD0_LOWER, 0);
    TTI_SFPLOADI(2, sfpi::SFPLOADI_MOD0_UPPER, 0);
    TTI_SFPLOADI(4, sfpi::SFPLOADI_MOD0_LOWER, 0x7C00);
    TTI_SFPLOADI(4, sfpi::SFPLOADI_MOD0_UPPER, 0x7C00);
    TTI_SFPLOADI(6, sfpi::SFPLOADI_MOD0_LOWER, 0xC3FF);
    TTI_SFPLOADI(6, sfpi::SFPLOADI_MOD0_UPPER, 0xC3FF);
#pragma GCC unroll 2
    for (int a = 0; a < 2; a++)
    {
#pragma GCC unroll 2
        for (int b = 0; b < 2; b++)
        {
            // The previous group's last product reads LReg7 before this load.
            TTI_SFPNOP;
            TTI_SFPLOAD(7, sfpi::SFPLOAD_MOD0_FMT_FP32, ADDR_MOD_7, 2 * (BFP_PS + pidx(a, b, 0)));
#pragma GCC unroll 2
            for (int h = 0; h < 2; h++)
            {
#pragma GCC unroll 4
                for (int j = 0; j < 4; j++)
                {
                    const int o = 16 * h + 8 * a + 2 * j + b;
                    if constexpr (LOADMACRO)
                    {
                        TTI_SFPLOADMACRO(3, InstrModLoadStore::FP16A, ADDR_MOD_7, 2 * o);
                        TTI_SFPLOADMACRO((1 << 2) | 2, LOAD_NONE, ADDR_MOD_7, 2 * o);
                        TTI_SFPNOP;
                    }
                    else
                    {
                        TTI_SFPLOAD(3, sfpi::SFPLOAD_MOD0_FMT_FP16A, ADDR_MOD_7, 2 * o);
                        TTI_SFPLUTFP32(3, sfpi::SFPLUTFP32_MOD0_FP16_6ENTRY_TABLE1 | sfpi::SFPLUTFP32_MOD0_SGN_RETAIN);
                        TTI_SFPMAD(7, 3, 9, 3, sfpi::SFPMAD_MOD1_OFFSET_NONE);
                        TTI_SFPSTORE(3, sfpi::SFPSTORE_MOD0_FMT_FP32, ADDR_MOD_7, 2 * o);
                    }
                }
            }
        }
    }
#pragma GCC unroll 4
    for (int n = 0; n < 4; n++)
    {
        TTI_SFPNOP;
    }
}

} // namespace gguf
} // namespace ckernel::sfpu
