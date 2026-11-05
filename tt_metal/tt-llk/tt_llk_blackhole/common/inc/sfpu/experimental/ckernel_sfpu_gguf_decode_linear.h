// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// GGUF SFPU decoders of the linear formats (Q4_*, Q5_*, Q8_0, Q*_K, TQ*, Q1_0, Q2_0): phase 1, the SFPLOADMACRO
// bodies with their compile-time issue scheduler, and TQ1_0.

#pragma once

#include <cstdint>
#include <utility>

#include "sfpu/experimental/ckernel_sfpu_gguf_decode_common.h"

namespace ckernel::sfpu
{
namespace gguf
{

using namespace ckernel::gguf;

template <int FMT>
constexpr bool q40_params = FMT == Q8_0 || FMT == Q4_0 || FMT == Q5_0 || FMT == Q4_1 || FMT == Q5_1;

// Q4_K and Q5_K: k4_group (sfpu_gguf_decode_group) unpacks a group's 6-bit scales and mins once, from the scale rows
// (0..5: 12 bytes per column), into scratch entries K4 .. K4 + 15. Word K4 + 4 (2a + b) + 2v + u, byte t: field v
// (0 scale, 1 min) of tile 4u + t, column half b, word parity a; the bytes of tiles 0..3 keep their native top two bits
// (bits() drops them). Q4_K's page holds rows 6, 7 (fp16 d, dmin) as stored: k4_group converts them and stores them
// where pf32 reads prepared copies, past the 1 KB page.
constexpr int K4 = 2 * TILE;

// f16_at with the fp16 rebias as an immediate (the body owns the programmable constants).
template <int HALF>
sfpi_inline vFloat k4_f16(const vUInt w)
{
    const vUInt m13 = HALF == 0 ? (w << 17u) >> 4u : ((w >> 16u) << 17u) >> 4u;
    const vFloat b  = sfpi::as<vFloat>(m13 + vUInt(0x38000000u));
    const vFloat a  = sfpi::addexp(b, 1) + -6.103515625e-05f;
    const vInt sgn  = sfpi::as<vInt>(HALF == 0 ? w << 16u : w);
    return sfpi::copysgn(sfpi::min_max(a, b).first, sgn);
}

template <int FMT>
sfpi_inline void k4_group()
{
    constexpr int src = CO + scale_row<FMT>() / 2;
#pragma GCC unroll 2
    for (int a = 0; a < 2; a++)
    {
        const int w0 = K4 + 8 * a, w1 = w0 + 4;
        // Scale bytes 0..15 (rows 0..3): column half 0's scales of tiles 0..3, mins of 0..3, low / high nibbles of
        // tiles 4..7; column half 1's scales of tiles 0..3.
        vUInt r0 = ldu(src + a), r1 = r0, r2 = r0, r3 = r0;
        transp4(r0, r1, r2, r3);
        sfpi::dst_reg[w0 + 0].mode<sfpi::DataLayout::U32>() = r0;
        sfpi::dst_reg[w0 + 2].mode<sfpi::DataLayout::U32>() = r1;
        sfpi::dst_reg[w0 + 1].mode<sfpi::DataLayout::U32>() = (r2 & vUInt(0x0F0F0F0Fu)) | ((r0 >> 2u) & vUInt(0x30303030u));
        sfpi::dst_reg[w0 + 3].mode<sfpi::DataLayout::U32>() = ((r2 >> 4u) & vUInt(0x0F0F0F0Fu)) | ((r1 >> 2u) & vUInt(0x30303030u));
        sfpi::dst_reg[w1 + 0].mode<sfpi::DataLayout::U32>() = r3;
        sfpi::dst_reg[w1 + 1].mode<sfpi::DataLayout::U32>() = (r3 >> 2u) & vUInt(0x30303030u);
        // Bytes 16..23 (rows 4, 5): column half 1's mins of tiles 0..3, nibbles of tiles 4..7.
        vUInt q0 = ldu(src + 2 + a), q1 = q0, q2 = q0, q3 = q0;
        transp4(q0, q1, q2, q3);
        sfpi::dst_reg[w1 + 2].mode<sfpi::DataLayout::U32>() = q0;
        sfpi::dst_reg[w1 + 1].mode<sfpi::DataLayout::U32>() = ldu(w1 + 1) | (q1 & vUInt(0x0F0F0F0Fu));
        sfpi::dst_reg[w1 + 3].mode<sfpi::DataLayout::U32>() = ((q1 >> 4u) & vUInt(0x0F0F0F0Fu)) | ((q0 >> 2u) & vUInt(0x30303030u));
        if constexpr (FMT == Q4_K)
        {
            // Rows 6, 7 (q2, q3): fp16 d and dmin, column half b in bits 16b.
#pragma GCC unroll 2
            for (int F = 0; F < 2; F++)
            {
                const vUInt w                         = F ? q3 : q2;
                sfpi::dst_reg[pf32_idx<FMT>(F, 0, a)] = k4_f16<0>(w);
                sfpi::dst_reg[pf32_idx<FMT>(F, 1, a)] = k4_f16<1>(w);
            }
        }
    }
}

// Phase 1 of the linear formats and of IQ4_NL (Q4_0's scale rows). TI = tile index inside the scale page's group of
// K-tiles.
template <int FMT, int TI>
sfpi_inline void linear_params()
{
    if constexpr (page_layout<static_cast<Format>(FMT)>().fp16_rows == 0 && FMT != Q4_K)
    {
        params_init<!q40_params<FMT>>();
    }
#pragma GCC unroll 2
    for (int a = 0; a < 2; a++)
    {
        if constexpr (q40_params<FMT>)
        {
            // Row 0: d of columns n0 (low half) and n1 (high half); Q4_1 row 1: m; Q5_0 row TI: d; Q5_1 rows 2 TI,
            // 2 TI + 1: d, m. The two columns' conversions interleave.
            vUInt r0, r1, r2, r3;
            rep4<FMT>(0, a, r0, r1, r2, r3);
            const vUInt dw = FMT == Q5_1 && TI % 2 ? r2 : FMT == Q5_0 && TI % 2 ? r1 : r0;
            vFloat d0, d1;
            f16_both(dw, d0, d1);
            if constexpr (FMT == Q4_1 || FMT == Q5_1)
            {
                vFloat m0, m1;
                f16_both(FMT == Q5_1 && TI % 2 ? r3 : r1, m0, m1);
                sfpi::dst_reg[PS + pidx(a, 0, 0)] = d0;
                sfpi::dst_reg[PS + pidx(a, 1, 0)] = d1;
                sfpi::dst_reg[PM + pidx(a, 0, 0)] = m0;
                sfpi::dst_reg[PM + pidx(a, 1, 0)] = m1;
            }
            else
            {
                // M = c d in d's register after its store.
                constexpr float c                 = FMT == Q8_0 ? -128.0f : FMT == Q4_0 ? -8.0f : -16.0f;
                sfpi::dst_reg[PS + pidx(a, 0, 0)] = d0;
                sfpi::dst_reg[PS + pidx(a, 1, 0)] = d1;
                d0                                = d0 * c;
                d1                                = d1 * c;
                sfpi::dst_reg[PM + pidx(a, 0, 0)] = d0;
                sfpi::dst_reg[PM + pidx(a, 1, 0)] = d1;
            }
        }
        else if constexpr (FMT == IQ4_NL || FMT == TQ2_0 || FMT == Q2_0 || FMT == Q1_0 || FMT == TQ1_0)
        {
            // Row 0: d of columns n0 (low half) and n1 (high half); TQ1_0 row 2: d.
            vUInt r0, r1, r2, r3;
            rep4<FMT>(0, a, r0, r1, r2, r3);
            const vUInt dw = FMT == TQ1_0 ? r2 : r0;
#pragma GCC unroll 2
            for (int b = 0; b < 2; b++)
            {
                vFloat d = b == 0 ? f16_at<0>(dw) : f16_at<1>(dw);
                vFloat m = 0.0f;
                if constexpr (FMT == TQ2_0 || FMT == Q2_0 || FMT == TQ1_0)
                {
                    m = -d;
                }
                else if constexpr (FMT == Q1_0)
                {
                    m = -d;
                    d = d * 2.0f;
                }
                sfpi::dst_reg[PS + pidx(a, b, 0)] = d;
                if constexpr (traits<FMT>::linear)
                {
                    sfpi::dst_reg[PM + pidx(a, b, 0)] = m;
                }
            }
        }
        else if constexpr (FMT == Q6_K)
        {
            // Row TI: int8 scales of sub-blocks 2 TI + h (byte 2h + b); field 0: d.
            vUInt r0, r1, r2, r3;
            rep4<FMT>(TI / 4, a, r0, r1, r2, r3);
            const vUInt sw = TI % 4 == 0 ? r0 : TI % 4 == 1 ? r1 : TI % 4 == 2 ? r2 : r3;
#pragma GCC unroll 2
            for (int b = 0; b < 2; b++)
            {
                const vFloat d = pf32<FMT>(0, b, a);
#pragma GCC unroll 2
                for (int h = 0; h < 2; h++)
                {
                    const vFloat s                    = d * (ufloat(bits(sw, 8 * (2 * h + b), 8) ^ 0x80u) + -128.0f);
                    sfpi::dst_reg[PS + pidx(a, b, h)] = s;
                    sfpi::dst_reg[PM + pidx(a, b, h)] = s * -32.0f;
                }
            }
        }
        else if constexpr (FMT == Q4_K || FMT == Q5_K)
        {
            // Tile TI's 6-bit scale and min of each column half, from k4_group's words; d / dmin as fp32.
#pragma GCC unroll 2
            for (int b = 0; b < 2; b++)
            {
                const int w                       = K4 + 4 * (2 * a + b) + TI / 4;
                const vUInt sc                    = bits(ldu(w), 8 * (TI % 4), 6);
                const vUInt mn                    = bits(ldu(w + 2), 8 * (TI % 4), 6);
                sfpi::dst_reg[PS + pidx(a, b, 0)] = pf32<FMT>(0, b, a) * ufloat(sc);
                sfpi::dst_reg[PM + pidx(a, b, 0)] = -(pf32<FMT>(1, b, a) * ufloat(mn));
            }
        }
        else if constexpr (FMT == Q3_K)
        {
            // Rows 0..5: 12 bytes per column holding 16 6-bit scales; field 0: d.
            rep4_to<FMT>(0, a, 0);
            rep4_to<FMT>(1, a, 4);
#pragma GCC unroll 2
            for (int b = 0; b < 2; b++)
            {
                const int o    = 12 * b;
                const vFloat d = pf32<FMT>(0, b, a);
#pragma GCC unroll 2
                for (int h = 0; h < 2; h++)
                {
                    const int i                       = 2 * TI + h;
                    const vUInt ls                    = byte_bits(0, o + i % 8, 4 * (i / 8), 4) | (byte_bits(0, o + 8 + i % 4, 2 * (i / 4), 2) << 4);
                    const vFloat s                    = d * (ufloat(ls) + -32.0f);
                    sfpi::dst_reg[PS + pidx(a, b, h)] = s;
                    sfpi::dst_reg[PM + pidx(a, b, h)] = s * -4.0f;
                }
            }
        }
        else if constexpr (FMT == Q2_K)
        {
            // Row TI: bytes (sc | m << 4) of sub-blocks 2 TI + h at byte 2h + b; fields 0 / 1: d / dmin.
            vUInt r0, r1, r2, r3;
            rep4<FMT>(TI / 4, a, r0, r1, r2, r3);
            const vUInt sw = TI % 4 == 0 ? r0 : TI % 4 == 1 ? r1 : TI % 4 == 2 ? r2 : r3;
#pragma GCC unroll 2
            for (int b = 0; b < 2; b++)
            {
#pragma GCC unroll 2
                for (int h = 0; h < 2; h++)
                {
                    const int k                       = 8 * (2 * h + b);
                    sfpi::dst_reg[PS + pidx(a, b, h)] = pf32<FMT>(0, b, a) * ufloat(bits(sw, k, 4));
                    sfpi::dst_reg[PM + pidx(a, b, h)] = -(pf32<FMT>(1, b, a) * ufloat(bits(sw, k + 4, 4)));
                }
            }
        }
    }
}

// body() of the linear formats but Q8_0 and TQ1_0 through SFPLOADMACRO. Three slots per output: the macro (shift,
// cast, S q + M, store), a shift of the output's field to the top of its word, and a slot without a Simple instruction
// for the macro's Round-unit shift. Under DISABLE_SFPLOADMACRO (ttsim) these formats use body().

template <int FMT>
constexpr bool macro_q4 = FMT == Q4_0 || FMT == Q4_1 || FMT == Q4_K;

template <int FMT>
constexpr bool macro_body = LOADMACRO && traits<FMT>::linear && FMT != Q8_0 && FMT != TQ1_0;

// Instruction templates: 0 = x >> (32 - B) (amount in LReg14), 1 = int to fp32, 2 / 3 = S x + M with S, M in L4, L5 /
// L6, L7. Macro b (column half b) runs the shift, cast, MAD (into LReg16) and store of LReg16 1, 2, 3 and 5
// instructions after it.
template <int FMT>
sfpi_inline void macro_init_linear()
{
    sfpi::vConstIntPrgm2 = mfield<FMT>::B - 32;
    TTI_SFPSHFT2(0, 14, 12, sfpi::SFPSHFT2_MOD1_SHFT_LREG);
    TTI_SFPCAST(0, 13, sfpi::SFPCAST_MOD1_SM32_TO_FP32_RNE);
    TTI_SFPMAD(4, 0, 5, 14, sfpi::SFPMAD_MOD1_OFFSET_NONE);
    TTI_SFPMAD(6, 0, 7, 15, sfpi::SFPMAD_MOD1_OFFSET_NONE);
    constexpr std::uint32_t simple = (2 << 3) | 5;
    constexpr std::uint32_t round  = 0x80 | (1 << 3) | 4;
    constexpr std::uint32_t store  = 0x40 | (5 << 3) | 3;
    constexpr std::uint32_t mad0   = 0xC0 | (3 << 3) | 6;
    TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mad0 << 8) | simple);
    TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store << 8) | round);
    TTI_SFPCONFIG(0, 4, 0);
    TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, ((mad0 + 1) << 8) | simple);
    TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store << 8) | round);
    TTI_SFPCONFIG(0, 5, 0);
    TTI_SFPCONFIG(0xF00, 8, sfpi::SFPCONFIG_MOD1_ARG_IMM); // stores as sfpi's; delays count instructions
    TTI_SFPNOP;
}

// Q4_K with both phases in one call: phase 1's operations (same operands) in a fixed register plan. Column (a, b): the
// 6-bit scale / min of byte TI % 4 of words w, w + 2 (two shifts), cast, S = d sc, M = (-m) mn. a = 0 first: codes in
// L0..L3, d / m into L4 + 2b, L5 + 2b, which keep S / M for body_macro_q4<true>. Then a = 1 in L0..L3, stored.
template <int FMT>
constexpr bool k4_regs = macro_body<FMT> && FMT == Q4_K;

template <int TI>
sfpi_inline void k4_ld(const unsigned r, const int a, const int b, const int v)
{
    TTI_SFPLOAD(r, sfpi::SFPLOAD_MOD0_FMT_INT32, ADDR_MOD_7, 2 * (K4 + 4 * (2 * a + b) + TI / 4 + 2 * v));
}

sfpi_inline void k4_ldf(const unsigned r, const int a, const int b, const int F)
{
    TTI_SFPLOAD(r, 0, ADDR_MOD_7, 2 * pf32_idx<Q4_K>(F, b, a));
}

template <int TI>
sfpi_inline void k4_codes(const unsigned r0, const unsigned n)
{
    constexpr unsigned shl = 26 - 8 * (TI % 4), shr = static_cast<unsigned>(-26) & 0xFFFu;
#pragma GCC unroll 4
    for (unsigned r = r0; r < r0 + n; r++)
    {
        TTI_SFPSHFT(shl, r, r, SHIFT_IMM_OF_C);
    }
#pragma GCC unroll 4
    for (unsigned r = r0; r < r0 + n; r++)
    {
        TTI_SFPSHFT(shr, r, r, SHIFT_IMM_OF_C);
    }
#pragma GCC unroll 4
    for (unsigned r = r0; r < r0 + n; r++)
    {
        TTI_SFPCAST(r, r, sfpi::SFPCAST_MOD1_SM32_TO_FP32_RNE);
    }
}

// The body's first macro must start six instructions after the last SFPMUL (with four, some Blackhole dies decoded
// wrong tiles): three stores and two NOPs here, then the body's first load.
template <int TI>
sfpi_inline void k4_params_regs()
{
#pragma GCC unroll 2
    for (int b = 0; b < 2; b++)
    {
        k4_ld<TI>(2 * b, 0, b, 0);
        k4_ld<TI>(2 * b + 1, 0, b, 1);
    }
    k4_codes<TI>(0, 4);
#pragma GCC unroll 2
    for (int b = 0; b < 2; b++)
    {
        k4_ldf(4 + 2 * b, 0, b, 0);
        k4_ldf(5 + 2 * b, 0, b, 1);
    }
#pragma GCC unroll 2
    for (unsigned b = 0; b < 2; b++)
    {
        TTI_SFPMUL(4 + 2 * b, 2 * b, p_sfpu::LCONST_0, 4 + 2 * b, 0);
        TTI_SFPMUL(5 + 2 * b, 2 * b + 1, p_sfpu::LCONST_0, 5 + 2 * b, 1);
    }
    // a = 1, b = 0 in L0, L1 (d / m in L2, L3). b = 1's codes in L2, L3, its d and m in turn in L0 (freed by b = 0's
    // S store).
    k4_ld<TI>(0, 1, 0, 0);
    k4_ld<TI>(1, 1, 0, 1);
    k4_codes<TI>(0, 2);
    k4_ldf(2, 1, 0, 0);
    k4_ldf(3, 1, 0, 1);
    TTI_SFPMUL(2, 0, p_sfpu::LCONST_0, 0, 0);
    TTI_SFPMUL(3, 1, p_sfpu::LCONST_0, 1, 1);
    k4_ld<TI>(2, 1, 1, 0);
    k4_ld<TI>(3, 1, 1, 1);
    constexpr unsigned shl = 26 - 8 * (TI % 4), shr = static_cast<unsigned>(-26) & 0xFFFu;
    TTI_SFPSHFT(shl, 2, 2, SHIFT_IMM_OF_C);
    TTI_SFPSHFT(shl, 3, 3, SHIFT_IMM_OF_C);
    TTI_SFPSTORE(0, 0, ADDR_MOD_7, 2 * (PS + pidx(1, 0, 0)));
    TTI_SFPSHFT(shr, 2, 2, SHIFT_IMM_OF_C);
    TTI_SFPSHFT(shr, 3, 3, SHIFT_IMM_OF_C);
    k4_ldf(0, 1, 1, 0);
    TTI_SFPCAST(2, 2, sfpi::SFPCAST_MOD1_SM32_TO_FP32_RNE);
    TTI_SFPMUL(0, 2, p_sfpu::LCONST_0, 2, 0);
    k4_ldf(0, 1, 1, 1);
    TTI_SFPCAST(3, 3, sfpi::SFPCAST_MOD1_SM32_TO_FP32_RNE);
    TTI_SFPMUL(0, 3, p_sfpu::LCONST_0, 3, 1);
    TTI_SFPSTORE(1, 0, ADDR_MOD_7, 2 * (PM + pidx(1, 0, 0)));
    TTI_SFPSTORE(2, 0, ADDR_MOD_7, 2 * (PS + pidx(1, 1, 0)));
    TTI_SFPSTORE(3, 0, ADDR_MOD_7, 2 * (PM + pidx(1, 1, 0)));
    TTI_SFPNOP;
    TTI_SFPNOP;
}

// Q4_0, Q4_1 and Q4_K: field i of code word 2h + a (L3) through macro i % 2, v in L0..L2 in turn; third slots load
// the next code word or S, M. IN_REGS: a = 0's S / M are already in L4..L7 (k4_params_regs).
template <bool IN_REGS = false>
sfpi_inline void body_macro_q4()
{
    TTI_SFPLOAD(3, sfpi::SFPLOAD_MOD0_FMT_INT32, ADDR_MOD_7, 2 * CO);
    if constexpr (!IN_REGS)
    {
        TTI_SFPLOAD(4, sfpi::SFPLOAD_MOD0_FMT_SRCB, ADDR_MOD_7, 2 * (PS + pidx(0, 0, 0)));
        TTI_SFPLOAD(5, sfpi::SFPLOAD_MOD0_FMT_SRCB, ADDR_MOD_7, 2 * (PM + pidx(0, 0, 0)));
        TTI_SFPLOAD(6, sfpi::SFPLOAD_MOD0_FMT_SRCB, ADDR_MOD_7, 2 * (PS + pidx(0, 1, 0)));
        TTI_SFPLOAD(7, sfpi::SFPLOAD_MOD0_FMT_SRCB, ADDR_MOD_7, 2 * (PM + pidx(0, 1, 0)));
    }
#pragma GCC unroll 2
    for (int a = 0; a < 2; a++)
    {
#pragma GCC unroll 2
        for (int h = 0; h < 2; h++)
        {
#pragma GCC unroll 8
            for (int i = 0; i < 8; i++)
            {
                const int v = (16 * a + 8 * h + i) % 3;
                TTI_SFPLOADMACRO(((i % 2) << 2) | v, LOAD_NONE, ADDR_MOD_7, 2 * (16 * h + 8 * a + i));
                TTI_SFPSHFT(28 - 4 * i, 3, v, SHIFT_IMM_OF_C);
                if (i == 7 && (h == 0 || a == 0))
                {
                    TTI_SFPLOAD(3, sfpi::SFPLOAD_MOD0_FMT_INT32, ADDR_MOD_7, 2 * (CO + (h == 0 ? 2 + a : 1)));
                }
                else if (a == 1 && h == 0 && i < 2)
                {
                    TTI_SFPLOAD(6 + i, sfpi::SFPLOAD_MOD0_FMT_SRCB, ADDR_MOD_7, 2 * ((i == 0 ? PS : PM) + pidx(1, 1, 0)));
                }
                else
                {
                    TTI_SFPNOP;
                }
            }
            if (a == 0 && h == 1)
            {
                TTI_SFPLOAD(4, sfpi::SFPLOAD_MOD0_FMT_SRCB, ADDR_MOD_7, 2 * (PS + pidx(1, 0, 0)));
                TTI_SFPLOAD(5, sfpi::SFPLOAD_MOD0_FMT_SRCB, ADDR_MOD_7, 2 * (PM + pidx(1, 0, 0)));
            }
        }
    }
#pragma GCC unroll 4
    for (int n = 0; n < 4; n++)
    {
        TTI_SFPNOP;
    }
}

// The other macro formats: an issue stream planned at compile time (make_mplan) in stream order k; loads take free
// third slots or extra slots. v alternates L0 / L1, code words L3 / L2, S / M of column half b in L4 + 2b / L5 + 2b.

enum : int
{
    MS_NOP,
    MS_MACRO, // macro x (column half), v = LReg y, row z
    MS_SHIFT, // v x = word register y << z
    MS_LOAD,  // register x, SFPLOAD mode y, row z
};

enum : int
{
    MT_WORD,
    MT_PARAM,
};

enum : int
{
    MA_NONE,
    MA_SHIFT, // stream position dep's shift has run
    MA_MAD,   // stream position dep's MAD has run
};

struct mtask
{
    int kind, k, reg, row, after, dep, late, done;
};

constexpr int MSLOTS = 144;

struct mplan
{
    mslot s[MSLOTS];
    int n;
};

// Task x may run in slot t: its register is free.
constexpr bool mready(const mtask& x, const int* T, const int t)
{
    switch (x.after)
    {
        case MA_SHIFT:
            return T[x.dep] >= 0 && t > T[x.dep] + 1;
        case MA_MAD:
            return T[x.dep] >= 0 && t > T[x.dep] + 4;
        default:
            return true;
    }
}

// Last slot task x may take if position knext's macro issues in slot tnext (3 slots per position).
constexpr int mdeadline(const mtask& x, const int tnext, const int knext)
{
    return tnext + 3 * (x.k - knext) + (x.late ? 2 : -1);
}

// The ready task with the earliest deadline (S / M loads first on ties, else in creation order); -1 if none.
constexpr int mpick(const mtask* tk, const int nt, const int* T, const int t, const int knext)
{
    int best = -1;
    for (int i = 0; i < nt; i++)
    {
        if (tk[i].done >= 0 || !mready(tk[i], T, t))
        {
            continue;
        }
        if (best < 0)
        {
            best = i;
            continue;
        }
        const int d  = mdeadline(tk[i], t + 1, knext);
        const int db = mdeadline(tk[best], t + 1, knext);
        if (d < db || (d == db && tk[i].kind == MT_PARAM && tk[best].kind != MT_PARAM))
        {
            best = i;
        }
    }
    return best;
}

template <int FMT, int CP>
constexpr mplan make_mplan()
{
    using M = mfield<FMT>;
    mplan p {};
    mtask tk[48] {};
    int nt = 0;
    int wreg[32] {};
    // Code words, one load per change of word, alternating L3 / L2.
    for (int k = 0, r = 2; k < M::clean; k++)
    {
        if (k == 0 || mword<FMT, CP>(k) != mword<FMT, CP>(k - 1))
        {
            r       = 5 - r;
            int dep = -1;
            for (int j = 0; j < k; j++)
            {
                dep = wreg[j] == r ? j : dep;
            }
            tk[nt++] = {MT_WORD, k, r, CO + mword<FMT, CP>(k), dep < 0 ? MA_NONE : MA_SHIFT, dep, 0, -1};
        }
        wreg[k] = r;
    }
    // S, M of column half b = k % 2 whenever its scale group (a, h) changes.
    for (int k = 0, last[2] = {-1, -1}, group[2] = {-1, -1}; k < M::clean; k++)
    {
        const int b = k % 2, a = k / 16, h = traits<FMT>::has_h ? k / 8 % 2 : 0;
        if (group[b] != 2 * a + h)
        {
            const int after = last[b] < 0 ? MA_NONE : MA_MAD;
            tk[nt++]        = {MT_PARAM, k, 4 + 2 * b, PS + pidx(a, b, h), after, last[b], 0, -1};
            tk[nt++]        = {MT_PARAM, k, 5 + 2 * b, PM + pidx(a, b, h), after, last[b], 1, -1};
        }
        last[b]  = k;
        group[b] = 2 * a + h;
    }
    int T[32] {};
    for (int k = 0; k < 32; k++)
    {
        T[k] = -1;
    }
    const auto issue = [&](const int i)
    {
        if (i < 0)
        {
            p.s[p.n++] = {MS_NOP, 0, 0, 0};
            return;
        }
        tk[i].done     = p.n;
        const int mode = tk[i].kind == MT_WORD ? sfpi::SFPLOAD_MOD0_FMT_INT32 : sfpi::SFPLOAD_MOD0_FMT_SRCB;
        p.s[p.n++]     = {MS_LOAD, tk[i].reg, mode, tk[i].row};
    };
    for (int k = 0; k < M::clean; k++)
    {
        // Extra slots until every task of position k but its second S / M loads has run.
        for (;;)
        {
            bool due = false;
            for (int i = 0; i < nt; i++)
            {
                due = due || (tk[i].done < 0 && tk[i].k <= k && !tk[i].late);
            }
            if (!due)
            {
                break;
            }
            issue(mpick(tk, nt, T, p.n, k));
        }
        T[k]       = p.n;
        p.s[p.n++] = {MS_MACRO, k % 2, k % 2, mrow(k)};
        p.s[p.n++] = {MS_SHIFT, k % 2, wreg[k], 32 - M::B - mlsb<FMT>(k)};
        issue(mpick(tk, nt, T, p.n, k + 1));
    }
    for (int i = 0; i < 4; i++)
    {
        p.s[p.n++] = {MS_NOP, 0, 0, 0};
    }
    return p;
}

template <int FMT, int CP>
constexpr mplan mplan_v = make_mplan<FMT, CP>();

template <int FMT, int CP, int I>
sfpi_inline void mslot_issue()
{
    constexpr mslot s = mplan_v<FMT, CP>.s[I];
    if constexpr (s.kind == MS_MACRO)
    {
        TTI_SFPLOADMACRO((s.x << 2) | s.y, LOAD_NONE, ADDR_MOD_7, 2 * s.z);
    }
    else if constexpr (s.kind == MS_SHIFT)
    {
        TTI_SFPSHFT(s.z, s.y, s.x, SHIFT_IMM_OF_C);
    }
    else if constexpr (s.kind == MS_LOAD)
    {
        TTI_SFPLOAD(s.x, s.y, ADDR_MOD_7, 2 * s.z);
    }
    else
    {
        TTI_SFPNOP;
    }
}

template <int FMT, int CP, int... I>
sfpi_inline void mslots_issue(std::integer_sequence<int, I...>)
{
    (mslot_issue<FMT, CP, I>(), ...);
}

// The split rows (B not dividing 32) through body()'s arithmetic, then the planned stream.
template <int FMT, int CP>
sfpi_inline void body_macro()
{
    using M = mfield<FMT>;
#pragma GCC unroll 2
    for (int k = M::clean; k < 32; k++)
    {
        const int o      = mrow(k);
        const int pi     = pidx(o / 8 % 2, o % 2, traits<FMT>::has_h ? o / 16 : 0);
        const vFloat x   = ufloat(code_of<FMT, CP>(o));
        sfpi::dst_reg[o] = x * sfpi::dst_reg[PS + pi] + sfpi::dst_reg[PM + pi];
    }
    mslots_issue<FMT, CP>(std::make_integer_sequence<int, mplan_v<FMT, CP>.n> {});
}

// TQ1_0: one page holds the group's eight K tiles as GGUF bytes. Byte j of lane (r, c): word-vector 4a + 2h + b
// (tiles 0..4): qs[16h + 4j + r] of column 16a + 2c + b; word-vector 8 + 2a + b (tiles 5..7): qs[32 + 4j + r]; scale
// rows 0 / 1, word 2c + a: qh of columns (a, 0) / (a, 1); row 2: d. Output o = 16h + 8a + 2j + b of tile TI is trit D
// of its byte: D = TI for tiles 0..4, 2 (TI - 5) + h for tiles 5, 6 and 4 for h = 0 of tile 7; h = 1 of tile 7 is trit
// j of qh[r]. Value = trit * d - d.
constexpr int QH = 2 * TILE; // tile 7: qh[r] of column (a, b) at QH + 2a + b
constexpr int NQ = QH + 4;   // -8r per lane

// Trit D of the bytes in bits 0..7 and 16..23 of x, at bits 8..9 and 24..25 (x 3^D < 2^16: no carry between halves).
// Programmable constant 0 = 0x00FF00FF.
template <int D>
sfpi_inline vUInt tq1_0_trits(vUInt x)
{
    if constexpr (D % 2 == 1)
    {
        x = x + (x << 1u);
    }
    if constexpr (D >= 2)
    {
        x = x + (x << 3u);
    }
    if constexpr (D == 4)
    {
        x = x + (x << 3u);
    }
    if constexpr (D > 0)
    {
        x = x & sfpi::as<vUInt>(sfpi::vConstIntPrgm0);
    }
    return x + (x << 1u);
}

// Outputs o0 + 2j, j = 0..3: trit D of byte j of w.
template <int D>
sfpi_inline void tq1_0_word(const vUInt w, const int o0, const vFloat S, const vFloat M)
{
    const vUInt lo        = tq1_0_trits<D>(w & sfpi::as<vUInt>(sfpi::vConstIntPrgm0)); // bytes 0, 2
    const vFloat v0       = ufloat(bits(lo, 8, 2)) * S + M;
    const vFloat v2       = ufloat(lo >> 24u) * S + M;
    sfpi::dst_reg[o0]     = v0;
    sfpi::dst_reg[o0 + 4] = v2;
    const vUInt hi        = tq1_0_trits<D>((w >> 8u) & sfpi::as<vUInt>(sfpi::vConstIntPrgm0)); // bytes 1, 3
    const vFloat v1       = ufloat(bits(hi, 8, 2)) * S + M;
    const vFloat v3       = ufloat(hi >> 24u) * S + M;
    sfpi::dst_reg[o0 + 2] = v1;
    sfpi::dst_reg[o0 + 6] = v3;
}

template <int TI>
sfpi_inline void tq1_0_body()
{
    sfpi::vConstIntPrgm0 = 0x00FF00FF;
    if constexpr (TI == 7)
    {
        // qh[r] of column (a, b): scale row b of word 2c + a to every lane, its byte r.
        sfpi::dst_reg[NQ].mode<sfpi::DataLayout::U32>() = sfpi::as<vUInt>(vInt(0) - sfpi::as<vInt>(sfpi::as<vUInt>(lane_row()) << 3u));
#pragma GCC unroll 2
        for (int a = 0; a < 2; a++)
        {
            vUInt r0, r1, r2, r3;
            rep4<TQ1_0>(0, a, r0, r1, r2, r3);
            sfpi::dst_reg[QH + 2 * a].mode<sfpi::DataLayout::U32>()     = (shift_after_transp(r0, NQ) << 24u) >> 24u;
            sfpi::dst_reg[QH + 2 * a + 1].mode<sfpi::DataLayout::U32>() = (shift_after_transp(r1, NQ) << 24u) >> 24u;
        }
    }
#pragma GCC unroll 2
    for (int a = 0; a < 2; a++)
    {
#pragma GCC unroll 2
        for (int b = 0; b < 2; b++)
        {
            const vFloat S = sfpi::dst_reg[PS + pidx(a, b, 0)];
            const vFloat M = sfpi::dst_reg[PM + pidx(a, b, 0)];
            const int o0   = 8 * a + b;
            if constexpr (TI < 5)
            {
                tq1_0_word<TI>(ldu(CO + 4 * a + b), o0, S, M);
                tq1_0_word<TI>(ldu(CO + 4 * a + 2 + b), o0 + 16, S, M);
            }
            else
            {
                tq1_0_word<2 * (TI - 5)>(ldu(CO + 8 + 2 * a + b), o0, S, M);
                if constexpr (TI < 7)
                {
                    tq1_0_word<2 * TI - 9>(ldu(CO + 8 + 2 * a + b), o0 + 16, S, M);
                }
                else
                {
                    vUInt x = ldu(QH + 2 * a + b);
#pragma GCC unroll 4
                    for (int j = 0; j < 4; j++)
                    {
                        const vUInt t                  = x + (x << 1u);
                        sfpi::dst_reg[o0 + 16 + 2 * j] = ufloat(t >> 8u) * S + M;
                        x                              = t & sfpi::as<vUInt>(sfpi::vConstIntPrgm0);
                    }
                }
            }
        }
    }
}

} // namespace gguf
} // namespace ckernel::sfpu
