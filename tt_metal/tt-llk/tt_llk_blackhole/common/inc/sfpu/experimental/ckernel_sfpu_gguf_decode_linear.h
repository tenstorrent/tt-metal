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
// (bits() drops them). Rows 6, 7 (fp16 d, dmin) are in the page as stored: k4_group converts them and stores them
// where pf32 reads them, past the page.
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
    // The masks in programmable constants 0 and 1 (the macro bodies use only constant 2).
    sfpi::vConstIntPrgm0 = 0x0F0F0F0F;
    sfpi::vConstIntPrgm1 = 0x30303030;
    const auto lo4       = sfpi::as<vUInt>(sfpi::vConstIntPrgm0);
    const auto hi2       = sfpi::as<vUInt>(sfpi::vConstIntPrgm1);
#pragma GCC unroll 2
    for (int a = 0; a < 2; a++)
    {
        const int w0 = K4 + 8 * a, w1 = w0 + 4;
        // Scale bytes 0..15 (rows 0..3, r): column half 0's scales of tiles 0..3, mins of 0..3, low / high nibbles of
        // tiles 4..7; column half 1's scales of tiles 0..3. Bytes 16..23 (rows 4, 5, q): column half 1's mins of tiles
        // 0..3, nibbles of tiles 4..7. One SFPTRANSP for both (LRegs 0..3 and 4..7).
        vUInt r0 = ldu(src + a), r1 = r0, r2 = r0, r3 = r0;
        vUInt q0 = ldu(src + 2 + a), q1 = q0, q2 = q0, q3 = q0;
        sfpi::subvec_transp(r0, r1, r2, r3, q0, q1, q2, q3);

        // Rows 6, 7 (q2, q3: fp16 d and dmin, column half b in bits 16b) wait where column half 1's fp32 values go.
        sfpi::dst_reg[pf32_idx<FMT>(0, 1, a)].template mode<sfpi::DataLayout::U32>() = q2;
        sfpi::dst_reg[pf32_idx<FMT>(1, 1, a)].template mode<sfpi::DataLayout::U32>() = q3;

        sfpi::dst_reg[w0 + 0].mode<sfpi::DataLayout::U32>() = r0;
        sfpi::dst_reg[w0 + 2].mode<sfpi::DataLayout::U32>() = r1;
        sfpi::dst_reg[w1 + 0].mode<sfpi::DataLayout::U32>() = r3;
        sfpi::dst_reg[w1 + 2].mode<sfpi::DataLayout::U32>() = q0;
        sfpi::dst_reg[w0 + 1].mode<sfpi::DataLayout::U32>() = (r2 & lo4) | ((r0 >> 2u) & hi2);
        sfpi::dst_reg[w0 + 3].mode<sfpi::DataLayout::U32>() = ((r2 >> 4u) & lo4) | ((r1 >> 2u) & hi2);
        sfpi::dst_reg[w1 + 1].mode<sfpi::DataLayout::U32>() = ((r3 >> 2u) & hi2) | (q1 & lo4);
        sfpi::dst_reg[w1 + 3].mode<sfpi::DataLayout::U32>() = ((q1 >> 4u) & lo4) | ((q0 >> 2u) & hi2);

#pragma GCC unroll 2
        for (int F = 0; F < 2; F++)
        {
            const vUInt w                         = ldu(pf32_idx<FMT>(F, 1, a));
            sfpi::dst_reg[pf32_idx<FMT>(F, 0, a)] = k4_f16<0>(w);
            sfpi::dst_reg[pf32_idx<FMT>(F, 1, a)] = k4_f16<1>(w);
        }
    }
}

// Q2_K's group step: rows 8, 9 (fp16 d and dmin) to fp32, where pf32 reads them, past the page.
template <int FMT>
sfpi_inline void q2_group()
{
#pragma GCC unroll 2
    for (int a = 0; a < 2; a++)
    {
        vUInt r0, r1, r2, r3;
        rep4<FMT>(2, a, r0, r1, r2, r3);
        sfpi::dst_reg[pf32_idx<FMT>(0, 0, a)] = k4_f16<0>(r0);
        sfpi::dst_reg[pf32_idx<FMT>(0, 1, a)] = k4_f16<1>(r0);
        sfpi::dst_reg[pf32_idx<FMT>(1, 0, a)] = k4_f16<0>(r1);
        sfpi::dst_reg[pf32_idx<FMT>(1, 1, a)] = k4_f16<1>(r1);
    }
}

// Q3_K's group step: the 16 6-bit scales of each column half, from the 12 scale bytes in rows 0..2 (column half 0)
// and 3..5 (half 1), as bytes into scratch entries Q3 .. Q3 + 15: word Q3 + 4 (2a + b) + j, byte t: scale 4j + t of
// column half b, word parity a. Row 6 (fp16 d) to fp32, where pf32 reads it, past the page.
constexpr int Q3 = 2 * TILE;

// Masks 0x0F0F0F0F, 0x03030303 in programmable constants 0, 1 (q3_group).
template <unsigned LS, unsigned HS>
sfpi_inline vUInt q3_word(const vUInt lo, const vUInt hi)
{
    const vUInt l = LS ? lo >> LS : lo;
    const vUInt h = HS ? hi >> HS : hi;
    return (l & sfpi::as<vUInt>(sfpi::vConstIntPrgm0)) | ((h & sfpi::as<vUInt>(sfpi::vConstIntPrgm1)) << 4u);
}

template <int FMT>
sfpi_inline void q3_group()
{
    sfpi::vConstIntPrgm0 = 0x0F0F0F0F;
    sfpi::vConstIntPrgm1 = 0x03030303;
#pragma GCC unroll 2
    for (int a = 0; a < 2; a++)
    {
        const int w0 = Q3 + 8 * a, w1 = w0 + 4;
        vUInt r0, r1, r2, r3;
        rep4<FMT>(0, a, r0, r1, r2, r3);
        sfpi::dst_reg[w0 + 0].mode<sfpi::DataLayout::U32>() = q3_word<0, 0>(r0, r2);
        sfpi::dst_reg[w0 + 1].mode<sfpi::DataLayout::U32>() = q3_word<0, 2>(r1, r2);
        sfpi::dst_reg[w0 + 2].mode<sfpi::DataLayout::U32>() = q3_word<4, 4>(r0, r2);
        sfpi::dst_reg[w0 + 3].mode<sfpi::DataLayout::U32>() = q3_word<4, 6>(r1, r2);
        sfpi::dst_reg[w1 + 0].mode<sfpi::DataLayout::U32>() = r3;  // row 3, until the transpose below is done
        vUInt q0, q1, q2, q3;
        rep4<FMT>(1, a, q0, q1, q2, q3);
        const vUInt l0 = ldu(w1 + 0);
        sfpi::dst_reg[w1 + 0].mode<sfpi::DataLayout::U32>() = q3_word<0, 0>(l0, q1);
        sfpi::dst_reg[w1 + 1].mode<sfpi::DataLayout::U32>() = q3_word<0, 2>(q0, q1);
        sfpi::dst_reg[w1 + 2].mode<sfpi::DataLayout::U32>() = q3_word<4, 4>(l0, q1);
        sfpi::dst_reg[w1 + 3].mode<sfpi::DataLayout::U32>() = q3_word<4, 6>(q0, q1);
        sfpi::dst_reg[pf32_idx<FMT>(0, 0, a)] = k4_f16<0>(q2);
        sfpi::dst_reg[pf32_idx<FMT>(0, 1, a)] = k4_f16<1>(q2);
    }
}

// fp16 in bits 16..31 of v (the low bits any) to fp32, exact for every finite value (as f16_at).
sfpi_inline vFloat f16_top(const vUInt v)
{
    const vUInt m13 = ((v >> 16u) << 17u) >> 4u;
    const vFloat b  = sfpi::as<vFloat>(m13 + sfpi::as<vUInt>(sfpi::vConstIntPrgm0));
    const vFloat a  = sfpi::addexp(b, 1) + -6.103515625e-05f;
    return sfpi::copysgn(sfpi::min_max(a, b).first, sfpi::as<vInt>(v));
}

// f16_top of v and w, the two chains interleaved.
sfpi_inline void f16_top_both(const vUInt v, const vUInt w, vFloat& x, vFloat& y)
{
    vUInt mv        = v >> 16u;
    vUInt mw        = w >> 16u;
    mv              = mv << 17u;
    mw              = mw << 17u;
    mv              = mv >> 4u;
    mw              = mw >> 4u;
    const vFloat bv = sfpi::as<vFloat>(mv + sfpi::as<vUInt>(sfpi::vConstIntPrgm0));
    const vFloat bw = sfpi::as<vFloat>(mw + sfpi::as<vUInt>(sfpi::vConstIntPrgm0));
    vFloat av       = sfpi::addexp(bv, 1);
    vFloat aw       = sfpi::addexp(bw, 1);
    av              = av + -6.103515625e-05f;
    aw              = aw + -6.103515625e-05f;
    x               = sfpi::copysgn(sfpi::min_max(av, bv).first, sfpi::as<vInt>(v));
    y               = sfpi::copysgn(sfpi::min_max(aw, bw).first, sfpi::as<vInt>(w));
}

template <int F>
sfpi_inline vUInt pick4(const vUInt l0, const vUInt l1, const vUInt l2, const vUInt l3)
{
    return F == 0 ? l0 : F == 1 ? l1 : F == 2 ? l2 : l3;
}

// Phase 1 of Q8_0, Q4_0, Q5_0, Q4_1 and Q5_1: the fp16 fields of scale rows 0..3 (d in row 0; Q4_1 m in row 1; Q5_0 d
// in row TI; Q5_1 d, m in rows 2 TI, 2 TI + 1) with one conversion per field for all four column groups. The scale
// words of parity a, column half b's field at the top (shifted for b = 0), are LRegs 2a + b of a transpose, which puts
// field row f of column group g = 2a + b in row group g of LReg f. After the conversion, a transpose of four copies of
// S (and of M) broadcasts row group g to every lane.
template <int FMT, int TI>
sfpi_inline void q40_params_regs()
{
    constexpr int w = CO + scale_row<FMT>() / 2;
    vUInt l1        = ldu(w);
    vUInt l3        = ldu(w + 1);
    vUInt l0        = l1 << 16u;
    vUInt l2        = l3 << 16u;
    vUInt l4, l5, l6, l7; // the transpose of LRegs 4..7 is not used
    sfpi::subvec_transp(l0, l1, l2, l3, l4, l5, l6, l7);
    constexpr int fd = FMT == Q5_0 ? TI % 2 : FMT == Q5_1 ? 2 * (TI % 2) : 0;
    vFloat s0, m0;
    if constexpr (FMT == Q4_1 || FMT == Q5_1)
    {
        f16_top_both(pick4<fd>(l0, l1, l2, l3), pick4<fd + 1>(l0, l1, l2, l3), s0, m0);
    }
    else
    {
        constexpr float c = FMT == Q8_0 ? -128.0f : FMT == Q4_0 ? -8.0f : -16.0f;
        s0                = f16_top(pick4<fd>(l0, l1, l2, l3));
        m0                = s0 * c;
    }
    vFloat s1 = s0, s2 = s0, s3 = s0;
    vFloat m1 = m0, m2 = m0, m3 = m0;
    sfpi::subvec_transp(s0, s1, s2, s3, m0, m1, m2, m3);
    sfpi::dst_reg[PS + pidx(0, 0, 0)] = s0;
    sfpi::dst_reg[PS + pidx(0, 1, 0)] = s1;
    sfpi::dst_reg[PS + pidx(1, 0, 0)] = s2;
    sfpi::dst_reg[PS + pidx(1, 1, 0)] = s3;
    sfpi::dst_reg[PM + pidx(0, 0, 0)] = m0;
    sfpi::dst_reg[PM + pidx(0, 1, 0)] = m1;
    sfpi::dst_reg[PM + pidx(1, 0, 0)] = m2;
    sfpi::dst_reg[PM + pidx(1, 1, 0)] = m3;
}

// The same phase 1 by column parity a: the scale rows of parity a replicated to all row groups, then both columns'
// conversions interleaved (f16_both). ONE_TRANSP (Q4_0): one transpose replicates both parities' rows.
template <int FMT, int TI, bool ONE_TRANSP>
sfpi_inline void q40_params_rep()
{
    constexpr float c = FMT == Q8_0 ? -128.0f : FMT == Q4_0 ? -8.0f : -16.0f;
    if constexpr (ONE_TRANSP)
    {
        static_assert(FMT == Q4_0);
        vUInt r0 = ldu(CO + scale_row<FMT>() / 2), r1 = r0, r2 = r0, r3 = r0;
        vUInt q0 = ldu(CO + scale_row<FMT>() / 2 + 1), q1 = q0, q2 = q0, q3 = q0;
        sfpi::subvec_transp(r0, r1, r2, r3, q0, q1, q2, q3);
#pragma GCC unroll 2
        for (int a = 0; a < 2; a++)
        {
            vFloat d0, d1;
            f16_both(a == 0 ? r0 : q0, d0, d1);
            sfpi::dst_reg[PS + pidx(a, 0, 0)] = d0;
            sfpi::dst_reg[PS + pidx(a, 1, 0)] = d1;
            d0                                = d0 * c;
            d1                                = d1 * c;
            sfpi::dst_reg[PM + pidx(a, 0, 0)] = d0;
            sfpi::dst_reg[PM + pidx(a, 1, 0)] = d1;
        }
        return;
    }
#pragma GCC unroll 2
    for (int a = 0; a < 2; a++)
    {
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
            sfpi::dst_reg[PS + pidx(a, 0, 0)] = d0;
            sfpi::dst_reg[PS + pidx(a, 1, 0)] = d1;
            d0                                = d0 * c;
            d1                                = d1 * c;
            sfpi::dst_reg[PM + pidx(a, 0, 0)] = d0;
            sfpi::dst_reg[PM + pidx(a, 1, 0)] = d1;
        }
    }
}

// Phase 1 variant of the q40_params formats: 2 = q40_params_regs, 1 = q40_params_rep with one transpose, 0 = by
// parity. Beside an unpack (the pack-thread decode that overlaps the matmul) phase 1 also paces the weight reads:
// there the variants are those whose tile loop keeps every shape at least as fast.
template <int FMT, bool BESIDE_UNPACK>
constexpr int q40_params_kind = !BESIDE_UNPACK || FMT == Q5_0 ? 2
                                : FMT == Q4_0                 ? 1
                                                              : 0;

// Phase 1 of the linear formats. TI = tile index inside the scale page's group of K-tiles.
template <int FMT, int TI, bool BESIDE_UNPACK>
sfpi_inline void linear_params()
{
    if constexpr (page_layout<static_cast<Format>(FMT)>().fp16_rows == 0 && !group_scales<FMT>)
    {
        params_init<!q40_params<FMT>>();
    }
    if constexpr (q40_params<FMT>)
    {
        constexpr int kind = q40_params_kind<FMT, BESIDE_UNPACK>;
        if constexpr (kind == 2)
        {
            q40_params_regs<FMT, TI>();
        }
        else
        {
            q40_params_rep<FMT, TI, kind == 1>();
        }
        return;
    }
#pragma GCC unroll 2
    for (int a = 0; a < 2; a++)
    {
        if constexpr (FMT == TQ2_0 || FMT == Q2_0 || FMT == Q1_0 || FMT == TQ1_0)
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
            // Sub-blocks 2 TI + h's 6-bit scales from q3_group's words; d as fp32.
#pragma GCC unroll 2
            for (int b = 0; b < 2; b++)
            {
                const vFloat d = pf32<FMT>(0, b, a);
                const vUInt w  = ldu(Q3 + 4 * (2 * a + b) + TI / 2);
#pragma GCC unroll 2
                for (int h = 0; h < 2; h++)
                {
                    const int i                       = 2 * TI + h;
                    const vUInt ls                    = bits(w, 8 * (i % 4), 6);
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

// body() of the linear formats but Q8_0 and TQ1_0 through SFPLOADMACRO: per output a shift of its field to the top of
// a word (v) and a macro (shift right, cast, S v + M, store). Under DISABLE_SFPLOADMACRO (ttsim) these formats use
// body(). Two variants: CYC (delays count cycles, stores arbitrated against the unpackers, may run beside an unpack
// to DEST) and instruction-counted (shorter streams, nothing may write DEST beside them).

template <int FMT>
constexpr bool macro_q4 = FMT == Q4_0 || FMT == Q4_1 || FMT == Q4_K;

template <int FMT>
constexpr bool macro_body = LOADMACRO && traits<FMT>::linear && FMT != Q8_0 && FMT != TQ1_0;

// Instruction templates: 0 = v >> (32 - B) (amount in LReg14), 1 = int to fp32, 2 / 3 = S v + M with S, M in L4, L5 /
// L6, L7. Macro b (column half b) runs the shift, cast, MAD (into LReg16) and store of LReg16 1, 2, 3 and 5 cycles
// after it (CYC), or 1, 2, 3 and 5 instructions after it. Blackhole arbitrates a macro's DEST write against the
// unpackers' only when it lands within 7 cycles of the macro, which delays that count instructions do not guarantee.
template <int FMT, bool CYC>
sfpi_inline void macro_init_linear()
{
    sfpi::vConstIntPrgm2 = mfield<FMT>::B - 32;
    TTI_SFPSHFT2(0, 14, 12, sfpi::SFPSHFT2_MOD1_SHFT_LREG);
    TTI_SFPCAST(0, 13, sfpi::SFPCAST_MOD1_SM32_TO_FP32_RNE);
    TTI_SFPMAD(4, 0, 5, 14, sfpi::SFPMAD_MOD1_OFFSET_NONE);
    TTI_SFPMAD(6, 0, 7, 15, sfpi::SFPMAD_MOD1_OFFSET_NONE);
    constexpr std::uint32_t d      = CYC ? 0 : 1; // a delay of d counts d + 1 cycles, or d instructions
    constexpr std::uint32_t simple = ((1 + d) << 3) | 5;
    constexpr std::uint32_t round  = 0x80 | (d << 3) | 4;
    constexpr std::uint32_t store  = 0x40 | ((4 + d) << 3) | 3;
    constexpr std::uint32_t mad0   = 0xC0 | ((2 + d) << 3) | 6;
    TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mad0 << 8) | simple);
    TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store << 8) | round);
    TTI_SFPCONFIG(0, 4, 0);
    TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, ((mad0 + 1) << 8) | simple);
    TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store << 8) | round);
    TTI_SFPCONFIG(0, 5, 0);
    TTI_SFPCONFIG(CYC ? 0 : 0xF00, 8, sfpi::SFPCONFIG_MOD1_ARG_IMM); // stores as sfpi's
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
// wrong tiles): three stores here, then two NOPs and the body's first load, or (TAIL = false) a body whose first
// macro is in its fourth slot or later.
template <int TI, bool TAIL = true>
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
    if constexpr (TAIL)
    {
        TTI_SFPNOP;
        TTI_SFPNOP;
    }
}

// The issue stream of the macro bodies, planned at compile time (make_mplan). A stall of the issue (a DEST access
// losing arbitration) delays later instructions but not the scheduled operations of earlier macros, so the plan holds
// without assuming issue timing: every input of a macro is written before the macro, an instruction that overwrites
// one comes after the macro's read, and no Simple / Round instruction (one LReg write port) is issued between a macro
// and its cast. Stream position j decodes output position k = order[j]; v of position j is LReg j % NV; code words in
// L3 (NV = 3) or alternating L3 / L2 (NV = 2); S / M of column half b in L4 + 2b / L5 + 2b.

enum : int
{
    MS_NOP,
    MS_MACRO, // macro x (column half), v = LReg y, row z
    MS_SHIFT, // v x = word register y << z
    MS_LOAD,  // register x, SFPLOAD mode y, row z
};

// The macro's load replaces only the low 16 bits of v (the shifted field stays in the top bits).
constexpr std::uint32_t LOAD_LO16_ONLY = 14;

// Cycles after a macro: its cast (last Simple / Round operation), its MAD (last read of v, S, M), its store.
constexpr int MC_CAST  = 2;
constexpr int MC_MAD   = 3;
constexpr int MC_STORE = 5;

constexpr int MSLOTS = 176;

struct mplan
{
    mslot s[MSLOTS];
    int n;
};

template <int FMT>
constexpr int mgroup(const int k)
{
    return 2 * (k / 16) + (traits<FMT>::has_h ? k / 8 % 2 : 0);
}

// Output order ORD: 0 = k; 1 = by scale group, then column half (one S / M pair per run, the other one reloads
// meanwhile); 2 = by code word, then column half.
template <int FMT, int CP, int ORD>
constexpr int morder_key(const int k)
{
    const int major = ORD == 1 ? mgroup<FMT>(k) : ORD == 2 ? mword<FMT, CP>(k) : 0;
    return (major * 2 + (ORD == 0 ? 0 : k % 2)) * 32 + k;
}

// IN_REGS: the first scale group's S / M are already in L4..L7 (k4_params_regs).
template <int FMT, int CP, int NV, int ORD, bool IN_REGS>
constexpr mplan make_mplan()
{
    using M         = mfield<FMT>;
    constexpr int N = M::clean;
    mplan p {};
    int ord[32] {}, ts[32] {}, tm[32] {}, wreg[32] {}, wload[32] {}, pload[32] {}, pdep[32] {};
    bool wneed[32] {};
    for (int j = 0; j < N; j++)
    {
        int i = j;
        for (; i > 0 && morder_key<FMT, CP, ORD>(ord[i - 1]) > morder_key<FMT, CP, ORD>(j); i--)
        {
            ord[i] = ord[i - 1];
        }
        ord[i] = j;
    }
    // Code word loads (wload[j]: issue slot, if j loads its word) and S / M loads (pload[j]: loads issued, 0..2).
    for (int j = 0, r = 2; j < N; j++)
    {
        const int k = ord[j];
        ts[j] = tm[j] = wload[j] = pdep[j] = -1;
        wneed[j]                           = j == 0 || mword<FMT, CP>(k) != mword<FMT, CP>(ord[j - 1]);
        if (wneed[j])
        {
            r = NV == 3 ? 3 : 5 - r;
        }
        wreg[j] = r;
        for (int i = j - 1; i >= 0 && pdep[j] < 0; i--)
        {
            pdep[j] = ord[i] % 2 == k % 2 ? i : -1; // the column half's previous position
        }
        const bool need = pdep[j] < 0 ? !IN_REGS || mgroup<FMT>(k) != 0 : mgroup<FMT>(ord[pdep[j]]) != mgroup<FMT>(k);
        pload[j]        = need ? 0 : 2;
        if (IN_REGS && need && mgroup<FMT>(k) == 0)
        {
            p.n = MSLOTS; // the first group's S / M exist only in registers: an order that returns to it is unusable
            return p;
        }
    }
    int last_m = -100, ns = 0, nm = 0;
    while (nm < N && p.n < MSLOTS)
    {
        const int t = p.n;
        // Shift of position ns: its word loaded, its v free (macro ns - NV past its MAD), past the last macro's cast.
        bool can_s = ns < N && (!wneed[ns] || wload[ns] >= 0) && t > last_m + MC_CAST;
        if (can_s && ns >= NV)
        {
            can_s = tm[ns - NV] >= 0 && t > tm[ns - NV] + MC_MAD;
        }
        if (can_s)
        {
            ts[ns]     = t;
            p.s[p.n++] = {MS_SHIFT, ns % NV, wreg[ns], 32 - M::B - mlsb<FMT>(ord[ns])};
            ns++;
            continue;
        }
        if (nm < ns && t > ts[nm] && t >= last_m + 2 && pload[nm] == 2)
        {
            tm[nm]     = t;
            last_m     = t;
            p.s[p.n++] = {MS_MACRO, ord[nm] % 2, nm % NV, mrow(ord[nm])};
            nm++;
            continue;
        }
        // A load: the code word of the next position that needs one (after the last shift reading its register), or
        // the S / M of the next position that needs them (after the last MAD reading their registers).
        int wj = -1, pj = -1;
        for (int j = ns; j < N && wj < 0; j++)
        {
            if (wneed[j] && wload[j] < 0)
            {
                bool free = true;
                for (int i = 0; i < j; i++)
                {
                    free = free && !(wreg[i] == wreg[j] && ts[i] < 0);
                }
                wj = free ? j : -2;
            }
        }
        for (int j = nm; j < N && pj < 0; j++)
        {
            if (pload[j] < 2)
            {
                const int d = pdep[j];
                pj          = d < 0 || (tm[d] >= 0 && t > tm[d] + MC_MAD) ? j : -2;
            }
        }
        if (wj >= 0 && (pj < 0 || wj <= pj))
        {
            wload[wj]  = t;
            p.s[p.n++] = {MS_LOAD, wreg[wj], sfpi::SFPLOAD_MOD0_FMT_INT32, CO + mword<FMT, CP>(ord[wj])};
        }
        else if (pj >= 0)
        {
            const int k = ord[pj], b = k % 2, a = k / 16, h = traits<FMT>::has_h ? k / 8 % 2 : 0;
            const int f = pload[pj]++;
            p.s[p.n++]  = {MS_LOAD, 4 + 2 * b + f, sfpi::SFPLOAD_MOD0_FMT_SRCB, (f ? PM : PS) + pidx(a, b, h)};
        }
        else
        {
            p.s[p.n++] = {MS_NOP, 0, 0, 0};
        }
    }
    // Nothing else on the SFPU until the last store.
    while (p.n <= last_m + MC_STORE && p.n < MSLOTS)
    {
        p.s[p.n++] = {MS_NOP, 0, 0, 0};
    }
    return p;
}

// The shortest of the register splits and output orders.
template <int FMT, int CP, bool IN_REGS>
constexpr mplan make_mplan_best()
{
    constexpr mplan c[6] = {
        make_mplan<FMT, CP, 3, 0, IN_REGS>(),
        make_mplan<FMT, CP, 3, 1, IN_REGS>(),
        make_mplan<FMT, CP, 3, 2, IN_REGS>(),
        make_mplan<FMT, CP, 2, 0, IN_REGS>(),
        make_mplan<FMT, CP, 2, 1, IN_REGS>(),
        make_mplan<FMT, CP, 2, 2, IN_REGS>()};
    int best = 0;
    for (int i = 1; i < 6; i++)
    {
        best = c[i].n < c[best].n ? i : best;
    }
    return c[best];
}

// The instruction-counted streams, three slots per output: the macro, the shift of its field into v (the macro's
// shift runs one instruction later) and a slot without a Simple instruction for the macro's Round-unit shift (a load
// or a NOP). Q4_0, Q4_1 and Q4_K: field i of code word 2h + a (L3) through macro i % 2, v in L0..L2 in turn; third
// slots load the next code word or S, M. IN_REGS: a = 0's S / M are already in L4..L7 (k4_params_regs).
template <bool IN_REGS>
sfpi_inline void body_macro_q4_icount()
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

// The other formats: tasks (code word and S / M loads) placed in free third slots or extra slots, earliest
// deadline first; code words alternate L3 / L2.
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
constexpr mplan make_mplan_icount()
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

template <int FMT, int CP, bool IN_REGS, bool CYC>
constexpr mplan mplan_v = CYC ? make_mplan_best<FMT, CP, IN_REGS>() : make_mplan_icount<FMT, CP>();

constexpr int mfirst_macro(const mplan& p)
{
    int i = 0;
    while (i < p.n && p.s[i].kind != MS_MACRO)
    {
        i++;
    }
    return i;
}

template <int FMT, int CP, bool IN_REGS, bool CYC, int I>
sfpi_inline void mslot_issue()
{
    constexpr mslot s = mplan_v<FMT, CP, IN_REGS, CYC>.s[I];
    if constexpr (s.kind == MS_MACRO)
    {
        TTI_SFPLOADMACRO((s.x << 2) | s.y, CYC ? LOAD_LO16_ONLY : LOAD_NONE, ADDR_MOD_7, 2 * s.z);
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

template <int FMT, int CP, bool IN_REGS, bool CYC, int... I>
sfpi_inline void mslots_issue(std::integer_sequence<int, I...>)
{
    (mslot_issue<FMT, CP, IN_REGS, CYC, I>(), ...);
}

// Q4_0, Q4_1 and Q4_K (one plan: four-bit fields, no k-halves). IN_REGS: a = 0's S / M are already in L4..L7
// (k4_params_regs).
template <bool IN_REGS, bool CYC>
sfpi_inline void body_macro_q4()
{
    if constexpr (CYC)
    {
        mslots_issue<Q4_0, 0, IN_REGS, true>(std::make_integer_sequence<int, mplan_v<Q4_0, 0, IN_REGS, true>.n> {});
    }
    else
    {
        body_macro_q4_icount<IN_REGS>();
    }
}

// The split rows (B not dividing 32) through body()'s arithmetic, then the planned stream.
template <int FMT, int CP, bool CYC>
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
    mslots_issue<FMT, CP, false, CYC>(std::make_integer_sequence<int, mplan_v<FMT, CP, false, CYC>.n> {});
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
