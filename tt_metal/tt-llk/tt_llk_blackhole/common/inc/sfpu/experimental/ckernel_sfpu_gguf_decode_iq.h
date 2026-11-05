// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// GGUF SFPU decoders of the codebook formats IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS and IQ3_S.

#pragma once

#include "sfpu/experimental/ckernel_sfpu_gguf_decode_common.h"

namespace ckernel::sfpu
{
namespace gguf
{

using namespace ckernel::gguf;

// IQ2_XXS page after lookup_grid: word-vectors 0 / 1 hold level words (lane (r, c) of word-vector a: entry l = r of
// columns 16a + 2c + b, element i at bits 16b + 2i, an index into (8, 25, 43)); rows 4..5 word n hold column n's sign
// word (byte l: entry l's 7 sign bits, scale bit l in bit 7); then d. Output o of lane (r, c) is element
// 4 ((o / 2) % 2) + r of entry 2h + (o / 4) % 2. The eighth sign bit is the parity of the seven. Value =
// (d / 4 (ls + 0.5)) * level with the sign set by copysgn, so d must not be negative (also IQ2_S, IQ2_XS, IQ3_S, IQ3_XXS).

constexpr int LV = 2 * TILE; // level words of entry l, columns (a, b): LV + 4a + l, pre-shifted
constexpr int NR = LV + 8;   // -r and -2r per lane (shift amounts: negative = right)
constexpr int NB = NR + 2;   // IQ1_S: row group 2 of word-vectors 2 and 3, replicated
constexpr int SG = PM;       // sign words with the parity bits, pre-shifted: SG + pidx(a, b, 0)

// IQ1_S, IQ1_M, IQ2_XXS, IQ2_S and IQ2_XS: the level stage as one SFPMAD per output with the level read from the LReg
// that the output's field indexes (iq2_ind_body, iq1_ind_body). ttsim has no indexed SFPMAD: under
// DISABLE_SFPLOADMACRO these formats use the cast-and-MAD bodies.
constexpr bool IQ_INDEXED = LOADMACRO;
// Bits 2..3 above each field the body reads: set for IQ2 (LReg index 12 + q), cleared for IQ1 (index q).
constexpr unsigned IQ_INDEX_BITS = IQ_INDEXED ? 0x0Cu : 0u;

template <unsigned SET, unsigned CLEAR>
sfpi_inline vUInt iq_index_bits(vUInt w)
{
    if constexpr (CLEAR != 0)
    {
        w = w & vUInt(~(CLEAR * 0x01010101u));
    }
    if constexpr (SET != 0)
    {
        w = w | vUInt(SET * 0x01010101u);
    }
    return w;
}

// Level words of the codebook formats (IQ2_XXS, IQ1_S): word-vector a's four row groups to every lane, shifted
// right by 2r per lane, into LV + 4a + l; the shift amounts into NR first. SET / CLEAR: bits set / cleared in every
// byte afterwards.
template <unsigned SET = 0, unsigned CLEAR = 0>
sfpi_inline void iq_level_words()
{
    {
        const vInt r                                        = lane_row();
        sfpi::dst_reg[NR].mode<sfpi::DataLayout::U32>()     = sfpi::as<vUInt>(vInt(0) - r);
        sfpi::dst_reg[NR + 1].mode<sfpi::DataLayout::U32>() = sfpi::as<vUInt>(vInt(0) - (r + r));
    }
#pragma GCC unroll 2
    for (int a = 0; a < 2; a++)
    {
        vUInt w0 = ldu(CO + a);
        vUInt w1 = w0;
        vUInt w2 = w0;
        vUInt w3 = w0;
        transp4(w0, w1, w2, w3);
        sfpi::dst_reg[LV + 4 * a + 0].mode<sfpi::DataLayout::U32>() = iq_index_bits<SET, CLEAR>(shift_after_transp(w0, NR + 1));
        sfpi::dst_reg[LV + 4 * a + 1].mode<sfpi::DataLayout::U32>() = iq_index_bits<SET, CLEAR>(shift_after_transp(w1, NR + 1));
        sfpi::dst_reg[LV + 4 * a + 2].mode<sfpi::DataLayout::U32>() = iq_index_bits<SET, CLEAR>(shift_after_transp(w2, NR + 1));
        sfpi::dst_reg[LV + 4 * a + 3].mode<sfpi::DataLayout::U32>() = iq_index_bits<SET, CLEAR>(shift_after_transp(w3, NR + 1));
    }
}

sfpi_inline void iq2_xxs_params()
{
    iq_level_words<IQ_INDEX_BITS, 0>();
#pragma GCC unroll 2
    for (int b = 0; b < 2; b++)
    {
        // Row groups 0 / 1 of word-vector 2 + b: columns b and 16 + b (+ 2c); groups 2 / 3 are grid indices.
        vUInt u0 = ldu(CO + 2 + b);
        vUInt u1 = u0;
        vUInt u2 = u0;
        vUInt u3 = u0;
        transp4(u0, u1, u2, u3);
#pragma GCC unroll 2
        for (int a = 0; a < 2; a++)
        {
            const vUInt u                                                   = a == 0 ? u0 : u1;
            vUInt y                                                         = (u >> 7u) & vUInt(0x01010101u);
            y                                                               = y | (y >> 7u);
            y                                                               = y | (y >> 14u);
            const vFloat ls                                                 = ufloat(y & vUInt(0xFu));
            sfpi::dst_reg[PS + pidx(a, b, 0)]                               = (pf32<IQ2_XXS>(0, b, a) * 0.25f) * (ls + 0.5f);
            const vUInt t                                                   = u & vUInt(0x7F7F7F7Fu);
            vUInt x                                                         = t ^ (t >> 4u);
            x                                                               = x ^ (x >> 2u);
            x                                                               = x ^ (x >> 1u);
            sfpi::dst_reg[SG + pidx(a, b, 0)].mode<sfpi::DataLayout::U32>() = shift_after_transp(t | ((x & vUInt(0x01010101u)) << 7u), NR);
        }
    }
}

// IQ1_S: the same level words (values -1, 0, 1 as indices 0..2). Row 6 word 2c + a holds, in half b, column
// 16a + 2c + b's scale ls (bits 6, 7, 14) and delta sign s (bit 15). Value = d (2 ls + 1) (q - 1 + delta),
// delta = +-0.125: PS = d (2 ls + 1), PM = delta - 1.
sfpi_inline void iq1_s_params()
{
    iq_level_words<0, IQ_INDEX_BITS>();
#pragma GCC unroll 2
    for (int a = 0; a < 2; a++)
    {
        vUInt u0 = ldu(CO + 2 + a);
        vUInt u1 = u0;
        vUInt u2 = u0;
        vUInt u3 = u0;
        transp4(u0, u1, u2, u3);
        sfpi::dst_reg[NB + a].mode<sfpi::DataLayout::U32>() = u2; // stored before any arithmetic
    }
#pragma GCC unroll 2
    for (int a = 0; a < 2; a++)
    {
        const vUInt u = ldu(NB + a);
#pragma GCC unroll 2
        for (int b = 0; b < 2; b++)
        {
            const vFloat ls                   = ufloat(bits(u, 6 + 16 * b, 2) | (bits(u, 14 + 16 * b, 1) << 2u));
            const vFloat sg                   = ufloat(bits(u, 15 + 16 * b, 1));
            sfpi::dst_reg[PS + pidx(a, b, 0)] = pf32<IQ1_S>(0, b, a) * (ls * 2.0f + 1.0f);
            sfpi::dst_reg[PM + pidx(a, b, 0)] = sg * -0.25f + -0.875f;
        }
    }
}

// IQ1_M: IQ1_S's pages; delta signs of entries 0..3 in bits 6, 7, 14, 15 of half b of row 6 word 2c + a. 3-bit
// scales from the group's scale page, rows 0..2 word 2c + a: field f = 4 TI + 2b + h at bit 3 (f % 10) of row f / 10
// (f = 30, 31: bits 0..1 at bits 30..31 of rows 0 / 1, bit 2 at bit f of row 2). PS per column and half, DM = delta - 1
// per column and entry.
constexpr int DM = CO + 16; // rows 32..63 of the page tile, past the page

template <int TI>
sfpi_inline void iq1_m_params()
{
    iq_level_words<0, IQ_INDEX_BITS>();
#pragma GCC unroll 2
    for (int a = 0; a < 2; a++)
    {
        vUInt u0 = ldu(CO + 2 + a);
        vUInt u1 = u0;
        vUInt u2 = u0;
        vUInt u3 = u0;
        transp4(u0, u1, u2, u3);
        sfpi::dst_reg[NB + a].mode<sfpi::DataLayout::U32>() = u2; // stored before any arithmetic
    }
#pragma GCC unroll 2
    for (int a = 0; a < 2; a++)
    {
        vUInt r0, r1, r2, r3;
        rep4<IQ1_M>(0, a, r0, r1, r2, r3);
#pragma GCC unroll 2
        for (int b = 0; b < 2; b++)
        {
#pragma GCC unroll 2
            for (int h = 0; h < 2; h++)
            {
                const int f = 4 * TI + 2 * b + h;
                vUInt ls;
                if (f < 30)
                {
                    ls = bits(f < 10 ? r0 : f < 20 ? r1 : r2, 3 * (f % 10), 3);
                }
                else
                {
                    ls = ((f == 30 ? r0 : r1) >> 30u) | (bits(r2, f, 1) << 2u);
                }
                sfpi::dst_reg[PS + pidx(a, b, h)] = pf32<IQ1_M>(0, b, a) * (ufloat(ls) * 2.0f + 1.0f);
            }
        }
    }
#pragma GCC unroll 2
    for (int a = 0; a < 2; a++)
    {
        const vUInt u = ldu(NB + a);
#pragma GCC unroll 2
        for (int b = 0; b < 2; b++)
        {
#pragma GCC unroll 4
            for (int l = 0; l < 4; l++)
            {
                const vFloat sg                         = ufloat(bits(u, 16 * b + (l < 2 ? 6 + l : 12 + l), 1));
                sfpi::dst_reg[DM + 4 * (2 * a + b) + l] = sg * -0.25f + -0.875f;
            }
        }
    }
}

// M: S per column and half and D per column and entry (IQ1_M) instead of both per column (IQ1_S).
template <bool M>
sfpi_inline void iq1_s_body()
{
#pragma GCC unroll 2
    for (int a = 0; a < 2; a++)
    {
#pragma GCC unroll 2
        for (int h = 0; h < 2; h++)
        {
#pragma GCC unroll 2
            for (int jj = 0; jj < 2; jj++)
            {
                const vUInt w = ldu(LV + 4 * a + 2 * h + jj);
#pragma GCC unroll 2
                for (int b = 0; b < 2; b++)
                {
                    const vFloat S        = sfpi::dst_reg[PS + pidx(a, b, M ? h : 0)];
                    const vFloat D        = sfpi::dst_reg[M ? DM + 4 * (2 * a + b) + 2 * h + jj : PM + pidx(a, b, 0)];
                    const int oa          = 16 * h + 8 * a + 4 * jj + b;
                    const vFloat qa       = ufloat(bits(w, 16 * b, 2));
                    const vFloat qb       = ufloat(bits(w, 16 * b + 8, 2));
                    sfpi::dst_reg[oa]     = (qa + D) * S;
                    sfpi::dst_reg[oa + 2] = (qb + D) * S;
                }
            }
        }
    }
}

// Level (8, 25, 43) of index q: (q / 2 + 16.5) q + 8, exact. Programmable constants 0..2 = 0.5, 16.5, 8.
sfpi_inline void iq2_xxs_init()
{
    sfpi::vConstFloatPrgm0 = 0.5f;
    sfpi::vConstFloatPrgm1 = 16.5f;
    sfpi::vConstFloatPrgm2 = 8.0f;
}

// IQ2_S: IQ2_XXS's level words and body. Rows 4..5 hold each column's 32 sign bits; row 8 word 2c + a holds, in half
// b, column 16a + 2c + b's scales of elements 0..15 / 16..31 in bits 4..7 / 12..15. S = d / 4 (ls + 0.5) per half.
sfpi_inline void iq2_s_params()
{
    iq_level_words<IQ_INDEX_BITS, 0>();
#pragma GCC unroll 2
    for (int b = 0; b < 2; b++)
    {
        vUInt u0 = ldu(CO + 2 + b);
        vUInt u1 = u0;
        vUInt u2 = u0;
        vUInt u3 = u0;
        transp4(u0, u1, u2, u3);
        sfpi::dst_reg[SG + pidx(0, b, 0)].mode<sfpi::DataLayout::U32>() = shift_after_transp(u0, NR);
        sfpi::dst_reg[SG + pidx(1, b, 0)].mode<sfpi::DataLayout::U32>() = shift_after_transp(u1, NR);
    }
#pragma GCC unroll 2
    for (int a = 0; a < 2; a++)
    {
        vUInt u0 = ldu(CO + 4 + a);
        vUInt u1 = u0;
        vUInt u2 = u0;
        vUInt u3 = u0;
        transp4(u0, u1, u2, u3);
        sfpi::dst_reg[NB + a].mode<sfpi::DataLayout::U32>() = u0; // stored before any arithmetic
    }
#pragma GCC unroll 2
    for (int a = 0; a < 2; a++)
    {
        const vUInt u = ldu(NB + a);
#pragma GCC unroll 2
        for (int b = 0; b < 2; b++)
        {
            const vFloat d4 = pf32<IQ2_S>(0, b, a) * 0.25f;
#pragma GCC unroll 2
            for (int h = 0; h < 2; h++)
            {
                sfpi::dst_reg[PS + pidx(a, b, h)] = d4 * (ufloat(bits(u, 16 * b + 4 + 8 * h, 4)) + 0.5f);
            }
        }
    }
}

// IQ2_XS: IQ2_S's level words and scales, IQ2_XXS's 7-bit signs per entry: rows 4..5 word 16 (l / 2) + 2c + l % 2
// holds entry l of columns 2c, 2c + 1, 16 + 2c, 17 + 2c in bytes 0..3. Scales from the group's scale page: rows 0..3
// word 2c + a, byte 2 (TI % 2) + b of row TI / 2.
template <int TI>
sfpi_inline void iq2_xs_params()
{
    iq_level_words<IQ_INDEX_BITS, 0>();
#pragma GCC unroll 2
    for (int p = 0; p < 2; p++)
    {
        vUInt u0 = ldu(CO + 2 + p);
        vUInt u1 = u0;
        vUInt u2 = u0;
        vUInt u3 = u0;
        transp4(u0, u1, u2, u3);
#pragma GCC unroll 2
        for (int g = 0; g < 2; g++)
        {
            const vUInt t                                               = (g == 0 ? u0 : u1) & vUInt(0x7F7F7F7Fu);
            vUInt x                                                     = t ^ (t >> 4u);
            x                                                           = x ^ (x >> 2u);
            x                                                           = x ^ (x >> 1u);
            sfpi::dst_reg[SG + 2 * g + p].mode<sfpi::DataLayout::U32>() = shift_after_transp(t | ((x & vUInt(0x01010101u)) << 7u), NR);
        }
    }
#pragma GCC unroll 2
    for (int a = 0; a < 2; a++)
    {
        vUInt r0, r1, r2, r3;
        rep4<IQ2_XS>(0, a, r0, r1, r2, r3);
        const vUInt sw = TI / 2 == 0 ? r0 : TI / 2 == 1 ? r1 : TI / 2 == 2 ? r2 : r3;
#pragma GCC unroll 2
        for (int b = 0; b < 2; b++)
        {
            const vFloat d4 = pf32<IQ2_XS>(0, b, a) * 0.25f;
#pragma GCC unroll 2
            for (int h = 0; h < 2; h++)
            {
                sfpi::dst_reg[PS + pidx(a, b, h)] = d4 * (ufloat(bits(sw, 8 * (2 * (TI % 2) + b) + 4 * h, 4)) + 0.5f);
            }
        }
    }
}

// IQ3_S: entries of 4 elements with 3-bit indices into (1, 3, .., 15). lookup_grid writes entry e of columns
// 16a + 2c + b in the halves of word-vector 2a + e / 4, row group e % 4, element i at bits 3i, and each column's 4-bit
// scale in bits 12..15 of its entry 0. Rows 8..9 hold each column's 32 sign bits. Value = (2q + 1) d (1 + 2 ls), signed.
constexpr int LV3 = 2 * TILE; // level words of entry e, columns (a, b): LV3 + 8a + e (IQ3_S: pre-shifted)
constexpr int NR3 = PS + 1;   // -r; NR3 + 2: -3r (the odd PS slots: S is per column)
constexpr int SC3 = PS + 5;   // entry-0 words of a = 0 / 1 (SC3, SC3 + 2), unshifted: the scales

sfpi_inline void iq3_s_params()
{
    {
        const vInt r                                         = lane_row();
        sfpi::dst_reg[NR3].mode<sfpi::DataLayout::U32>()     = sfpi::as<vUInt>(vInt(0) - r);
        sfpi::dst_reg[NR3 + 2].mode<sfpi::DataLayout::U32>() = sfpi::as<vUInt>(vInt(0) - (r + r + r));
    }
#pragma GCC unroll 4
    for (int wv = 0; wv < 4; wv++)
    {
        vUInt w0 = ldu(CO + wv);
        vUInt w1 = w0;
        vUInt w2 = w0;
        vUInt w3 = w0;
        transp4(w0, w1, w2, w3);
        const int e0 = LV3 + 8 * (wv / 2) + 4 * (wv % 2);
        if (wv % 2 == 0)
        {
            sfpi::dst_reg[SC3 + wv].mode<sfpi::DataLayout::U32>() = w0;
        }
        sfpi::dst_reg[e0 + 0].mode<sfpi::DataLayout::U32>() = shift_after_transp(w0, NR3 + 2);
        sfpi::dst_reg[e0 + 1].mode<sfpi::DataLayout::U32>() = shift_after_transp(w1, NR3 + 2);
        sfpi::dst_reg[e0 + 2].mode<sfpi::DataLayout::U32>() = shift_after_transp(w2, NR3 + 2);
        sfpi::dst_reg[e0 + 3].mode<sfpi::DataLayout::U32>() = shift_after_transp(w3, NR3 + 2);
    }
#pragma GCC unroll 2
    for (int b = 0; b < 2; b++)
    {
        vUInt u0 = ldu(CO + 4 + b);
        vUInt u1 = u0;
        vUInt u2 = u0;
        vUInt u3 = u0;
        transp4(u0, u1, u2, u3);
        sfpi::dst_reg[SG + pidx(0, b, 0)].mode<sfpi::DataLayout::U32>() = shift_after_transp(u0, NR3);
        sfpi::dst_reg[SG + pidx(1, b, 0)].mode<sfpi::DataLayout::U32>() = shift_after_transp(u1, NR3);
    }
#pragma GCC unroll 2
    for (int a = 0; a < 2; a++)
    {
        const vUInt u = ldu(SC3 + 2 * a);
#pragma GCC unroll 2
        for (int b = 0; b < 2; b++)
        {
            sfpi::dst_reg[PS + pidx(a, b, 0)] = pf32<IQ3_S>(0, b, a) * (ufloat(bits(u, 12 + 16 * b, 4)) * 2.0f + 1.0f);
        }
    }
}

sfpi_inline void iq3_s_body()
{
#pragma GCC unroll 2
    for (int a = 0; a < 2; a++)
    {
#pragma GCC unroll 2
        for (int h = 0; h < 2; h++)
        {
#pragma GCC unroll 2
            for (int b = 0; b < 2; b++)
            {
                const vFloat S  = sfpi::dst_reg[PS + pidx(a, b, 0)];
                const vFloat S2 = S + S;
                const vUInt sg  = ldu(SG + pidx(a, b, 0));
#pragma GCC unroll 4
                for (int j = 0; j < 4; j++)
                {
                    const vUInt w    = ldu(LV3 + 8 * a + 4 * h + j);
                    const vFloat q   = ufloat(bits(w, 16 * b, 3));
                    const vFloat v   = q * S2 + S;
                    const int o      = 16 * h + 8 * a + 2 * j + b;
                    sfpi::dst_reg[o] = sfpi::copysgn(v, sfpi::as<vInt>(sg << static_cast<unsigned>(31 - 16 * h - 4 * j)));
                }
            }
        }
    }
}

// IQ3_XXS: IQ3_S's level words with 4-bit codes f (element r at bits 16b + 4r; f = 0..6 for 4, 12, .., 52 and 8 for
// 62), stored unshifted; IQ2_XXS's sign words in rows 8..9. Value = (d / 2 (ls + 0.5)) (65 - |61 - 8 f|), signed. AM3,
// AM3 + 2: per-lane left shift 28 - 16b - 4r of element r of half b to the top.
constexpr int AM3 = PS + 3;

sfpi_inline void iq3_xxs_params()
{
    sfpi::vConstIntPrgm0 = 0x01010101;
    sfpi::vConstIntPrgm1 = 0x7F7F7F7F;
    const vUInt ones     = sfpi::as<vUInt>(sfpi::vConstIntPrgm0);
    {
        const vInt r                                         = lane_row();
        const vInt r4                                        = (r + r) + (r + r);
        sfpi::dst_reg[NR3].mode<sfpi::DataLayout::U32>()     = sfpi::as<vUInt>(vInt(0) - r);
        sfpi::dst_reg[AM3].mode<sfpi::DataLayout::U32>()     = sfpi::as<vUInt>(vInt(28) - r4);
        sfpi::dst_reg[AM3 + 2].mode<sfpi::DataLayout::U32>() = sfpi::as<vUInt>(vInt(12) - r4);
    }
#pragma GCC unroll 4
    for (int wv = 0; wv < 4; wv++)
    {
        vUInt w0 = ldu(CO + wv);
        vUInt w1 = w0;
        vUInt w2 = w0;
        vUInt w3 = w0;
        transp4(w0, w1, w2, w3);
        const int e0                                        = LV3 + 8 * (wv / 2) + 4 * (wv % 2);
        sfpi::dst_reg[e0 + 0].mode<sfpi::DataLayout::U32>() = w0;
        sfpi::dst_reg[e0 + 1].mode<sfpi::DataLayout::U32>() = w1;
        sfpi::dst_reg[e0 + 2].mode<sfpi::DataLayout::U32>() = w2;
        sfpi::dst_reg[e0 + 3].mode<sfpi::DataLayout::U32>() = w3;
    }
#pragma GCC unroll 2
    for (int b = 0; b < 2; b++)
    {
        // Row groups 0 / 1 of word-vector 4 + b: columns b and 16 + b (+ 2c).
        vUInt u0 = ldu(CO + 4 + b);
        vUInt u1 = u0;
        vUInt u2 = u0;
        vUInt u3 = u0;
        transp4(u0, u1, u2, u3);
#pragma GCC unroll 2
        for (int a = 0; a < 2; a++)
        {
            const vUInt u                                                   = a == 0 ? u0 : u1;
            vUInt y                                                         = (u >> 7u) & ones;
            y                                                               = y | (y >> 7u);
            y                                                               = y | (y >> 14u);
            const vFloat ls                                                 = ufloat(y & vUInt(0xFu));
            sfpi::dst_reg[PS + pidx(a, b, 0)]                               = (pf32<IQ3_XXS>(0, b, a) * 0.5f) * (ls + 0.5f);
            const vUInt t                                                   = u & sfpi::as<vUInt>(sfpi::vConstIntPrgm1);
            vUInt x                                                         = t ^ (t >> 4u);
            x                                                               = x ^ (x >> 2u);
            x                                                               = x ^ (x >> 1u);
            sfpi::dst_reg[SG + pidx(a, b, 0)].mode<sfpi::DataLayout::U32>() = shift_after_transp(t | ((x & ones) << 7u), NR3);
        }
    }
}

sfpi_inline void iq3_xxs_init()
{
    sfpi::vConstFloatPrgm0 = -8.0f;
    sfpi::vConstFloatPrgm1 = 61.0f;
}

sfpi_inline void iq3_xxs_body()
{
#pragma GCC unroll 2
    for (int b = 0; b < 2; b++)
    {
        const vUInt am = ldu(AM3 + 2 * b);
#pragma GCC unroll 2
        for (int a = 0; a < 2; a++)
        {
            const vFloat S = sfpi::dst_reg[PS + pidx(a, b, 0)];
            const vFloat M = S * -65.0f;
            const vUInt sg = ldu(SG + pidx(a, b, 0));
#pragma GCC unroll 2
            for (int h = 0; h < 2; h++)
            {
#pragma GCC unroll 4
                for (int j = 0; j < 4; j++)
                {
                    const vUInt w    = ldu(LV3 + 8 * a + 4 * h + j);
                    const vFloat f   = ufloat((w << am) >> 28u);
                    const vFloat y   = sfpi::abs(f * sfpi::vConstFloatPrgm0 + sfpi::vConstFloatPrgm1);
                    const vFloat v   = y * S + M; // -(65 - y) S: copysgn sets the sign
                    const int o      = 16 * h + 8 * a + 2 * j + b;
                    sfpi::dst_reg[o] = sfpi::copysgn(v, sfpi::as<vInt>(sg << static_cast<unsigned>(31 - 16 * h - 4 * j)));
                }
            }
        }
    }
}

// PER_H: S per column and half (IQ2_S, IQ2_XS) instead of per column (IQ2_XXS). ENTRY_SIGNS: a sign word per entry,
// column (a, b) in byte 2a + b (IQ2_XS), instead of one per column, entry l in byte l.
template <bool PER_H, bool ENTRY_SIGNS = false>
sfpi_inline void iq2_xxs_body()
{
#pragma GCC unroll 2
    for (int a = 0; a < 2; a++)
    {
#pragma GCC unroll 2
        for (int h = 0; h < 2; h++)
        {
#pragma GCC unroll 2
            for (int jj = 0; jj < 2; jj++)
            {
                const vUInt w = ldu(LV + 4 * a + 2 * h + jj);
#pragma GCC unroll 2
                for (int b = 0; b < 2; b++)
                {
                    const vFloat S = sfpi::dst_reg[PS + pidx(a, b, PER_H ? h : 0)];
                    const vUInt sg = ldu(SG + (ENTRY_SIGNS ? 2 * h + jj : pidx(a, b, 0)));
                    // Outputs j = 2jj and 2jj + 1, interleaved.
                    const int oa      = 16 * h + 8 * a + 4 * jj + b;
                    const int ob      = oa + 2;
                    const vFloat qa   = ufloat(bits(w, 16 * b, 2));
                    const vFloat qb   = ufloat(bits(w, 16 * b + 8, 2));
                    vFloat la         = qa * sfpi::vConstFloatPrgm0 + sfpi::vConstFloatPrgm1;
                    vFloat lb         = qb * sfpi::vConstFloatPrgm0 + sfpi::vConstFloatPrgm1;
                    la                = la * qa + sfpi::vConstFloatPrgm2;
                    lb                = lb * qb + sfpi::vConstFloatPrgm2;
                    const vFloat va   = la * S;
                    const vFloat vb   = lb * S;
                    const int pa      = ENTRY_SIGNS ? 8 * (2 * a + b) : 16 * h + 8 * jj;
                    sfpi::dst_reg[oa] = sfpi::copysgn(va, sfpi::as<vInt>(sg << static_cast<unsigned>(31 - pa)));
                    sfpi::dst_reg[ob] = sfpi::copysgn(vb, sfpi::as<vInt>(sg << static_cast<unsigned>(27 - pa)));
                }
            }
        }
    }
}

constexpr unsigned IQ_ZERO = p_sfpu::LCONST_0;

constexpr unsigned iq_shr(const int k)
{
    return static_cast<unsigned>(-k) & 0xFFFu;
}

// IQ2_XXS, IQ2_S, IQ2_XS: levels 8, 25, 43 in the programmable constants (LReg 12..14). iq2_xxs_body's outputs: the
// level word shifted to the output's field gives L7 = 12 + q, then T[q] S and the grid sign. L0 level word, L1 S,
// L2 sign word, L3..L6 results.
sfpi_inline void iq2_ind_init()
{
    sfpi::vConstFloatPrgm0 = 8.0f;
    sfpi::vConstFloatPrgm1 = 25.0f;
    sfpi::vConstFloatPrgm2 = 43.0f;
}

template <bool PER_H, bool ENTRY_SIGNS = false>
sfpi_inline void iq2_ind_body()
{
#pragma GCC unroll 2
    for (int a = 0; a < 2; a++)
    {
#pragma GCC unroll 2
        for (int b = 0; b < 2; b++)
        {
#pragma GCC unroll 2
            for (int h = 0; h < 2; h++)
            {
                if (h == 0 || PER_H)
                {
                    TTI_SFPLOAD(1, sfpi::SFPLOAD_MOD0_FMT_SRCB, ADDR_MOD_7, 2 * (PS + pidx(a, b, PER_H ? h : 0)));
                }
#pragma GCC unroll 2
                for (int jj = 0; jj < 2; jj++)
                {
                    if (ENTRY_SIGNS || (h == 0 && jj == 0))
                    {
                        TTI_SFPLOAD(2, sfpi::SFPLOAD_MOD0_FMT_INT32, ADDR_MOD_7, 2 * (SG + (ENTRY_SIGNS ? 2 * h + jj : pidx(a, b, 0))));
                    }
                    TTI_SFPLOAD(0, sfpi::SFPLOAD_MOD0_FMT_INT32, ADDR_MOD_7, 2 * (LV + 4 * a + 2 * h + jj));
                    const int oa = 16 * h + 8 * a + 4 * jj + b;
                    const int pa = ENTRY_SIGNS ? 8 * (2 * a + b) : 16 * h + 8 * jj;
                    TTI_SFPSHFT(iq_shr(16 * b), 0, 7, SHIFT_IMM_OF_C);
                    TTI_SFPMAD(0, 1, IQ_ZERO, 3, sfpi::SFPMAD_MOD1_SRCA_LREG7);
                    TTI_SFPSHFT(31 - pa, 2, 4, SHIFT_IMM_OF_C);
                    TTI_SFPSHFT(iq_shr(16 * b + 8), 0, 7, SHIFT_IMM_OF_C);
                    TTI_SFPSETSGN(0, 3, 4, 0);
                    TTI_SFPMAD(0, 1, IQ_ZERO, 5, sfpi::SFPMAD_MOD1_SRCA_LREG7);
                    TTI_SFPSTORE(4, sfpi::SFPSTORE_MOD0_FMT_SRCB, ADDR_MOD_7, 2 * oa);
                    TTI_SFPSHFT(27 - pa, 2, 6, SHIFT_IMM_OF_C);
                    TTI_SFPSETSGN(0, 5, 6, 0);
                    TTI_SFPSTORE(6, sfpi::SFPSTORE_MOD0_FMT_SRCB, ADDR_MOD_7, 2 * (oa + 2));
                }
            }
        }
    }
}

// IQ1_S, IQ1_M: levels 0, 1, 2 in L0..L2 (L7 = q), q S + D S: iq1_s_body's (q + D) S with the same single rounding,
// since D S is exact. L3 level word, L4 S, L5 D S, L6 results.
sfpi_inline void iq1_ind_init()
{
    TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_FLOATB, 0x0000); // 0.0
    TTI_SFPLOADI(1, sfpi::SFPLOADI_MOD0_FLOATB, 0x3F80); // 1.0
    TTI_SFPLOADI(2, sfpi::SFPLOADI_MOD0_FLOATB, 0x4000); // 2.0
}

template <bool M>
sfpi_inline void iq1_ind_body()
{
#pragma GCC unroll 2
    for (int a = 0; a < 2; a++)
    {
#pragma GCC unroll 2
        for (int b = 0; b < 2; b++)
        {
#pragma GCC unroll 2
            for (int h = 0; h < 2; h++)
            {
                if (h == 0 || M)
                {
                    TTI_SFPLOAD(4, sfpi::SFPLOAD_MOD0_FMT_SRCB, ADDR_MOD_7, 2 * (PS + pidx(a, b, M ? h : 0)));
                }
#pragma GCC unroll 2
                for (int jj = 0; jj < 2; jj++)
                {
                    if (M || (h == 0 && jj == 0))
                    {
                        TTI_SFPLOAD(5, sfpi::SFPLOAD_MOD0_FMT_SRCB, ADDR_MOD_7, 2 * (M ? DM + 4 * (2 * a + b) + 2 * h + jj : PM + pidx(a, b, 0)));
                    }
                    TTI_SFPLOAD(3, sfpi::SFPLOAD_MOD0_FMT_INT32, ADDR_MOD_7, 2 * (LV + 4 * a + 2 * h + jj));
                    if (M || (h == 0 && jj == 0))
                    {
                        TTI_SFPMAD(5, 4, IQ_ZERO, 5, sfpi::SFPMAD_MOD1_OFFSET_NONE); // D S
                    }
                    const int oa = 16 * h + 8 * a + 4 * jj + b;
                    TTI_SFPSHFT(iq_shr(16 * b), 3, 7, SHIFT_IMM_OF_C);
                    TTI_SFPMAD(0, 4, 5, 6, sfpi::SFPMAD_MOD1_SRCA_LREG7);
                    TTI_SFPSHFT(iq_shr(16 * b + 8), 3, 7, SHIFT_IMM_OF_C);
                    TTI_SFPSTORE(6, sfpi::SFPSTORE_MOD0_FMT_SRCB, ADDR_MOD_7, 2 * oa);
                    TTI_SFPMAD(0, 4, 5, 6, sfpi::SFPMAD_MOD1_SRCA_LREG7);
                    TTI_SFPSTORE(6, sfpi::SFPSTORE_MOD0_FMT_SRCB, ADDR_MOD_7, 2 * (oa + 2));
                }
            }
        }
    }
}

template <bool PER_H, bool ENTRY_SIGNS = false>
sfpi_inline void iq2_stage()
{
    if constexpr (IQ_INDEXED)
    {
        iq2_ind_init();
        iq2_ind_body<PER_H, ENTRY_SIGNS>();
    }
    else
    {
        iq2_xxs_init();
        iq2_xxs_body<PER_H, ENTRY_SIGNS>();
    }
}

template <bool M>
sfpi_inline void iq1_stage()
{
    if constexpr (IQ_INDEXED)
    {
        iq1_ind_init();
        iq1_ind_body<M>();
    }
    else
    {
        iq1_s_body<M>();
    }
}

} // namespace gguf
} // namespace ckernel::sfpu
