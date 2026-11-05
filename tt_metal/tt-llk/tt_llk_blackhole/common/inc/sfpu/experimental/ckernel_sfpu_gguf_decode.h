// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// SFPU decoders of GGUF weight formats: one page (experimental/ckernel_gguf_page.h) to one Float32 [K][N] tile in DEST
// with the block scales applied. Needs a 32-bit DEST (fp32_dest_acc_en).
//
// DEST, from the tile the call starts at (sfpu_gguf_decode_tile): tile 0 is the output, tile 1 the page (unpacked as
// UInt32), tile 2 scratch for the per-lane scales S and offsets M. Output o of SFPU lane (r, c) is element
// k = 16h + 4((o / 2) % 4) + r, n = 16a + 2c + b of the tile, with h = o / 16, a = (o / 8) % 2, b = o % 2: the K
// row and the N column (the weight's row) of a [K][N] tile, as a matmul's SrcA wants it.
//
// The page's code words sit where those lanes load them: word-vector v (DEST rows 4 (v / 2) .. + 3 of the page tile,
// columns of parity v % 2) holds, in lane (r, c), the codes of that lane's outputs. Its scale fields start at row
// R = scale_offset / 64, regrouped so that word w = 2c + p of a row carries the fields of columns 16p + 2c and
// 16p + 2c + 1 (low half, high half).
//
// Phase 1 (params) reads the scale fields and writes S[a][b][h] and M[a][b][h] per lane to the scratch tile: a 4-row
// block of scale rows is loaded four times and transposed (transp4), which leaves each row replicated over the four
// SFPU row groups; fp16 fields come converted (f16_at) or as the fp32 rows ckernel_gguf_page.h prepare_scales adds.
// Phase 2 (body) extracts each output's code and writes q * S + M (linear formats) or v(q) * S (value-table and
// codebook formats). Results equal ggml's dequantization bit for bit: every product of a code and a scale is exact in
// fp32, and q * S + M rounds once like ggml's d * q - m.

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "experimental/ckernel_gguf_page.h"
#include "sfpi.h"

namespace ckernel::sfpu
{
namespace gguf
{

using namespace ckernel::gguf;

// SFPTRANSP transposes LRegs 0..3 and 4..7 as two 4x4 blocks; the second block is scratch here. It uses all eight
// LRegs, so no other value stays in an LReg across it.
template <typename V>
sfpi_inline void transp4(V& a, V& b, V& c, V& d)
{
    V e = a, f = a, g = a, h = a;
    sfpi::subvec_transp(a, b, c, d, e, f, g, h);
}

constexpr int TILE = 32;            // dst_reg entries per Float32 DEST tile
constexpr int CO   = TILE;          // the page
constexpr int PS   = 2 * TILE + 16; // S[(a * 2 + b) * 2 + h]
constexpr int PM   = 2 * TILE + 24; // M[(a * 2 + b) * 2 + h]

using sfpi::vFloat;
using sfpi::vInt;
using sfpi::vUInt;

sfpi_inline vUInt ldu(const int i)
{
    return sfpi::dst_reg[i].mode<sfpi::DataLayout::U32>();
}

sfpi_inline vFloat ufloat(const vUInt v)
{
    return sfpi::convert<vFloat>(sfpi::as<sfpi::vMag>(v), sfpi::RoundMode::Nearest);
}

template <int FMT>
struct traits
{
    static constexpr int lo      = FMT == Q8_0 ? 8 : (FMT == Q3_K || FMT == Q2_K || FMT == TQ2_0 || FMT == Q2_0) ? 2 : FMT == Q1_0 ? 1 : 4;
    static constexpr int hi      = (FMT == Q5_K || FMT == Q3_K || FMT == Q5_0 || FMT == Q5_1) ? 1 : FMT == Q6_K ? 2 : 0;
    static constexpr bool pair   = FMT == Q5_K || FMT == Q3_K || FMT == Q1_0 || FMT == Q5_0 || FMT == Q5_1;
    static constexpr bool has_h  = FMT == NVFP4 || FMT == Q6_K || FMT == Q3_K || FMT == Q2_K;
    static constexpr bool linear = !(FMT == IQ4_NL || FMT == IQ4_XS || FMT == MXFP4 || FMT == NVFP4);
};

// The page's code rows, before the scale fields.
template <int FMT>
constexpr int scale_row()
{
    return page_layout<static_cast<Format>(FMT)>().scale_offset / 64;
}

// Rows 4g..4g+3 of the scale fields, word parity p, each replicated to all SFPU row groups.
template <int FMT>
sfpi_inline void rep4(const int g, const int p, vUInt& r0, vUInt& r1, vUInt& r2, vUInt& r3)
{
    r0 = ldu(CO + scale_row<FMT>() / 2 + 2 * g + p);
    r1 = r0;
    r2 = r0;
    r3 = r0;
    transp4(r0, r1, r2, r3);
}

// Replicated copies of scale rows 4g..4g+3, parity p, into scratch dst_reg[base + 0..3] of the output tile (phase 2
// overwrites the output tile afterwards).
template <int FMT>
sfpi_inline void rep4_to(const int g, const int p, const int base)
{
    vUInt r0, r1, r2, r3;
    rep4<FMT>(g, p, r0, r1, r2, r3);
    sfpi::dst_reg[base + 0].mode<sfpi::DataLayout::U32>() = r0;
    sfpi::dst_reg[base + 1].mode<sfpi::DataLayout::U32>() = r1;
    sfpi::dst_reg[base + 2].mode<sfpi::DataLayout::U32>() = r2;
    sfpi::dst_reg[base + 3].mode<sfpi::DataLayout::U32>() = r3;
}

// Bit field [lsb, lsb + width) of w, unsigned: at most two shifts, no mask constant.
sfpi_inline vUInt bits(const vUInt w, const int lsb, const int width)
{
    if (lsb + width == 32)
    {
        return lsb == 0 ? w : w >> static_cast<unsigned>(lsb);
    }
    return (w << static_cast<unsigned>(32 - lsb - width)) >> static_cast<unsigned>(32 - width);
}

// Prepared formats (fp16_rows > 0): fp32 field F (fp16 row F after the raw rows) of column 16a + 2c + b, from rows
// R + 4 (2F + b) .. + 3 of the scale fields, R = scale rows rounded up to 4 (ckernel_gguf_page.h prepare_scales): one
// load gives every lane its value without an fp16 conversion.
template <int FMT>
sfpi_inline vFloat pf32(const int F, const int b, const int a)
{
    constexpr int fb = round_up(page_layout<static_cast<Format>(FMT)>().scale_rows, 4);
    return sfpi::dst_reg[CO + scale_row<FMT>() / 2 + 2 * (fb / 4 + 2 * F + b) + a];
}

// Byte B (0..23) of a lane's scale bytes in scratch rows base + B / 4 (rep4_to copies): field [lsb, lsb + width).
sfpi_inline vUInt byte_bits(const int base, const int B, const int lsb, const int width)
{
    return bits(ldu(base + B / 4), 8 * (B % 4) + lsb, width);
}

// Scale conversions without branches on the value class. B rebias-adds the field's bits into the fp32 exponent: exact
// for normal values, larger than the true value for subnormal ones. A = 2 B - (smallest normal): exact for subnormal
// values, not smaller than the true value for normal ones. The minimum is the value. Only normal fp32 values occur.
// Constants (params_init): programmable constant 0 = 112 << 23 (fp16), 1 = 119 << 23 (UE4M3 halved).
sfpi_inline void params_init()
{
    sfpi::vConstIntPrgm0 = 0x38000000;
    sfpi::vConstIntPrgm1 = 0x3B800000;
}

// fp16 in bits [16 HALF, 16 HALF + 16) of w to fp32, exact for every finite value.
template <int HALF>
sfpi_inline vFloat f16_at(const vUInt w)
{
    const vUInt m13 = HALF == 0 ? (w << 17u) >> 4u : ((w >> 16u) << 17u) >> 4u;
    const vFloat b  = sfpi::as<vFloat>(m13 + sfpi::as<vUInt>(sfpi::vConstIntPrgm0));
    const vFloat a  = sfpi::addexp(b, 1) + -6.103515625e-05f;
    const vInt sgn  = sfpi::as<vInt>(HALF == 0 ? w << 16u : w);
    return sfpi::copysgn(sfpi::min_max(a, b).first, sgn);
}

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

sfpi_inline int pidx(const int a, const int b, const int h)
{
    return (a * 2 + b) * 2 + h;
}

// ---------------------------------------------------------------------------------------------------------------
// IQ2_XXS: after the grid lookup (ckernel_gguf_page.h lookup_grid) the page holds:
//   word-vectors 0 / 1 (rows 0..3): level words, lane (r, c) of word-vector a: entry l = r (elements 8l .. 8l + 7) of
//     columns 16a + 2c + b, element i at bits 16b + 2i, an index into (8, 25, 43);
//   rows 4..5, word n: column n's sign word, byte l = the 7 sign bits of entry l and, in bit 7, scale bit l;
//   the d page (fp16, prepared to fp32).
// Output o of lane (r, c) is element i = 4 ((o / 2) % 2) + r of entry l = 2h + (o / 4) % 2 of column (a, b). A
// transpose gives every lane all four row groups of a word-vector, and a right shift by 2r (levels) or r (signs) per
// lane moves each output's field to a position that depends on o only. The eighth sign bit of an entry is the parity
// of its seven (ksigns_iq2xs). Value = (d / 4 (ls + 0.5)) * level, sign applied: ggml's db * grid * sign, one
// rounding. The sign bits set the value's sign (copysgn), so d must not be negative, as ggml's quantizers write it; the
// same holds for IQ2_S, IQ2_XS, IQ3_S and IQ3_XXS.

constexpr int LV = 2 * TILE; // level words of entry l, columns (a, b): LV + 4a + l, pre-shifted
constexpr int NR = LV + 8;   // -r and -2r per lane (shift amounts: negative = right)
constexpr int NB = NR + 2;   // IQ1_S: row group 2 of word-vectors 2 and 3, replicated
constexpr int SG = PM;       // sign words with the parity bits, pre-shifted: SG + pidx(a, b, 0)

// No per-lane value stays in an LReg across transp4: the shift amounts are reloaded from DEST after each transpose.
sfpi_inline vUInt shift_after_transp(const vUInt w, const int amount)
{
    return w << ldu(amount);
}

// Row r of each SFPU lane (r, c): vConstTileId = 2 lane, lane = 8r + c.
sfpi_inline vInt lane_row()
{
    return sfpi::as<vInt>(sfpi::as<vUInt>(sfpi::vConstTileId) >> 4u);
}

// Level words of the codebook formats (IQ2_XXS, IQ1_S): word-vector a's four row groups to every lane, shifted
// right by 2r per lane, into LV + 4a + l; the shift amounts into NR first.
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
        sfpi::dst_reg[LV + 4 * a + 0].mode<sfpi::DataLayout::U32>() = shift_after_transp(w0, NR + 1);
        sfpi::dst_reg[LV + 4 * a + 1].mode<sfpi::DataLayout::U32>() = shift_after_transp(w1, NR + 1);
        sfpi::dst_reg[LV + 4 * a + 2].mode<sfpi::DataLayout::U32>() = shift_after_transp(w2, NR + 1);
        sfpi::dst_reg[LV + 4 * a + 3].mode<sfpi::DataLayout::U32>() = shift_after_transp(w3, NR + 1);
    }
}

sfpi_inline void iq2_xxs_params()
{
    iq_level_words();
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

// IQ1_S: the same level words (values -1, 0, 1 as indices 0..2); row 6 word j = 2c + a holds, in half b, column
// 16a + 2c + b's scale ls (bits 6, 7, 14) and delta sign s (bit 15) among the lookup's high index bits, so row group 2
// of word-vector 2 + a has columns (a, 0) and (a, 1) in its two halves. Value = d (2 ls + 1) * (q - 1 + delta),
// delta = 0.125 (s = 0) or -0.125: ggml's dl * (grid + delta), one rounding. PS = d (2 ls + 1), PM = delta - 1.
sfpi_inline void iq1_s_params()
{
    iq_level_words();
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

// IQ1_M: IQ1_S's pages and level words, with column 16a + 2c + b's delta signs of entries 0..3 in bits 6, 7, 14, 15 of
// half b of row 6 word 2c + a. The 3-bit scales come from the group's scale page: rows 0..2 word 2c + a, field
// f = 4 TI + 2b + h at bit 3 (f % 10) of row f / 10 (fields 30 / 31: bits 0..1 at bits 30..31 of rows 0 / 1, bit 2 at
// bit f of row 2). Value = d (2 ls + 1) * (q - 1 + delta): ggml's dl * (grid + delta),
// one rounding. PS = d (2 ls + 1) per column and half, DM = delta - 1 per column and entry.
constexpr int DM = CO + 16; // rows 32..63 of the page tile, past the page

template <int TI>
sfpi_inline void iq1_m_params()
{
    iq_level_words();
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

// IQ2_S: the same level words and body as IQ2_XXS. Rows 4..5 hold each column's 32 sign bits (no parity bit), row 8
// word j = 2c + a holds, in half b, column 16a + 2c + b's scales of elements 0..15 / 16..31 in bits 4..7 / 12..15
// among the lookup's high index bits, so row group 0 of word-vector 4 + a has columns (a, 0) and (a, 1) in its halves.
// S per column and half: d / 4 (ls + 0.5), as IQ2_XXS.
sfpi_inline void iq2_s_params()
{
    iq_level_words();
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

// IQ2_XS: IQ2_S's level words and scales per column and half, IQ2_XXS's 7-bit sign indices with their parity as the
// eighth bit, grouped per entry: rows 4..5 word 16 (l / 2) + 2c + l % 2 holds entry l of columns 2c, 2c + 1, 16 + 2c,
// 17 + 2c in bytes 0..3 (bit 7: the lookup's ninth index bit), so row groups 0 / 1 of word-vector 2 + p give every lane
// entries p and 2 + p, each column (a, b) in byte 2a + b. The scales come from the group's scale page: rows 0..3 word
// 2c + a, byte 2 (TI % 2) + b of row TI / 2. S = d / 4 (ls + 0.5), as IQ2_S.
template <int TI>
sfpi_inline void iq2_xs_params()
{
    iq_level_words();
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

// IQ3_S: grid entries of 4 elements (elements 4e .. 4e + 3, so element r of every entry belongs to lane row r) with
// 3-bit indices into (1, 3, .., 15). The lookup (variant 5) writes 16-bit level codes, element i at bits 3i, of entry
// e of columns 16a + 2c + b in the two halves of word-vector 2a + e / 4, row group e % 4, and puts each column's 4-bit
// scale in bits 12..15 of its entry 0. Rows 8..9 hold each column's 32 sign bits. Each word-vector's four row groups go
// to every lane; levels are shifted right by 3r, signs by r. Value = (2q + 1) * d (1 + 2 ls), as q * 2S + S, with the
// sign: ggml's db * grid * sign, exactly (every product is exact in fp32).
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

// S, 2 S and the sign word are loaded once per (a, h, b) and the level word per output.
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

// IQ3_XXS: the page holds IQ3_S's level words with 4-bit codes f (element r at bits
// 16b + 4r; f = 0..6 for the grid values 4, 12, .., 52 and 8 for 62), and IQ2_XXS's sign words in rows 8..9 (byte l:
// the 7 sign bits of elements 8l .. 8l + 6, scale bit l in bit 7). Value = (d / 2 (ls + 0.5)) (65 - |61 - 8 f|), sign
// applied: ggml's db * grid * sign, exact. The level words are IQ3_S's, stored unshifted: the body brings element
// r of half b to the top with a left shift by 28 - 16b - 4r per lane (AM3, AM3 + 2). The sign stage is IQ2_XXS's,
// with its masks in programmable constants 0 / 1 (iq3_xxs_init sets them again for the body).
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

// The shift amount once per b, S, its multiple and the sign word once per (b, a), the level word per output.
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

// ---------------------------------------------------------------------------------------------------------------
// Phase 1 per format. TI = tile index inside the scale page's group of K-tiles.

template <int FMT, int TI>
sfpi_inline void params()
{
    if constexpr (FMT != MXFP4 && page_layout<static_cast<Format>(FMT)>().fp16_rows == 0)
    {
        params_init();
    }
#pragma GCC unroll 2
    for (int a = 0; a < 2; a++)
    {
        if constexpr (
            FMT == Q8_0 || FMT == Q4_0 || FMT == IQ4_NL || FMT == TQ2_0 || FMT == Q2_0 || FMT == Q1_0 || FMT == Q4_1 || FMT == Q5_0 || FMT == Q5_1 ||
            FMT == TQ1_0)
        {
            // Row 0: d of columns n0 (low half) and n1 (high half); Q4_1 row 1: m; Q5_0 row TI: d; Q5_1 rows 2 TI,
            // 2 TI + 1: d, m; TQ1_0 row 2: d.
            vUInt r0, r1, r2, r3;
            rep4<FMT>(0, a, r0, r1, r2, r3);
            const vUInt dw = FMT == TQ1_0 || (FMT == Q5_1 && TI % 2) ? r2 : FMT == Q5_0 && TI % 2 ? r1 : r0;
            const vUInt mw = FMT == Q5_1 && TI % 2 ? r3 : r1;
#pragma GCC unroll 2
            for (int b = 0; b < 2; b++)
            {
                vFloat d = b == 0 ? f16_at<0>(dw) : f16_at<1>(dw);
                vFloat m = 0.0f;
                if constexpr (FMT == Q8_0)
                {
                    m = d * -128.0f;
                }
                else if constexpr (FMT == Q4_0)
                {
                    m = d * -8.0f;
                }
                else if constexpr (FMT == Q5_0)
                {
                    m = d * -16.0f;
                }
                else if constexpr (FMT == Q4_1 || FMT == Q5_1)
                {
                    m = b == 0 ? f16_at<0>(mw) : f16_at<1>(mw);
                }
                else if constexpr (FMT == TQ2_0 || FMT == Q2_0 || FMT == TQ1_0)
                {
                    m = -d;
                }
                else if constexpr (FMT == Q1_0)
                {
                    m = -d;
                    d = d * 2.0f;
                }
                sfpi::dst_reg[PS + pidx(a, b, 0)] = d;
                sfpi::dst_reg[PM + pidx(a, b, 0)] = m;
            }
        }
        else if constexpr (FMT == MXFP4)
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
            // Rows 0..5: 12 scale bytes per column; fields 0 / 1: d / dmin.
            rep4_to<FMT>(0, a, 0);
            rep4_to<FMT>(1, a, 4);
            constexpr int j = TI;
#pragma GCC unroll 2
            for (int b = 0; b < 2; b++)
            {
                const int o = 12 * b;
                vUInt sc, mn;
                if constexpr (j < 4)
                {
                    sc = byte_bits(0, o + j, 0, 6);
                    mn = byte_bits(0, o + j + 4, 0, 6);
                }
                else
                {
                    sc = byte_bits(0, o + j + 4, 0, 4) | (byte_bits(0, o + j - 4, 6, 2) << 4);
                    mn = byte_bits(0, o + j + 4, 4, 4) | (byte_bits(0, o + j, 6, 2) << 4);
                }
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
        else if constexpr (FMT == IQ4_XS)
        {
            // Rows 0..2: 96-bit string of 6-bit scales, ls_j(column b) at bit 48b + 6j; field 0: d.
            rep4_to<FMT>(0, a, 0);
#pragma GCC unroll 2
            for (int b = 0; b < 2; b++)
            {
                const int B = 48 * b + 6 * TI;
                const int k = B / 32;
                const int s = B % 32;
                vUInt ls;
                if (s <= 26)
                {
                    ls = bits(ldu(k), s, 6);
                }
                else
                {
                    ls = (ldu(k) >> static_cast<unsigned>(s)) | (bits(ldu(k + 1), 0, s - 26) << static_cast<unsigned>(32 - s));
                }
                sfpi::dst_reg[PS + pidx(a, b, 0)] = pf32<FMT>(0, b, a) * (ufloat(ls) + -32.0f);
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Phase 2: codes. LO_BITS-wide low plane (output o: word-vector lo_base + o / (32 / LO_BITS), field o % (32 / LO_BITS))
// plus an optional HI_BITS-wide high plane; Q3_K stores 3-bit fields instead (q3k_code). Pair pages (Q3_K, Q5_K, Q1_0)
// hold two tiles; CP selects the tile.

// Field f (LO bits wide) of word w, as an unsigned value: two shifts, one for the top field (no mask constant).
template <int LO>
sfpi_inline vUInt field(const vUInt w, const int f)
{
    constexpr int per = 32 / LO;
    if (f == per - 1)
    {
        return w >> static_cast<unsigned>(32 - LO);
    }
    return (w << static_cast<unsigned>(32 - LO - LO * f)) >> static_cast<unsigned>(32 - LO);
}

// Field extraction from code words already in LRegs: lo_w holds output o's low-plane word, hi_w its high-plane word.
// Programmable constant 1 (code_init): the high-plane field's mask at bits LO .. LO + HI - 1, or Q8_0's 0x80.
template <int FMT>
sfpi_inline vUInt code_from(const int o, const vUInt lo_w, const vUInt hi_w)
{
    using T = traits<FMT>;
    vUInt q = field<T::lo>(lo_w, o % (32 / T::lo));
    if constexpr (T::hi > 0)
    {
        // The field's bits moved to bits LO .. LO + HI - 1, the others masked off.
        const int at = T::hi * (o % (32 / T::hi)) - T::lo;
        vUInt hs     = hi_w;
        if (at > 0)
        {
            hs = hi_w >> static_cast<unsigned>(at);
        }
        else if (at < 0)
        {
            hs = hi_w << static_cast<unsigned>(-at);
        }
        q = q | (hs & sfpi::as<vUInt>(sfpi::vConstIntPrgm1));
    }
    if constexpr (FMT == Q8_0)
    {
        q = q ^ sfpi::as<vUInt>(sfpi::vConstIntPrgm1); // int8 -> offset binary; M = -128 S
    }
    return q;
}

template <int FMT>
sfpi_inline void code_init()
{
    using T = traits<FMT>;
    if constexpr (T::hi > 0)
    {
        sfpi::vConstIntPrgm1 = ((1 << T::hi) - 1) << T::lo;
    }
    else if constexpr (FMT == Q8_0)
    {
        sfpi::vConstIntPrgm1 = 0x80;
    }
}

// Q3_K code of output o, 3-bit fields: group g = o / 8 < 3 at bit 3 (o % 8) of word g, group
// 3 in the top byte of words 0..2 (w0, w1, w2; for g < 3 all three are word g).
sfpi_inline vUInt q3k_code(const int o, const vUInt w0, const vUInt w1, const vUInt w2)
{
    const int k = o % 8;
    if (o < 24)
    {
        return bits(w0, 3 * k, 3);
    }
    if (k < 6)
    {
        return bits(k < 2 ? w0 : k < 4 ? w1 : w2, 24 + 3 * (k % 2), 3);
    }
    if (k == 6)
    {
        return (w0 >> 30u) | (bits(w1, 30, 1) << 2u);
    }
    return (w1 >> 31u) | ((w2 >> 30u) << 1u);
}

template <int FMT>
sfpi_inline vUInt code_at(const int o, const int s, const vUInt l0, const vUInt l1, const vUInt hw)
{
    if constexpr (FMT == Q3_K)
    {
        return q3k_code(o, l0, l1, hw);
    }
    else
    {
        return code_from<FMT>(o, s < 2 ? l0 : l1, hw);
    }
}

template <int FMT, int CP>
sfpi_inline int lo_wv(const int o)
{
    using T = traits<FMT>;
    return (T::pair ? CP * T::lo : 0) + o / (32 / T::lo);
}

template <int FMT, int CP>
sfpi_inline int hi_wv(const int o)
{
    using T = traits<FMT>;
    return (T::pair ? 2 * T::lo + CP : T::lo) + o / (32 / (T::hi > 0 ? T::hi : 1));
}

// IQ4_NL value table: round(P4(q)) with P4 = minimax fit whose constant term is the integer -127 (max error 0.4397,
// exact for q = 0..15 under fp32 Horner, fused or not). The constant term is folded into the rounding constant
// 1.5 * 2^23, which keeps it exact; c4..c2 live in the three programmable constant registers (set in iq4_init).
constexpr float IQ4_C4  = 0.0014480184763669968f;
constexpr float IQ4_C3  = 0.03627324849367142f;
constexpr float IQ4_C2  = -1.4385637044906616f;
constexpr float IQ4_C1  = 24.500598907470703f;
constexpr float IQ4_C0R = 12582785.0f; // -127 + 1.5 * 2^23

sfpi_inline void iq4_init()
{
    sfpi::vConstFloatPrgm0 = IQ4_C4;
    sfpi::vConstFloatPrgm1 = IQ4_C3;
    sfpi::vConstFloatPrgm2 = IQ4_C2;
}

// E2M1 magnitudes kv = {0, 1, 2, 3, 4, 6, 8, 12} (doubled values) as nibbles, kv(e) at bits 28 - 4e .. 31 - 4e, so
// that shifting left by 4e brings it to the top nibble.
constexpr unsigned E2M1_TABLE = 0x0123468Cu;

sfpi_inline void e2m1_init()
{
    sfpi::vConstIntPrgm0 = static_cast<int>(E2M1_TABLE);
    sfpi::vConstIntPrgm1 = 0x1C;
}

// E2M1 code in field f of w (sign in the field's bit 3): +-kv as fp32.
sfpi_inline vFloat e2m1_value(const vUInt w, const int f)
{
    // 4 * (code & 7); the mask (programmable constant 1) drops the neighbouring field's bits.
    const vUInt e4 = (f == 0 ? w << 2u : w >> static_cast<unsigned>(4 * f - 2)) & sfpi::as<vUInt>(sfpi::vConstIntPrgm1);
    vUInt t        = sfpi::as<vUInt>(sfpi::vConstIntPrgm0);
    t              = t << e4;
    const vFloat m = ufloat(t >> 28u);
    const vInt sgn = sfpi::as<vInt>(f == 7 ? w : w << static_cast<unsigned>(28 - 4 * f));
    return sfpi::copysgn(m, sgn);
}

template <int FMT, int CP>
sfpi_inline void body()
{
    using T = traits<FMT>;
    if constexpr (T::linear && FMT != Q3_K)
    {
        code_init<FMT>();
    }
    vFloat c1  = 0.0f;
    vFloat c0r = 0.0f;
    if constexpr (FMT == IQ4_NL || FMT == IQ4_XS)
    {
        c1  = IQ4_C1;
        c0r = IQ4_C0R;
    }
#pragma GCC unroll 2
    for (int a = 0; a < 2; a++)
    {
#pragma GCC unroll 2
        for (int h = 0; h < 2; h++)
        {
            // The 8 outputs of (a, h) read at most two low-plane words (one unless 8-bit) and one high-plane word; Q3_K
            // reads its group's word, or all three for group 3.
            const int o0 = 16 * h + 8 * a;
            vUInt l0, l1, hw;
            if constexpr (FMT == Q3_K)
            {
                const int g = o0 / 8;
                l0          = ldu(CO + 3 * CP + (g < 3 ? g : 0));
                l1          = g < 3 ? l0 : ldu(CO + 3 * CP + 1);
                hw          = g < 3 ? l0 : ldu(CO + 3 * CP + 2);
            }
            else
            {
                l0 = ldu(CO + lo_wv<FMT, CP>(o0));
                l1 = T::lo == 8 ? ldu(CO + lo_wv<FMT, CP>(o0 + 4)) : l0;
                hw = l0;
                if constexpr (T::hi > 0)
                {
                    hw = ldu(CO + hi_wv<FMT, CP>(o0));
                }
            }
#pragma GCC unroll 2
            for (int b = 0; b < 2; b++)
            {
                const int pi   = pidx(a, b, T::has_h ? h : 0);
                const vFloat S = sfpi::dst_reg[PS + pi];
                vFloat M       = 0.0f;
                if constexpr (T::linear)
                {
                    M = sfpi::dst_reg[PM + pi];
                }
// Two outputs per step, interleaved: SFPMAD results are consumed two instructions later instead of at once.
#pragma GCC unroll 2
                for (int s = 0; s < 4; s += 2)
                {
                    const int oa = o0 + 2 * s + b;
                    const int ob = oa + 2;
                    if constexpr (T::linear)
                    {
                        const vFloat xa   = ufloat(code_at<FMT>(oa, s, l0, l1, hw));
                        const vFloat xb   = ufloat(code_at<FMT>(ob, s, l0, l1, hw));
                        const vFloat va   = xa * S + M;
                        const vFloat vb   = xb * S + M;
                        sfpi::dst_reg[oa] = va;
                        sfpi::dst_reg[ob] = vb;
                    }
                    else if constexpr (FMT == IQ4_NL || FMT == IQ4_XS)
                    {
                        const vFloat xa = ufloat(code_from<FMT>(oa, l0, hw));
                        const vFloat xb = ufloat(code_from<FMT>(ob, l0, hw));
                        vFloat pa       = xa * sfpi::vConstFloatPrgm0 + sfpi::vConstFloatPrgm1;
                        vFloat pb       = xb * sfpi::vConstFloatPrgm0 + sfpi::vConstFloatPrgm1;
                        pa              = pa * xa + sfpi::vConstFloatPrgm2;
                        pb              = pb * xb + sfpi::vConstFloatPrgm2;
                        pa              = pa * xa + c1;
                        pb              = pb * xb + c1;
                        pa              = pa * xa + c0r;
                        pb              = pb * xb + c0r;
                        pa              = pa + -12582912.0f;
                        pb              = pb + -12582912.0f;
                        // Both products before either store: a store right after its SFPMUL stalls a cycle.
                        const vFloat ra   = pa * S;
                        const vFloat rb   = pb * S;
                        sfpi::dst_reg[oa] = ra;
                        sfpi::dst_reg[ob] = rb;
                    }
                    else
                    {
                        const vFloat va   = e2m1_value(l0, oa % 8);
                        const vFloat vb   = e2m1_value(l0, ob % 8);
                        sfpi::dst_reg[oa] = va * S;
                        sfpi::dst_reg[ob] = vb * S;
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Q4_0, Q4_1 and Q4_K: body() through SFPLOADMACRO, with the same arithmetic. Per output, a macro on the output's row
// runs q >> 28, its conversion to fp32, S q + M into LReg16 and the store of LReg16, while the regular instruction
// after the macro moves the output's field to the top of the code word. ttsim does not model the macros: under
// DISABLE_SFPLOADMACRO these formats use body().

#ifdef DISABLE_SFPLOADMACRO
constexpr bool LOADMACRO = false;
#else
constexpr bool LOADMACRO = true;
#endif

template <int FMT>
constexpr bool macro_body = LOADMACRO && (FMT == Q4_0 || FMT == Q4_1 || FMT == Q4_K);

constexpr std::uint32_t LOAD_NONE      = 11; // SFPLOAD mode that loads nothing: the macro only takes the row
constexpr std::uint32_t SHIFT_IMM_OF_C = 5;  // SFPSHFT: VD = LReg[VC] shifted by the immediate

// Instruction templates: 0 = x >> 28 (amount in LReg14), 1 = int to fp32, 2 / 3 = S x + M with S, M in L4, L5 /
// L6, L7. Macro b (column half b) runs the shift, cast, MAD (into LReg16) and store of LReg16 1, 2, 3 and 5
// instructions after it.
template <int FMT>
inline void macro_init()
{
    if constexpr (macro_body<FMT>)
    {
        sfpi::vConstIntPrgm2 = -28;
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
}

// Output field i of code word 2h + a (L3) through macro i % 2, with L0..L2 in turn. Each output takes three slots: the
// macro, the shift and one without a Simple instruction, where its Round-unit shift runs (a Simple and a Round
// instruction in one cycle must not both write LRegs 0..7); that slot loads the next code word or S, M when due. S, M
// of the next a load once the MADs reading the old ones have run; the last store runs before the body returns.
sfpi_inline void body_macro()
{
    TTI_SFPLOAD(3, sfpi::SFPLOAD_MOD0_FMT_INT32, ADDR_MOD_7, 2 * CO);
    TTI_SFPLOAD(4, sfpi::SFPLOAD_MOD0_FMT_SRCB, ADDR_MOD_7, 2 * (PS + pidx(0, 0, 0)));
    TTI_SFPLOAD(5, sfpi::SFPLOAD_MOD0_FMT_SRCB, ADDR_MOD_7, 2 * (PM + pidx(0, 0, 0)));
    TTI_SFPLOAD(6, sfpi::SFPLOAD_MOD0_FMT_SRCB, ADDR_MOD_7, 2 * (PS + pidx(0, 1, 0)));
    TTI_SFPLOAD(7, sfpi::SFPLOAD_MOD0_FMT_SRCB, ADDR_MOD_7, 2 * (PM + pidx(0, 1, 0)));
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

// ---------------------------------------------------------------------------------------------------------------
// TQ1_0: one page holds the group's eight K tiles as the GGUF bytes. Byte j of lane (r, c):
//   word-vector 4a + 2h + b (tiles 0..4): qs[16h + 4j + r] of column 16a + 2c + b;
//   word-vector 8 + 2a + b (tiles 5..7): qs[32 + 4j + r];
//   scale rows 0 / 1, word 2c + a: qh of columns (a, 0) / (a, 1); row 2: d.
// Output o = 16h + 8a + 2j + b of tile TI is trit D of its byte, ((x 3^D) mod 256) 3 >> 8 as in dequantize_row_tq1_0:
// D = TI for tiles 0..4, 2 (TI - 5) + h for tiles 5 and 6, 4 for h = 0 of tile 7; h = 1 of tile 7 is trit j of qh[r].
// Value = trit * d - d.
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

// Phase 1 of one tile. TI = the tile's index in its group of K tiles.
template <int FMT, int TI>
inline void _calculate_gguf_decode_params_()
{
    if constexpr (FMT == IQ2_XXS)
    {
        iq2_xxs_params();
    }
    else if constexpr (FMT == IQ1_S)
    {
        iq1_s_params();
    }
    else if constexpr (FMT == IQ2_S)
    {
        iq2_s_params();
    }
    else if constexpr (FMT == IQ3_S)
    {
        iq3_s_params();
    }
    else if constexpr (FMT == IQ3_XXS)
    {
        iq3_xxs_params();
    }
    else if constexpr (FMT == IQ2_XS)
    {
        iq2_xs_params<TI>();
    }
    else if constexpr (FMT == IQ1_M)
    {
        iq1_m_params<TI>();
    }
    else
    {
        params<FMT, TI>();
    }
}

// IQ2_XS decodes in two calls, its tile-index phase 1 and the phase 2 below, shared by every tile index: with phase 2
// inlined into each of the eight phase 1 instantiations its decode wrote zeros in a matmul kernel.
template <int FMT>
constexpr bool two_calls = FMT == IQ2_XS;

// Phase 2 of one tile (IQ2_XS: of every tile).
template <int FMT, int TI>
inline void _calculate_gguf_decode_body_()
{
    if constexpr (FMT == IQ2_XXS)
    {
        iq2_xxs_init();
        iq2_xxs_body<false>();
    }
    else if constexpr (FMT == IQ1_S)
    {
        iq1_s_body<false>();
    }
    else if constexpr (FMT == IQ2_S)
    {
        iq2_xxs_init();
        iq2_xxs_body<true>();
    }
    else if constexpr (FMT == IQ3_S)
    {
        iq3_s_body();
    }
    else if constexpr (FMT == IQ3_XXS)
    {
        iq3_xxs_init();
        iq3_xxs_body();
    }
    else if constexpr (FMT == IQ2_XS)
    {
        iq2_xxs_init();
        iq2_xxs_body<true, true>();
    }
    else if constexpr (FMT == IQ1_M)
    {
        iq1_s_body<true>();
    }
    else if constexpr (FMT == TQ1_0)
    {
        tq1_0_body<TI>();
    }
    else if constexpr (macro_body<FMT>)
    {
        body_macro();
    }
    else
    {
        if constexpr (FMT == IQ4_NL || FMT == IQ4_XS)
        {
            iq4_init();
        }
        else if constexpr (FMT == MXFP4 || FMT == NVFP4)
        {
            e2m1_init();
        }
        body<FMT, traits<FMT>::pair ? TI % 2 : 0>();
    }
}

} // namespace gguf

// The SFPU at DEST tile dst_index, `rows` rows further on (a page tile holding several pages: the page `rows` rows
// into tile dst_index + 1, the output over the rows before it).
inline void sfpu_gguf_decode_start(const std::uint32_t dst_index, const std::uint32_t rows)
{
    TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, (dst_index << 6) + rows + get_dest_buffer_base());
    TTI_STALLWAIT(p_stall::STALL_SFPU, p_stall::MATH);
}

inline void sfpu_gguf_decode_done()
{
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
}

template <int FMT, int TI, bool PARAMS, bool BODY>
inline void sfpu_gguf_decode_tile_ti(const std::uint32_t dst_index, const std::uint32_t rows)
{
    sfpu_gguf_decode_start(dst_index, rows);
    if constexpr (PARAMS)
    {
        gguf::_calculate_gguf_decode_params_<FMT, TI>();
    }
    if constexpr (BODY && !gguf::two_calls<FMT>)
    {
        gguf::_calculate_gguf_decode_body_<FMT, TI>();
    }
    sfpu_gguf_decode_done();
}

/**
 * @brief Set up the SFPU for @ref sfpu_gguf_decode_tile of format FMT.
 *
 * @tparam FMT: ckernel::gguf::Format of the pages.
 * @note Call after _llk_math_eltwise_unary_sfpu_init_, and again after other SFPU code that programs the load macros
 *       or programmable constant 2.
 */
template <int FMT>
inline void sfpu_gguf_decode_init()
{
    gguf::macro_init<FMT>();
}

/**
 * @brief Decode one GGUF weight tile into a Float32 [K][N] tile in DEST, scales applied (ggml's dequantization).
 *
 * @tparam FMT: ckernel::gguf::Format of the page.
 * @tparam PARAMS, BODY: Run phase 1 (the scales) and phase 2 (the codes). A decoded tile needs both; one alone is for
 *                       timing it.
 * @param dst_index: DEST tile of the output. The page must be in tile dst_index + 1 (unpacked to DEST as UInt32);
 *                   tile dst_index + 2 is scratch.
 * @param ti: The tile's index in its group of K tiles (page_layout<FMT>().group_tiles): the formats whose scales or
 *            page differ per tile select them by it.
 * @param rows: DEST row offset of the page inside tile dst_index + 1 (several pages unpacked as one tile); the output
 *              tile and the scratch tile move with it.
 * @note Needs a 32-bit DEST and @ref sfpu_gguf_decode_init. Overwrites the programmable constants and may record
 *       into the replay buffer.
 */
template <int FMT, bool PARAMS = true, bool BODY = true>
inline void sfpu_gguf_decode_tile(const std::uint32_t dst_index, const std::uint32_t ti, const std::uint32_t rows = 0)
{
    using namespace gguf;
    if constexpr (FMT == MXFP4 || FMT == Q1_0 || FMT == Q5_0 || FMT == Q5_1)
    {
        if (ti & 1)
        {
            sfpu_gguf_decode_tile_ti<FMT, 1, PARAMS, BODY>(dst_index, rows);
        }
        else
        {
            sfpu_gguf_decode_tile_ti<FMT, 0, PARAMS, BODY>(dst_index, rows);
        }
    }
    else if constexpr (!(FMT == Q6_K || FMT == Q4_K || FMT == Q5_K || FMT == Q3_K || FMT == Q2_K || FMT == IQ4_XS || FMT == IQ2_XS || FMT == IQ1_M ||
                         FMT == TQ1_0))
    {
        sfpu_gguf_decode_tile_ti<FMT, 0, PARAMS, BODY>(dst_index, rows);
    }
    else
    {
        switch (ti)
        {
            case 0:
                sfpu_gguf_decode_tile_ti<FMT, 0, PARAMS, BODY>(dst_index, rows);
                break;
            case 1:
                sfpu_gguf_decode_tile_ti<FMT, 1, PARAMS, BODY>(dst_index, rows);
                break;
            case 2:
                sfpu_gguf_decode_tile_ti<FMT, 2, PARAMS, BODY>(dst_index, rows);
                break;
            case 3:
                sfpu_gguf_decode_tile_ti<FMT, 3, PARAMS, BODY>(dst_index, rows);
                break;
            case 4:
                sfpu_gguf_decode_tile_ti<FMT, 4, PARAMS, BODY>(dst_index, rows);
                break;
            case 5:
                sfpu_gguf_decode_tile_ti<FMT, 5, PARAMS, BODY>(dst_index, rows);
                break;
            case 6:
                sfpu_gguf_decode_tile_ti<FMT, 6, PARAMS, BODY>(dst_index, rows);
                break;
            default:
                sfpu_gguf_decode_tile_ti<FMT, 7, PARAMS, BODY>(dst_index, rows);
                break;
        }
    }
    if constexpr (BODY && gguf::two_calls<FMT>)
    {
        sfpu_gguf_decode_start(dst_index, rows);
        gguf::_calculate_gguf_decode_body_<FMT, 0>();
        sfpu_gguf_decode_done();
    }
}

} // namespace ckernel::sfpu
