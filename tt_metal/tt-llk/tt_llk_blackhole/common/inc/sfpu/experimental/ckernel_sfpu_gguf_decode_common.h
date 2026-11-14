// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Shared parts of the GGUF SFPU decoders (ckernel_sfpu_gguf_decode.h): DEST layout constants, bit and scale-field
// helpers, the code-word layout of the linear formats and the generic phase 2 (body).

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

// SFPTRANSP of LRegs 0..3 (4..7 are scratch). Uses all eight LRegs: no other value stays in an LReg across it.
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
struct traits; // ckernel_sfpu_gguf_decode.h

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

// rep4 into dst_reg[base + 0..3] of the output tile, which phase 2 overwrites.
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

// Bit field [lsb, lsb + width) of w, unsigned.
sfpi_inline vUInt bits(const vUInt w, const int lsb, const int width)
{
    if (lsb + width == 32)
    {
        return lsb == 0 ? w : w >> static_cast<unsigned>(lsb);
    }
    return (w << static_cast<unsigned>(32 - lsb - width)) >> static_cast<unsigned>(32 - width);
}

// Prepared formats (fp16_rows > 0): the fp32 copy of fp16 field F of column 16a + 2c + b, written with the page;
// group_scales formats: by the group step, past the page.
template <int FMT>
constexpr int pf32_idx(const int F, const int b, const int a)
{
    if constexpr (group_scales<FMT>)
    {
        return CO + static_cast<int>(page_layout<static_cast<Format>(FMT)>().rows) / 2 + 2 * (2 * F + b) + a;
    }
    constexpr int fb = round_up(page_layout<static_cast<Format>(FMT)>().scale_rows, 4);
    return CO + scale_row<FMT>() / 2 + 2 * (fb / 4 + 2 * F + b) + a;
}

template <int FMT>
sfpi_inline vFloat pf32(const int F, const int b, const int a)
{
    return sfpi::dst_reg[pf32_idx<FMT>(F, b, a)];
}

// Byte B (0..23) of a lane's scale bytes in scratch rows base + B / 4 (rep4_to copies): field [lsb, lsb + width).
sfpi_inline vUInt byte_bits(const int base, const int B, const int lsb, const int width)
{
    return bits(ldu(base + B / 4), 8 * (B % 4) + lsb, width);
}

// f16_at and ue4m3_half: b rebias-adds the field into the fp32 exponent (exact for normals), a = 2 b - smallest normal
// (exact for subnormals), the value is min(a, b). Programmable constants 0 = 112 << 23 (fp16), 1 = 119 << 23 (UE4M3).
// PRGM1 = false: constant 1 unused (Q8_0, Q4_0, Q5_0, Q4_1, Q5_1).
template <bool PRGM1 = true>
sfpi_inline void params_init()
{
    sfpi::vConstIntPrgm0 = 0x38000000;
    if constexpr (PRGM1)
    {
        sfpi::vConstIntPrgm1 = 0x3B800000;
    }
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

// f16_at<0> and f16_at<1> of w (the same operations), the two chains interleaved: no SFPMAD result is read by the next
// instruction. bias: 112 << 23, by default programmable constant 0 (params_init).
template <typename Bias = decltype(sfpi::vConstIntPrgm0)>
sfpi_inline void f16_both(const vUInt w, vFloat& lo, vFloat& hi, const Bias& bias = sfpi::vConstIntPrgm0)
{
    const vUInt x   = w << 16u;
    vUInt m1        = w >> 16u;
    vUInt m0        = x << 1u;
    m0              = m0 >> 4u;
    m1              = m1 << 17u;
    m1              = m1 >> 4u;
    const vFloat b0 = sfpi::as<vFloat>(m0 + sfpi::as<vUInt>(bias));
    const vFloat b1 = sfpi::as<vFloat>(m1 + sfpi::as<vUInt>(bias));
    vFloat a0       = sfpi::addexp(b0, 1);
    vFloat a1       = sfpi::addexp(b1, 1);
    a0              = a0 + -6.103515625e-05f;
    a1              = a1 + -6.103515625e-05f;
    lo              = sfpi::copysgn(sfpi::min_max(a0, b0).first, sfpi::as<vInt>(x));
    hi              = sfpi::copysgn(sfpi::min_max(a1, b1).first, sfpi::as<vInt>(w));
}

constexpr int pidx(const int a, const int b, const int h)
{
    return (a * 2 + b) * 2 + h;
}

// Shift amounts are reloaded from DEST: no per-lane value stays in an LReg across transp4.
sfpi_inline vUInt shift_after_transp(const vUInt w, const int amount)
{
    return w << ldu(amount);
}

// Row r of each SFPU lane (r, c): vConstTileId = 2 lane, lane = 8r + c.
sfpi_inline vInt lane_row()
{
    return sfpi::as<vInt>(sfpi::as<vUInt>(sfpi::vConstTileId) >> 4u);
}

// Linear code layout: one B-bit code per output (Q6_K: ql | qh << 4) in the tile's B code word-vectors (pair formats:
// tile CP's from word-vector B CP). Stream position k is row mrow(k), its own inverse. B dividing 32: row o is field
// o % F of word o / F. Otherwise position k < F B is field k % F of word k / F, and bit j of the words' top-2-bit
// string (word 0 bits 30, 31, word 1 bits 30, 31, ..) is bit j % B of position F B + j / B.

template <int FMT>
struct mfield
{
    static constexpr int B     = traits<FMT>::lo + traits<FMT>::hi;
    static constexpr int F     = 32 / B;
    static constexpr int clean = 32 % B == 0 ? 32 : F * B; // stream positions with their field in one word
};

constexpr int mrow(const int k)
{
    return 16 * (k / 8 % 2) + 8 * (k / 16) + k % 8;
}

template <int FMT, int CP>
constexpr int mbase()
{
    return traits<FMT>::pair ? CP * mfield<FMT>::B : 0;
}

template <int FMT, int CP>
constexpr int mword(const int k)
{
    using M = mfield<FMT>;
    return mbase<FMT, CP>() + (32 % M::B == 0 ? mrow(k) / M::F : k / M::F);
}

template <int FMT>
constexpr int mlsb(const int k)
{
    using M = mfield<FMT>;
    return M::B * (32 % M::B == 0 ? mrow(k) % M::F : k % M::F);
}

// Field f (LO bits wide) of word w, unsigned.
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

// Code of row o, unsigned, from DEST (B not dividing 32).
template <int FMT, int CP>
sfpi_inline vUInt code_of(const int o)
{
    using M     = mfield<FMT>;
    const int k = mrow(o);
    if (k < M::clean)
    {
        return bits(ldu(CO + mword<FMT, CP>(k)), mlsb<FMT>(k), M::B);
    }
    const int j0 = M::B * (k - M::clean);
    vUInt q      = 0;
#pragma GCC unroll 4
    for (int j = j0; j < j0 + M::B; j = j / 2 * 2 + 2)
    {
        const int n       = (j / 2 * 2 + 2 < j0 + M::B ? j / 2 * 2 + 2 : j0 + M::B) - j;
        const vUInt piece = bits(ldu(CO + mbase<FMT, CP>() + j / 2), M::F * M::B + j % 2, n);
        q                 = q | (j == j0 ? piece : piece << static_cast<unsigned>(j - j0));
    }
    return q;
}

// Code of row o, unsigned; w holds the row's code word when B divides 32. Q8_0: offset binary (M = -128 S) through
// programmable constant 1 = 0x80.
template <int FMT, int CP>
sfpi_inline vUInt code_at(const int o, const vUInt w)
{
    using M = mfield<FMT>;
    if constexpr (32 % M::B != 0)
    {
        return code_of<FMT, CP>(o);
    }
    else if constexpr (FMT == Q8_0)
    {
        return field<M::B>(w, o % M::F) ^ sfpi::as<vUInt>(sfpi::vConstIntPrgm1);
    }
    else
    {
        return field<M::B>(w, o % M::F);
    }
}

template <int FMT, int CP>
sfpi_inline void body()
{
    using T = traits<FMT>;
    using M = mfield<FMT>;
    if constexpr (FMT == Q8_0)
    {
        sfpi::vConstIntPrgm1 = 0x80;
    }
#pragma GCC unroll 2
    for (int a = 0; a < 2; a++)
    {
#pragma GCC unroll 2
        for (int h = 0; h < 2; h++)
        {
            // B dividing 32: one code word per (a, h), two for 8-bit codes.
            const int o0 = 16 * h + 8 * a;
            vUInt l0, l1;
            if constexpr (32 % M::B == 0)
            {
                l0 = ldu(CO + mbase<FMT, CP>() + o0 / M::F);
                l1 = M::B == 8 ? ldu(CO + mbase<FMT, CP>() + (o0 + 4) / M::F) : l0;
            }
#pragma GCC unroll 2
            for (int b = 0; b < 2; b++)
            {
                const int pi   = pidx(a, b, T::has_h ? h : 0);
                const vFloat S = sfpi::dst_reg[PS + pi];
                const vFloat M = sfpi::dst_reg[PM + pi];
// Two outputs per step, interleaved to hide the SFPMAD latency.
#pragma GCC unroll 2
                for (int s = 0; s < 4; s += 2)
                {
                    const int oa      = o0 + 2 * s + b;
                    const int ob      = oa + 2;
                    const vFloat xa   = ufloat(code_at<FMT, CP>(oa, s < 2 ? l0 : l1));
                    const vFloat xb   = ufloat(code_at<FMT, CP>(ob, s < 2 ? l0 : l1));
                    const vFloat va   = xa * S + M;
                    const vFloat vb   = xb * S + M;
                    sfpi::dst_reg[oa] = va;
                    sfpi::dst_reg[ob] = vb;
                }
            }
        }
    }
}

// SFPLOADMACRO bodies (linear, FP4 and IQ4 families).

#ifdef DISABLE_SFPLOADMACRO
constexpr bool LOADMACRO = false;
#else
constexpr bool LOADMACRO = true;
#endif

constexpr std::uint32_t LOAD_NONE      = 11; // SFPLOAD mode that loads nothing: the macro only takes the row
constexpr std::uint32_t SHIFT_IMM_OF_C = 5;  // SFPSHFT: VD = LReg[VC] shifted by the immediate

struct mslot
{
    int kind, x, y, z;
};

} // namespace gguf
} // namespace ckernel::sfpu
