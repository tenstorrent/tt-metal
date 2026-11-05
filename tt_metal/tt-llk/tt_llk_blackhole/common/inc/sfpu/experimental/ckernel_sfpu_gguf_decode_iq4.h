// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// GGUF SFPU decoders of IQ4_NL and IQ4_XS.

#pragma once

#include <cstdint>
#include <utility>

#include "sfpu/experimental/ckernel_sfpu_gguf_decode_common.h"

namespace ckernel::sfpu
{
namespace gguf
{

using namespace ckernel::gguf;

sfpi_inline void iq4_init()
{
    sfpi::vConstFloatPrgm0 = IQ4_C4;
    sfpi::vConstFloatPrgm1 = IQ4_C3;
    sfpi::vConstFloatPrgm2 = IQ4_C2;
}

// Phase 1 of IQ4_XS (IQ4_NL: linear_params). TI = tile index inside the scale page's group of K-tiles.
template <int FMT, int TI>
sfpi_inline void iq4_params()
{
    if constexpr (page_layout<static_cast<Format>(FMT)>().fp16_rows == 0)
    {
        params_init();
    }
#pragma GCC unroll 2
    for (int a = 0; a < 2; a++)
    {
        if constexpr (FMT == IQ4_XS)
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

// IQ4_NL and IQ4_XS through SFPLOADMACRO (body_macro_iq4).
template <int FMT>
constexpr bool iq4_macro_body = LOADMACRO && (FMT == IQ4_NL || FMT == IQ4_XS);

// IQ4 instruction templates (body_macro_iq4): 0 = v >> 28 (amount in L7), 1 = int to fp32, 2 = TF32 round to nearest,
// 3 = S u + M (L5, L6). Macro 0 runs the shift and the cast 2 and 3 instructions after it; macro 1 the round, the MAD
// into LReg16 and the store of LReg16 4, 5 and 7 instructions after it.
inline void macro_init_iq4()
{
    TTI_SFPSHFT2(0, 7, 12, sfpi::SFPSHFT2_MOD1_SHFT_LREG);
    TTI_SFPCAST(0, 13, sfpi::SFPCAST_MOD1_SM32_TO_FP32_RNE);
    TTI_SFP_STOCH_RND(sfpi::SFPSTOCHRND_RND_EVEN, 0, 9, 0, 14, sfpi::SFPSTOCHRND_MOD1_FP32_TO_FP16A);
    TTI_SFPMAD(5, 0, 6, 15, sfpi::SFPMAD_MOD1_OFFSET_NONE);
    constexpr std::uint32_t cast  = (2 << 3) | 5;
    constexpr std::uint32_t shift = 0x80 | (1 << 3) | 4;
    constexpr std::uint32_t mad   = 0xC0 | (4 << 3) | 7;
    constexpr std::uint32_t round = (3 << 3) | 6;
    constexpr std::uint32_t store = 0x40 | (6 << 3) | 3;
    TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, cast);
    TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, shift);
    TTI_SFPCONFIG(0, 4, 0);
    TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, mad << 8);
    TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store << 8) | round);
    TTI_SFPCONFIG(0, 5, 0);
    TTI_SFPCONFIG(0xF00, 8, sfpi::SFPCONFIG_MOD1_ARG_IMM);
    TTI_SFPNOP;
}

// ---------------------------------------------------------------------------------------------------------------
// IQ4_NL / IQ4_XS through SFPLOADMACRO. The level is a degree-4 polynomial of q (minimax, max error 0.433) whose
// constant term carries K = -1536, so a TF32 round to nearest gives kv + K exactly, and S (kv + K) + M with M = -K S
// rounds once to kv S like body(). Zeros may differ in sign from body() (IQ4_XS sub-blocks with ls = 32 give +0 where
// body() gives -0 for negative levels). Stream position k:
//   7k          macro 0 on v = L[k % 2] loads the code word; v >> 28 and the cast run 2 and 3 instructions on
//   7k + 1      v <<= 28 - 4i
//   7k + 4,6,9  Horner into t = L[3 + k % 2] (c4, c3, c2, c1 in L12, L13, L14, L11)
//   7k + 10     macro 1 on L2 (loads zero, output row); 7k + 12: L2 = t v + c0K (L7); the round, S L2 + M into
//               LReg16 and its store run at 7k + 14, 15 and 17.
// Outputs go by scale group (a, b), b = i % 2. S, M (L5, L6) reload right after the group's last MAD, where only the
// first two macro ops of a later output are pending. The low five bits of c0K make L7 the shift amount -28. L11 gets
// back its -1.0 at the end.
constexpr std::uint32_t IQ4M_C4    = 0x3ACFE2C4;
constexpr std::uint32_t IQ4M_C3    = 0x3CFEA8FD;
constexpr std::uint32_t IQ4M_C2    = 0xBFAF9B4E;
constexpr std::uint32_t IQ4M_C1    = 0x41C16401;
constexpr std::uint32_t IQ4M_C0K   = 0xC4CFD224; // c0 - 1536
constexpr std::uint32_t IQ4M_NEG_K = 0x44C0;     // 1536, bf16

enum : int
{
    QS_NOP,
    QS_WORD,  // macro 0, v = x, code word row y
    QS_SHIFT, // v x <<= y
    QS_STEP,  // Horner step z (1..4) of t x, v y
    QS_OUT,   // macro 1, output row x
    QS_SCALE, // register x = S of row y
    QS_NEGK,  // register x *= -K
};

constexpr int QBASE  = 7 * 31 + 18;
constexpr int QSLOTS = QBASE + 9;

struct qplan
{
    mslot s[QSLOTS];
    int n;
};

// Stream position k: scale group g = k / 8 = (a, b), then h, then i = b, b + 2, b + 4, b + 6.
constexpr qplan make_qplan()
{
    qplan p {};
    mslot base[QBASE] {};
    for (int k = 0; k < 32; k++)
    {
        const int g = k / 8, a = g / 2, b = g % 2, h = k / 4 % 2, i = b + 2 * (k % 4);
        const int v = k % 2, t = 3 + k % 2, s = 7 * k;
        base[s]      = {QS_WORD, v, CO + 2 * h + a, 0};
        base[s + 1]  = {QS_SHIFT, v, 28 - 4 * i, 0};
        base[s + 4]  = {QS_STEP, t, v, 1};
        base[s + 6]  = {QS_STEP, t, v, 2};
        base[s + 9]  = {QS_STEP, t, v, 3};
        base[s + 10] = {QS_OUT, 16 * h + 8 * a + i, 0, 0};
        base[s + 12] = {QS_STEP, t, v, 4};
    }
    for (int q = 0; q < QBASE; q++)
    {
        p.s[p.n++]  = base[q];
        const int k = (q - 15) / 7;
        if (q >= 15 && (q - 15) % 7 == 0 && k % 8 == 7 && k < 31)
        {
            const int g = k / 8 + 1;
            p.s[p.n++]  = {QS_SCALE, 5, PS + pidx(g / 2, g % 2, 0), 0};
            p.s[p.n++]  = {QS_SCALE, 6, PS + pidx(g / 2, g % 2, 0), 0};
            p.s[p.n++]  = {QS_NEGK, 6, 0, 0};
        }
    }
    return p;
}

constexpr qplan qplan_iq4 = make_qplan();

template <int I>
sfpi_inline void qslot_issue()
{
    constexpr mslot s = qplan_iq4.s[I];
    if constexpr (s.kind == QS_WORD)
    {
        TTI_SFPLOADMACRO(s.x, sfpi::SFPLOAD_MOD0_FMT_INT32, ADDR_MOD_7, 2 * s.y);
    }
    else if constexpr (s.kind == QS_SHIFT)
    {
        TTI_SFPSHFT(s.y, s.x, s.x, SHIFT_IMM_OF_C);
    }
    else if constexpr (s.kind == QS_STEP)
    {
        constexpr int a = s.z == 1 ? 12 : s.x;
        constexpr int c = s.z == 1 ? 13 : s.z == 2 ? 14 : s.z == 3 ? 11 : 7;
        TTI_SFPMAD(a, s.y, c, s.z == 4 ? 2 : s.x, sfpi::SFPMAD_MOD1_OFFSET_NONE);
    }
    else if constexpr (s.kind == QS_OUT)
    {
        TTI_SFPLOADMACRO((1 << 2) | 2, LOAD_NONE, ADDR_MOD_7, 2 * s.x);
    }
    else if constexpr (s.kind == QS_SCALE)
    {
        TTI_SFPLOAD(s.x, sfpi::SFPLOAD_MOD0_FMT_SRCB, ADDR_MOD_7, 2 * s.y);
    }
    else if constexpr (s.kind == QS_NEGK)
    {
        TTI_SFPMULI(IQ4M_NEG_K, s.x, 0);
    }
    else
    {
        TTI_SFPNOP;
    }
}

template <int... I>
sfpi_inline void qslots_issue(std::integer_sequence<int, I...>)
{
    (qslot_issue<I>(), ...);
}

template <int REG, std::uint32_t BITS>
sfpi_inline void lreg_bits()
{
    TTI_SFPLOADI(REG, sfpi::SFPLOADI_MOD0_LOWER, BITS & 0xFFFF);
    TTI_SFPLOADI(REG, sfpi::SFPLOADI_MOD0_UPPER, BITS >> 16);
}

sfpi_inline void body_macro_iq4()
{
    lreg_bits<0, IQ4M_C1>();
    TTI_SFPCONFIG(0, 11, 0);
    lreg_bits<0, IQ4M_C4>();
    TTI_SFPCONFIG(0, 12, 0);
    lreg_bits<0, IQ4M_C3>();
    TTI_SFPCONFIG(0, 13, 0);
    lreg_bits<0, IQ4M_C2>();
    TTI_SFPCONFIG(0, 14, 0);
    lreg_bits<7, IQ4M_C0K>();
    TTI_SFPLOAD(5, sfpi::SFPLOAD_MOD0_FMT_SRCB, ADDR_MOD_7, 2 * (PS + pidx(0, 0, 0)));
    TTI_SFPLOAD(6, sfpi::SFPLOAD_MOD0_FMT_SRCB, ADDR_MOD_7, 2 * (PS + pidx(0, 0, 0)));
    TTI_SFPMULI(IQ4M_NEG_K, 6, 0);
    qslots_issue(std::make_integer_sequence<int, qplan_iq4.n> {});
    TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_FLOATB, 0xBF80);
    TTI_SFPCONFIG(0, 11, 0);
}

// One copy for IQ4_XS's eight tile indices (inlined per tile index, its instruction fetches are slower).
inline __attribute__((noinline)) void body_macro_iq4_shared()
{
    body_macro_iq4();
}

} // namespace gguf
} // namespace ckernel::sfpu
