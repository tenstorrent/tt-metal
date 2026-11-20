// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// GGUF SFPU decoders of IQ4_NL and IQ4_XS, whose pages hold their codes as 4-bit block-float datums.

#pragma once

#include <cstdint>
#include <utility>

#include "sfpu/experimental/ckernel_sfpu_gguf_decode_common.h"

namespace ckernel::sfpu
{
namespace gguf
{

using namespace ckernel::gguf;

// The page's codes are BFP4 datums in the [K][N] tile's face order, unpacked with the exponent forced to 15 into the
// 16-bit DEST rows that the output tile's rows address (as MXFP4 / NVFP4), so the SFPU loads x = +-c / 4 (FP16) where
// its output goes; the code of sign 1 and magnitude 0 loads as x8 = -131008. The host maps the IQ4 codes 0..14 (the
// values in increasing order) to x = 1.75 .. -1.75 and code 15 to x8 (bfp_codes in the host packing). Then
// w = kv + 14 = round(p(x)) for the cubic p below (error < 0.45 on the fifteen finite codes) and p(x8) > 127, which
// SFPSTOCHRND's clamp to 127 turns into the largest value: kv = 113. The output is S w + M with M = -14 S.
constexpr std::uint32_t IQ4B_NEG14 = 0xC160;     // -14, bf16
constexpr std::uint32_t IQ4B_C3    = 0xC09F69F7; // -4.981685
constexpr std::uint32_t IQ4B_C2    = 0xC03C14E6; // -2.938776
constexpr std::uint32_t IQ4B_C1    = 0xC239D4DD; // -46.457874
constexpr std::uint32_t IQ4B_C0    = 0x40639783; // 3.5561225

// S of group p = pidx(a, b, 0), past the page.
template <int FMT>
constexpr int iq4_ps()
{
    return CO + static_cast<int>(page_layout<static_cast<Format>(FMT)>().rows) / 2;
}

// Phase 1: S of the four column groups. IQ4_NL: row 0 holds d of columns n0 (low half) and n1 (high half); IQ4_XS:
// rows 0..2 hold the 96-bit string of 6-bit scales, ls_j(column b) at bit 48 b + 6 j, and S = d (ls - 32). The two
// columns' chains interleave (no SFPMAD result used at once). TI = tile index in the group.
template <int FMT, int TI>
sfpi_inline void iq4_params()
{
#pragma GCC unroll 2
    for (int a = 0; a < 2; a++)
    {
        vFloat f0, f1;
        if constexpr (FMT == IQ4_NL)
        {
            vUInt r0, r1, r2, r3;
            rep4<FMT>(0, a, r0, r1, r2, r3);
            const vUInt bias = 0x38000000u; // after rep4, which needs four registers
            f16_both(r0, f0, f1, bias);
        }
        else
        {
            vUInt r0, r1, r2, r3;
            rep4<FMT>(0, a, r0, r1, r2, r3);
            (void)r3;
            constexpr int B0 = 6 * TI, B1 = 48 + 6 * TI;
            constexpr int k0 = B0 / 32, s0 = B0 % 32, k1 = B1 / 32, s1 = B1 % 32;
            const vUInt w0 = k0 == 0 ? r0 : r1;
            const vUInt w1 = k1 == 1 ? r1 : r2;
            vUInt l0, l1;
            if constexpr (s0 <= 26)
            {
                l0 = bits(w0, s0, 6);
            }
            else
            {
                l0 = (w0 >> static_cast<unsigned>(s0)) | (bits(r1, 0, s0 - 26) << static_cast<unsigned>(32 - s0));
            }
            if constexpr (s1 <= 26)
            {
                l1 = bits(w1, s1, 6);
            }
            else
            {
                l1 = (w1 >> static_cast<unsigned>(s1)) | (bits(r2, 0, s1 - 26) << static_cast<unsigned>(32 - s1));
            }
            f0              = ufloat(l0);
            f1              = ufloat(l1);
            f0              = f0 + -32.0f;
            f1              = f1 + -32.0f;
            const vFloat d0 = pf32<FMT>(0, 0, a);
            const vFloat d1 = pf32<FMT>(0, 1, a);
            f0              = d0 * f0;
            f1              = d1 * f1;
        }
        sfpi::dst_reg[iq4_ps<FMT>() + pidx(a, 0, 0)] = f0;
        sfpi::dst_reg[iq4_ps<FMT>() + pidx(a, 1, 0)] = f1;
    }
}

// Phase 2 through SFPLOADMACRO, one output every six instructions. LRegs: x in L0, the Horner sum in L1, the outputs
// of even and odd stream positions in L2 and L3, S and M = -14 S of the current column group in L4, L5, c0 in L6 and
// -14 in L7; c3, c2, c1 in L12..L14 (iq4_init). Per output, plain: load x, L1 = c3 x + c2, L1 = L1 x + c1,
// L2|3 = L1 x + c0, and its store; macro 1 takes the output's L2|3 and zeroes it, then rounds it to an integer
// (saturating), casts it to Float32 and forms S w + M there, 3, 4 and 6 instructions on. The stores are plain
// SFPSTOREs: arbitrated against the unpacker and the FPU, so nothing has to wait for the next page's copy or the
// matmul.
constexpr int IQ4B_RND = 2, IQ4B_CAST = 3, IQ4B_MAD = 5; // delays, in instructions

// c3, c2, c1 in L12..L14; with SFPLOADMACRO, templates 0 = round to a saturated integer, 1 = sign-magnitude integer
// to Float32, 2 = S w + M with S, M in L4, L5, whose delays count instructions (the stores are plain).
inline void iq4_init()
{
    TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, IQ4B_C3 & 0xFFFF);
    TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, IQ4B_C3 >> 16);
    TTI_SFPCONFIG(0, 12, 0);
    TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, IQ4B_C2 & 0xFFFF);
    TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, IQ4B_C2 >> 16);
    TTI_SFPCONFIG(0, 13, 0);
    TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, IQ4B_C1 & 0xFFFF);
    TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, IQ4B_C1 >> 16);
    TTI_SFPCONFIG(0, 14, 0);
    TTI_SFPNOP;
    if constexpr (!LOADMACRO)
    {
        return;
    }
    TTI_SFP_STOCH_RND(sfpi::SFPSTOCHRND_RND_EVEN, 0, 9, 0, 12, sfpi::SFPSTOCHRND_MOD1_FP32_TO_INT8);
    TTI_SFPCAST(0, 13, sfpi::SFPCAST_MOD1_SM32_TO_FP32_RNE);
    TTI_SFPMAD(4, 0, 5, 14, sfpi::SFPMAD_MOD1_OFFSET_NONE);
    constexpr std::uint32_t cast  = (IQ4B_CAST << 3) | 5;
    constexpr std::uint32_t round = (IQ4B_RND << 3) | 4;
    constexpr std::uint32_t mad   = 0x80 | (IQ4B_MAD << 3) | 6;
    TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, cast | (mad << 8));
    TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, round);
    TTI_SFPCONFIG(0, 5, 0);
    TTI_SFPCONFIG(0xF00, 8, sfpi::SFPCONFIG_MOD1_ARG_IMM);
    TTI_SFPNOP;
}

enum : int
{
    QB_NOP,
    QB_LDX,  // x of output row y
    QB_MAD1, // L1 = c3 x + c2
    QB_MAD2, // L1 = L1 x + c1
    QB_MAD3, // L(2 + x) = L1 x + c0
    QB_MAC,  // macro 1 on L(2 + y), row z
    QB_ST,   // store L(2 + x) to row y
    QB_LDS,  // S of group x into L4
    QB_LDM,  // M of group x into L5: L4 * -14
};

// Stream position k = 8 g + j: column group g (a, b) = (g / 2, g % 2), row o = 16 (j / 4) + 8 a + 2 (j % 4) + b.
// Output k's instructions at 2 + 6 k + {0: load x, 1: MAD1, 3: MAD2, 4: macro, 5: MAD3, 14: store (the last two: 12,
// where the next positions' instructions would be)}; group 0 loads
// its S and forms its M first, group g = 1..3 after the load of x of position 8 g + 1: after the last macro operation
// of group g - 1 that reads them and before the first of group g, where no macro operation in flight reads a value of
// a later plain instruction.
constexpr int QB_BASE  = 2 + 6 * 31 + 13;
constexpr int QB_SLOTS = QB_BASE + 6;

struct qbplan
{
    mslot s[QB_SLOTS];
    int n;
};

constexpr int qb_row(const int k)
{
    const int g = k / 8, j = k % 8;
    return 16 * (j / 4) + 8 * (g / 2) + 2 * (j % 4) + g % 2;
}

constexpr qbplan make_qbplan()
{
    qbplan p {};
    mslot base[QB_BASE] {};
    base[0] = {QB_LDS, 0, 0, 0};
    base[1] = {QB_LDM, 0, 0, 0};
    for (int k = 0; k < 32; k++)
    {
        const int s                  = 2 + 6 * k;
        base[s]                      = {QB_LDX, 0, qb_row(k), 0};
        base[s + 1]                  = {QB_MAD1, 0, 0, 0};
        base[s + 3]                  = {QB_MAD2, 0, 0, 0};
        base[s + 4]                  = {QB_MAC, k / 8, k % 2, qb_row(k)};
        base[s + 5]                  = {QB_MAD3, k % 2, 0, 0};
        base[s + (k < 30 ? 14 : 12)] = {QB_ST, k % 2, qb_row(k), 0};
    }
    for (int q = 0; q < QB_BASE; q++)
    {
        p.s[p.n++] = base[q];
        if (q == 2 + 6 * 9 || q == 2 + 6 * 17 || q == 2 + 6 * 25)
        {
            const int g = ((q - 2) / 6 - 1) / 8;
            p.s[p.n++]  = {QB_LDS, g, 0, 0};
            p.s[p.n++]  = {QB_LDM, g, 0, 0};
        }
    }
    return p;
}

constexpr qbplan qbplan_iq4 = make_qbplan();

template <int FMT, int I>
sfpi_inline void qbslot_issue()
{
    constexpr mslot s = qbplan_iq4.s[I];
    if constexpr (s.kind == QB_LDX)
    {
        TTI_SFPLOAD(0, sfpi::SFPLOAD_MOD0_FMT_FP16A, ADDR_MOD_7, 2 * s.y);
    }
    else if constexpr (s.kind == QB_MAD1)
    {
        TTI_SFPMAD(12, 0, 13, 1, sfpi::SFPMAD_MOD1_OFFSET_NONE);
    }
    else if constexpr (s.kind == QB_MAD2)
    {
        TTI_SFPMAD(1, 0, 14, 1, sfpi::SFPMAD_MOD1_OFFSET_NONE);
    }
    else if constexpr (s.kind == QB_MAD3)
    {
        TTI_SFPMAD(1, 0, 6, 2 + s.x, sfpi::SFPMAD_MOD1_OFFSET_NONE);
    }
    else if constexpr (s.kind == QB_MAC)
    {
        TTI_SFPLOADMACRO((1 << 2) | (2 + s.y), LOAD_NONE, ADDR_MOD_7, 2 * s.z);
    }
    else if constexpr (s.kind == QB_ST)
    {
        TTI_SFPSTORE(2 + s.x, sfpi::SFPSTORE_MOD0_FMT_FP32, ADDR_MOD_7, 2 * s.y);
    }
    else if constexpr (s.kind == QB_LDS)
    {
        TTI_SFPLOAD(4, sfpi::SFPLOAD_MOD0_FMT_FP32, ADDR_MOD_7, 2 * (iq4_ps<FMT>() + 2 * s.x));
    }
    else if constexpr (s.kind == QB_LDM)
    {
        TTI_SFPMAD(4, 7, 9, 5, sfpi::SFPMAD_MOD1_OFFSET_NONE);
    }
    else
    {
        TTI_SFPNOP;
    }
}

template <int FMT, int... I>
sfpi_inline void qbslots_issue(std::integer_sequence<int, I...>)
{
    (qbslot_issue<FMT, I>(), ...);
}

template <int FMT>
sfpi_inline void body_iq4()
{
    TTI_SFPLOADI(6, sfpi::SFPLOADI_MOD0_LOWER, IQ4B_C0 & 0xFFFF);
    TTI_SFPLOADI(6, sfpi::SFPLOADI_MOD0_UPPER, IQ4B_C0 >> 16);
    TTI_SFPLOADI(7, sfpi::SFPLOADI_MOD0_FLOATB, IQ4B_NEG14);
    if constexpr (LOADMACRO)
    {
        qbslots_issue<FMT>(std::make_integer_sequence<int, qbplan_iq4.n> {});
    }
    else
    {
#pragma GCC unroll 4
        for (int g = 0; g < 4; g++)
        {
            TTI_SFPLOAD(4, sfpi::SFPLOAD_MOD0_FMT_FP32, ADDR_MOD_7, 2 * (iq4_ps<FMT>() + 2 * g));
            TTI_SFPMAD(4, 7, 9, 5, sfpi::SFPMAD_MOD1_OFFSET_NONE);
#pragma GCC unroll 8
            for (int j = 0; j < 8; j++)
            {
                const int o = qb_row(8 * g + j);
                TTI_SFPLOAD(0, sfpi::SFPLOAD_MOD0_FMT_FP16A, ADDR_MOD_7, 2 * o);
                TTI_SFPMAD(12, 0, 13, 1, sfpi::SFPMAD_MOD1_OFFSET_NONE);
                TTI_SFPMAD(1, 0, 14, 1, sfpi::SFPMAD_MOD1_OFFSET_NONE);
                TTI_SFPMAD(1, 0, 6, 2, sfpi::SFPMAD_MOD1_OFFSET_NONE);
                TTI_SFP_STOCH_RND(sfpi::SFPSTOCHRND_RND_EVEN, 0, 9, 2, 2, sfpi::SFPSTOCHRND_MOD1_FP32_TO_INT8);
                TTI_SFPCAST(2, 2, sfpi::SFPCAST_MOD1_SM32_TO_FP32_RNE);
                TTI_SFPMAD(4, 2, 5, 2, sfpi::SFPMAD_MOD1_OFFSET_NONE);
                TTI_SFPSTORE(2, sfpi::SFPSTORE_MOD0_FMT_FP32, ADDR_MOD_7, 2 * o);
            }
        }
    }
}

// One copy shared by IQ4_XS's eight tile indices.
inline __attribute__((noinline)) void body_iq4_xs_shared()
{
    body_iq4<IQ4_XS>();
}

} // namespace gguf
} // namespace ckernel::sfpu
