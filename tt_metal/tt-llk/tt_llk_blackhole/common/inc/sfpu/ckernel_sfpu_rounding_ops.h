// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-FileCopyrightText: © 2025 Jason Davies <jason@jasondavies.com>
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <climits>
#include <cstdint>

#include "ckernel.h"
#include "ckernel_addrmod.h"
#include "ckernel_defs.h"
#include "ckernel_instr_params.h"
#include "ckernel_ops.h"
#include "ckernel_sfpu_isinf_isnan.h"
#include "lltt.h"
#include "sfpi.h"

namespace ckernel
{
namespace sfpu
{

// #41922: CONSTRUCTION ZONE
// Implemented:Standard calling API, old bodies
// ToDo: use sfpi API

// compute truncate to zero
sfpi_inline sfpi::vFloat _trunc_body_(sfpi::vFloat val)
{
    sfpi::l_reg[sfpi::LRegs::LReg0] = val;
    // set L3=23.  TODO: this could be stored in a constant register, but use by rdiv prevents this for now.
    TTI_SFPLOADI(p_sfpu::LREG3, sfpi::SFPLOADI_MOD0_SHORT, 23);
    // mask = 0x8000_0000
    TTI_SFPLOADI(p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_FLOATB, 0x8000);
    // disable lanes where exp < 0
    TTI_SFPEXEXP(0, p_sfpu::LREG0, p_sfpu::LREG2, sfpi::SFPEXEXP_MOD1_SET_CC_SGN_EXP | sfpi::SFPEXEXP_MOD1_SET_CC_COMP_EXP);
    // mask = 0xffff_ffff
    TTI_SFPLOADI(p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_SHORT, 0xffff);
    // exp = 23 - exp
    TTI_SFPIADD(0, p_sfpu::LREG3, p_sfpu::LREG2, sfpi::SFPIADD_MOD1_ARG_2SCOMP_LREG_DST | sfpi::SFPIADD_MOD1_CC_GTE0);
    // mask <<= exp
    TTI_SFPSHFT2(p_sfpu::LREG1, p_sfpu::LREG2, p_sfpu::LREG1, sfpi::SFPSHFT2_MOD1_SHFT_LREG);
    // reset lanes
    TTI_SFPENCC(0, 0, 0, 0);
    // apply mask
    TTI_SFPAND(0, p_sfpu::LREG0, p_sfpu::LREG1, 0);

    sfpi::l_reg[sfpi::LRegs::LReg2].in_use();
    sfpi::l_reg[sfpi::LRegs::LReg3].in_use();

    return sfpi::l_reg[sfpi::LRegs::LReg1];
}

// compute floor
sfpi_inline sfpi::vFloat _floor_body_(sfpi::vFloat val)
{
    sfpi::l_reg[sfpi::LRegs::LReg1] = _trunc_body_(val);
    // if v>u, set v=v-1.
    TTI_SFPGT(0, p_sfpu::LREG0, p_sfpu::LREG1, 1); // SFPGT_MOD1_SET_CC
    TTI_SFPMAD(p_sfpu::LCONST_1, p_sfpu::LREG1, p_sfpu::LCONST_neg1, p_sfpu::LREG1, 0);
    TTI_SFPENCC(0, 0, 0, 0);

    return sfpi::l_reg[sfpi::LRegs::LReg1];
}

// computes ceil
sfpi_inline sfpi::vFloat _ceil_body_(sfpi::vFloat val)
{
    sfpi::l_reg[sfpi::LRegs::LReg1] = _trunc_body_(val);
    // if v<u, set v=v+1.
    TTI_SFPGT(0, p_sfpu::LREG1, p_sfpu::LREG0, 1); // SFPGT_MOD1_SET_CC
    TTI_SFPMAD(p_sfpu::LCONST_1, p_sfpu::LREG1, p_sfpu::LCONST_1, p_sfpu::LREG1, 0);
    TTI_SFPENCC(0, 0, 0, 0);

    return sfpi::l_reg[sfpi::LRegs::LReg1];
}

inline constexpr std::array<float, 84> PRECOMPUTED_POW10_TABLE = {
    1e-45F, 1e-44F, 1e-43F, 1e-42F, 1e-41F, 1e-40F, 1e-39F, 1e-38F, 1e-37F, 1e-36F, 1e-35F, 1e-34F, 1e-33F, 1e-32F, 1e-31F, 1e-30F, 1e-29F,
    1e-28F, 1e-27F, 1e-26F, 1e-25F, 1e-24F, 1e-23F, 1e-22F, 1e-21F, 1e-20F, 1e-19F, 1e-18F, 1e-17F, 1e-16F, 1e-15F, 1e-14F, 1e-13F, 1e-12F,
    1e-11F, 1e-10F, 1e-9F,  1e-8F,  1e-7F,  1e-6F,  1e-5F,  1e-4F,  1e-3F,  1e-2F,  1e-1F,  1e0F,   1e1F,   1e2F,   1e3F,   1e4F,   1e5F,
    1e6F,   1e7F,   1e8F,   1e9F,   1e10F,  1e11F,  1e12F,  1e13F,  1e14F,  1e15F,  1e16F,  1e17F,  1e18F,  1e19F,  1e20F,  1e21F,  1e22F,
    1e23F,  1e24F,  1e25F,  1e26F,  1e27F,  1e28F,  1e29F,  1e30F,  1e31F,  1e32F,  1e33F,  1e34F,  1e35F,  1e36F,  1e37F,  1e38F,
};

// Fast bf16 floor for Blackhole. Origin: llk-bench (LLM-agent-written kernel, claude-fable-5, 2026-08),
// validated exhaustively over all 65,536 bf16 inputs vs the exact golden: max 0 ULP (gate <= 0).
// Measured 238.1 cycles/tile vs 483.1 for the previous kernel on p150b (tt-metal v0.76.0 baseline).
// bf16-DEST only (LO16 loads, bf16 bit tricks); never used for fp32 dest. Programs: LREG12 (1.5*2^23) and LREG0
// (-1.0f, must survive across faces: the kernel only writes L1/L2/L7), instruction templates 0..3, macro sequence
// slots 4..6, LoadMacroConfig.Misc (slot 8); no replay slots, no ADDR_MOD_6 (ADDR_MOD_7 only).
//
// Algorithm (per 32-lane vector, bf16 values as fp32 in LREGs):
//   T  = x + 1.5*2^23        (RNE: rounds x to nearest integer in-range;
//                             == x + M exactly for all bf16 with exp >= 7)
//   t  = T - 1.5*2^23
//   t  = rnd16b(t)           (round-to-bf16-precision on the Round unit:
//                             identity on every correct lane — floor of bf16
//                             is bf16-exact — and repairs the lone x=+2^48
//                             lane where the magic round-trip lands one fp32
//                             ulp low)
//   ts = copysign(t, x)      (fixes x = -0, where t = +0 and the total-order
//                             compare below would falsely decrement)
//   L7 = -(x >= ts)          (SFPLE nodec mask; raw-bit total order handles
//                             denormal x exactly)
//   ts-1 -> LReg[L7 & 15]    (SFPMAD with INDIRECT_VD: nodec lanes hit index
//                             15, which is not writable, so the write is
//                             dropped; dec lanes hit index 0 = the ts
//                             register. Completely CC-free floor correction.)
//
// Issue schedule: 6 issued instructions per vector, everything else rides
// SFPLOADMACRO slots (WaitForElapsedInstructions, so positions below are
// issued-instruction counts). Steady-state block for vector v (B = block
// start):
//   B+0  LOADMACRO m0 -> L1     | x->L1@1; MAD1@1 (T@3); RND@5 (t2@6)
//   B+1  SFPNOP
//   B+2  MAD  (INDIRECT_VA)     | tail of v-1: FINAL(v-1) in L2 @4
//   B+3  MAD2: L1 = T - M (t@5)
//   B+4  LOADMACRO m2 -> L7     | x->L7@5; LE@7 (mask@8)
//   B+5  LOADMACRO m1 -> L2     | x->L2@6; SETSGN@6 (ts@7); STORE@10
// (m2 uses offset OFF+1: load target L7 forces the address LSB to 1, and the
// Dst addressing ignores that bit, so the same 32 datums are fetched.)
// The store at B+10 reads L2 after the tail MAD of this vector (issued at
// B+8 = slot 2 of the next block) lands its result at B+10.
inline void _init_floor_bf16_fast_()
{
    // vConstFloatPrgm0 -> LReg12 = 1.5 * 2^23
    sfpi::vConstFloatPrgm0 = 12582912.0f;
    // All lanes enabled; no CC is used anywhere in the kernel.
    TTI_SFPENCC(0x3, 0, 0, 0xA);

    constexpr std::uint32_t L10 = p_sfpu::LCONST_1; // 1.0f
    constexpr std::uint32_t L12 = 12;               // 1.5*2^23

    // InstructionTemplate[0] (backdoor via VD=12): MAD1, T = 1.0*VB + M.
    TTI_SFPMAD(L10, 0, L12, 12, 0);
    // InstructionTemplate[1] (backdoor via VD=13): STOCHRND fp32->bf16
    // precision, round-nearest; VC/VD overridden to the load target.
    TTI_SFP_STOCH_RND(0, 0, 0, 0, 13, 1);
    // InstructionTemplate[2] (backdoor via VD=14): SETSGN, magnitude from L1
    // (t2), sign from VB = load target (x), result into the load target.
    TTI_SFPSETSGN(0, 1, 14, 0);
    // InstructionTemplate[3]: SFPLE with SET_VD; VB comes from the template
    // VD field (must be L2 = ts), VC is overridden to the load target (x),
    // VD is overridden to the load target (mask -> L7). VD field 0 < 12
    // would execute rather than backdoor-load, so install via SFPCONFIG.
    TTI_SFPLOADI(0, 0xA, TT_OP_SFPLE(0, 0, 2, 8) & 0xFFFF);
    TTI_SFPLOADI(0, 0x8, (TT_OP_SFPLE(0, 0, 2, 8) >> 16) & 0xFFFF);
    TTI_SFPCONFIG(0, 3, 0);

    // Macro Sequence Register 0 (dest 4): MAD slot = template 0 (mux 4),
    // delay 0, VB = loaded value (0x84); Round slot = template 1 (mux 5),
    // delay 4, VC = load target (0x25).
    TTI_SFPLOADI(0, 0xA, 0x8400);
    TTI_SFPLOADI(0, 0x8, 0x0025);
    TTI_SFPCONFIG(0, 4, 0);

    // Macro Sequence Register 1 (dest 5): Simple slot = template 2 (mux 6),
    // delay 0, VB = load target (0x86); Store slot = builtin (mux 3),
    // delay 4, source = load target (0x23).
    TTI_SFPLOADI(0, 0xA, 0x0086);
    TTI_SFPLOADI(0, 0x8, 0x2300);
    TTI_SFPCONFIG(0, 5, 0);

    // Macro Sequence Register 2 (dest 6): Simple slot = template 3 (mux 7),
    // delay 2, VC = load target (0x17).
    TTI_SFPLOADI(0, 0xA, 0x0017);
    TTI_SFPLOADI(0, 0x8, 0x0000);
    TTI_SFPCONFIG(0, 6, 0);

    // LoadMacroConfig.Misc: StoreMod0 = 0 (SrcB default -> bf16),
    // UsesLoadMod0ForStore = 0, UnitDelayKind = 0xF (count instructions).
    TTI_SFPCONFIG(0xF00, 8, 1);
    // L0 = -1.0f: the INDIRECT_VA decrement operand (index 0). Programmed
    // last because the SFPCONFIG staging above goes through L0.
    TTI_SFPLOADI(0, 0xA, 0x0000);
    TTI_SFPLOADI(0, 0x8, 0xBF80);
    TTI_SFPNOP;
}

// The floor correction for the previous vector: out = LReg[L7 & 15]*1 + ts.
// Nodec lanes (L7 = -1) read LReg15 (denormal lane ids, flushed to +0 by the
// MAD); dec lanes (L7 = 0) read L0 = -1.0f. INDIRECT_VA's conservative
// hazard check looks at the preceding *issued* instruction (an SFPNOP), so
// this never stalls.
#define FLOOR_FAST_TAIL_PREV() TTI_SFPMAD(0, p_sfpu::LCONST_1, 2, 2, 4)

template <int OFF, bool FIRST>
inline void _floor_fast_vec_()
{
    constexpr std::uint32_t L1  = 1;
    constexpr std::uint32_t L10 = p_sfpu::LCONST_1; // 1.0f
    constexpr std::uint32_t L12 = 12;               // 1.5*2^23

    TTI_SFPLOADMACRO((0 << 2) | 1, 0, ADDR_MOD_7, OFF); // m0: x->L1
    TTI_SFPNOP;
    if constexpr (FIRST)
    {
        // No previous vector: park L7 at -1 (nodec) so nothing is written.
        TTI_SFPLOADI(7, 0x2, 0xFFFF);
    }
    else
    {
        FLOOR_FAST_TAIL_PREV();
    }
    TTI_SFPMAD(L1, L10, L12, L1, 2);                        // MAD2: t = T - M
    TTI_SFPLOADMACRO((2 << 2) | 3, 0, ADDR_MOD_7, OFF + 1); // m2: x->L7
    TTI_SFPLOADMACRO((1 << 2) | 2, 0, ADDR_MOD_7, OFF);     // m1: x->L2
}

// One face (8 dst vectors) in place, at dst offsets 0,2,...,14 relative to the current dst base.
inline void _calculate_floor_bf16_fast_()
{
    _floor_fast_vec_<0, true>();
    _floor_fast_vec_<2, false>();
    _floor_fast_vec_<4, false>();
    _floor_fast_vec_<6, false>();
    _floor_fast_vec_<8, false>();
    _floor_fast_vec_<10, false>();
    _floor_fast_vec_<12, false>();
    _floor_fast_vec_<14, false>();
    // Epilogue: tick SETSGN/LE of the last vector, apply its decrement, and
    // tick its scheduled store (fires 5 issues after its LOADMACRO m1).
    TTI_SFPNOP;
    TTI_SFPNOP;
    FLOOR_FAST_TAIL_PREV();
    TTI_SFPNOP;
}

#undef FLOOR_FAST_TAIL_PREV

template <bool is_fp32_dest_acc_en>
inline void _init_floor_()
{
    // The common prologue (SFPU config reg + ADDR_MOD_7 + counter reset) is run by the caller
    // (_llk_math_eltwise_unary_sfpu_init_<SfpuType::floor>() via the callback init overload).
#ifndef DISABLE_SFPLOADMACRO
    if constexpr (!is_fp32_dest_acc_en)
    {
        _init_floor_bf16_fast_();
    }
#endif
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, int ITERATIONS = 8>
sfpi_inline void _calculate_floor_()
{
#ifndef DISABLE_SFPLOADMACRO
    if constexpr (!is_fp32_dest_acc_en && ITERATIONS == 8)
    {
        _calculate_floor_bf16_fast_();
        return;
    }
#endif
    for (int d = 0; d < ITERATIONS; d++)
    {
        sfpi::dst_reg[0] = _floor_body_(sfpi::dst_reg[0]);
        sfpi::dst_reg++;
    }
}

// Fast bf16 ceil for Blackhole. Origin: llk-bench (LLM-agent-written kernel, claude-fable-5, 2026-08),
// validated exhaustively over all 65,536 bf16 inputs vs the exact golden: max 0 ULP (gate <= 0).
// Measured 198.1 cycles/tile vs 483.1 for the previous kernel on p150b (tt-metal v0.76.0 baseline).
// bf16-DEST only; never used for fp32 dest. Programs: LREG12 (1.5*2^23), LREG13 (-1.5*2^23), LREG14 (1.0f),
// instruction templates 0..3, macro sequence slots 4..6, LoadMacroConfig.Misc (slot 8); no replay slots, no
// ADDR_MOD_6 (ADDR_MOD_7 only). Scratch: L0, L2, L3, L5, L16 (staging).
//
// Math (exhaustively verified over all 65536 bf16 patterns against an exact
// model of the SFPU: fused single-rounding MAD with DAZ/FTZ, sign-magnitude
// SFPGT compare, SFP_STOCH_RND half-up on the 16 discarded bits):
//
//   t    = (x + C) - C          C = 1.5*2^23; round-to-nearest-int(x)
//   mask = (x > t) ? ~0 : 0     native SFPGT (sign-magnitude, sees denormals)
//   m    = mask & 1.0f
//   r    = m + t                = ceil(x) exactly, except x = +2^48 where the
//                                magic-add path yields 2^48 - 2^24
//   out  = rne_bf16(r)          SFP_STOCH_RND FP32->FP16B repairs the single
//                                broken lane (0x577FFFFF -> 2^48); every other
//                                r is exactly bf16, so it is a no-op. The
//                                truncating SFPSTORE is then exact.
//
// Schedule: 5 issue slots per row. Each row k (dst offset 2k) issues three
// SFPLOADMACROs plus two explicit ops; the macros run the rest of the
// pipeline on the otherwise-idle Simple/MAD/Round/Store sub-units
// (delays count elapsed cycles; fire = issue + delay + 1 at 1 issue/cycle):
//
//   5k+0  LMA  load x->L0        macro0: MAD d0: L0 = 1*L0 + C      @5k+1
//   5k+1  gt_{k-1} (explicit)    SFPGT: m = (m > L5) ? ~0 : 0
//   5k+2  LMB  load x->m_k       macro1: Simple d4: m &= 1.0f       @5k+7
//                                        MAD    d5: m = 1*m + L5    @5k+8
//   5k+3  LMC  load x->m_k       macro2: Round  d6: L16 = rnd(m)    @5k+10
//                                        Store  d7: dst[2k] <- L16  @5k+11
//   5k+4  mad2_k (explicit)      L5 = L0*1 + (-C)
//
// m_k alternates L2/L3 (live 5k+2 .. 5k+10). L5 is written at 5k+4 and last
// read by mad3 at 5k+8, before mad2_{k+1} rewrites it at 5k+9. Unit usage per
// phase (mod 5) is collision-free: Load {0,2,3}, MAD {1,4,3}, Simple {1,2},
// Round {0}, Store {1}. Because the delay counters run on cycles, the last
// row's pending macro ops (and_7 .. store_7) drain on wall-clock during the
// wrapper's between-faces work; scheduled stores capture their Dst address at
// issue time (verified: the exhaustive sweep passes across face increments).
inline void _init_ceil_bf16_fast_()
{
    sfpi::vConstFloatPrgm0 = 12582912.0f;  // L12 = C
    sfpi::vConstFloatPrgm1 = -12582912.0f; // L13 = -C
    sfpi::vConstFloatPrgm2 = 1.0f;         // L14 = 1.0f

    // Instruction templates via backdoor load (VD = 12+i writes template i).
    // T0: mad1  [ld] = LCONST_1*[ld] + C     (VB substituted by the load reg)
    TTI_SFPMAD(p_sfpu::LCONST_1, 0, p_sfpu::LREG12, 12, 0);
    // T1: and   [ld] = [ld] & 1.0f           (dest substituted)
    TTI_SFPAND(0, p_sfpu::LREG14, 13, 0);
    // T2: mad3  [ld] = LCONST_1*[ld] + L5    (VB substituted)
    TTI_SFPMAD(p_sfpu::LCONST_1, 0, p_sfpu::LREG5, 14, 0);
    // T3: rnd   L16 = rne_bf16([ld])         (srcc substituted, 0x40 -> L16)
    TTI_SFP_STOCH_RND(sfpi::SFPSTOCHRND_RND_EVEN, 0, 0, 0, 15, sfpi::SFPSTOCHRND_MOD1_FP32_TO_FP16B);

    // Sequence bytes: 0x80 = substitute the dest-ish field (else src-c field),
    // 0x40 = route via LReg16, bits 5:3 = delay, bits 2:0 = 4+template (3 = store).
    // Macro 0: MAD = T0, delay 0, subst VB, in-place.
    TTI_SFPCONFIG(0x8400, 4 + 0, 1);
    // Macro 1: Simple = T1 (d4, subst dest) = 0xA5; MAD = T2 (d5, subst VB) = 0xAE.
    TTI_SFPCONFIG(0xAEA5, 4 + 1, 1);
    // Macro 2: Round = T3 (d6, subst srcc, out L16) = 0x77;
    //          Store (d7, src L16) = 0x7B.  (round/store live in the high half.)
    TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, 0x0000);
    TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, 0x7B77);
    TTI_SFPCONFIG(0, 4 + 2, 0);
    // Misc: StoreMod0 = 0, UsesLoadMod0ForStore = macro2 (bit 6),
    // UnitDelayKind = WaitForElapsedCycles for all four sub-units: the last
    // row's pending macro ops keep draining on wall-clock during the
    // wrapper's between-faces work, so no trailing NOPs are needed.
    TTI_SFPCONFIG(0x040, 8, 1);
}

// One row: LMA / (gt of previous row) / LMB / LMC / mad2.
// M = m_k register (2 or 3), MP = m_{k-1}, OFF = 2k. GT_PREV emits the
// explicit SFPGT for the previous row (SFPNOP for the first row).
#define CEIL_FAST_ROW(OFF, M, GT_PREV)                      \
    TTI_SFPLOADMACRO((0 << 2) | 0, 0, ADDR_MOD_7, OFF);     \
    GT_PREV;                                                \
    TTI_SFPLOADMACRO((1 << 2) | (M), 0, ADDR_MOD_7, OFF);   \
    TTI_SFPLOADMACRO((2 << 2) | (M), 0, ADDR_MOD_7, OFF);   \
    TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LCONST_1, p_sfpu::LREG13, p_sfpu::LREG5, 0)

#define CEIL_FAST_GT(MP) TTI_SFPGT(0, p_sfpu::LREG5, (MP), 8)

// One face (8 dst vectors) in place, at dst offsets 0,2,...,14 relative to the current dst base.
inline void _calculate_ceil_bf16_fast_()
{
    CEIL_FAST_ROW(0, 2, TTI_SFPNOP);
    CEIL_FAST_ROW(2, 3, CEIL_FAST_GT(2));
    CEIL_FAST_ROW(4, 2, CEIL_FAST_GT(3));
    CEIL_FAST_ROW(6, 3, CEIL_FAST_GT(2));
    CEIL_FAST_ROW(8, 2, CEIL_FAST_GT(3));
    CEIL_FAST_ROW(10, 3, CEIL_FAST_GT(2));
    CEIL_FAST_ROW(12, 2, CEIL_FAST_GT(3));
    CEIL_FAST_ROW(14, 3, CEIL_FAST_GT(2));
    // Epilogue: one NOP so L5 (p2_7) is ready when gt_7 executes, then the
    // gt for the last row. The remaining macro ops of row 7 (and_7 ..
    // store_7) fire on elapsed cycles while the caller advances the face
    // address, so no drain NOPs are required.
    TTI_SFPNOP;
    CEIL_FAST_GT(3);
}

#undef CEIL_FAST_ROW
#undef CEIL_FAST_GT

template <bool is_fp32_dest_acc_en>
inline void _init_ceil_()
{
    // Common prologue (config reg + ADDR_MOD_7 + counter reset) is run by the callback init overload.
#ifndef DISABLE_SFPLOADMACRO
    if constexpr (!is_fp32_dest_acc_en)
    {
        _init_ceil_bf16_fast_();
    }
#endif
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, int ITERATIONS = 8>
sfpi_inline void _calculate_ceil_()
{
#ifndef DISABLE_SFPLOADMACRO
    if constexpr (!is_fp32_dest_acc_en && ITERATIONS == 8)
    {
        _calculate_ceil_bf16_fast_();
        return;
    }
#endif
    for (int d = 0; d < ITERATIONS; d++)
    {
        sfpi::dst_reg[0] = _ceil_body_(sfpi::dst_reg[0]);
        sfpi::dst_reg++;
    }
}

// Fast bf16 trunc for Blackhole. Origin: llk-bench (LLM-agent-written kernel, claude-fable-5, 2026-08),
// validated exhaustively over all 65,536 bf16 inputs vs the exact golden: max 0 ULP (gate <= 0).
// Measured 190.1 cycles/tile vs 387.1 for the previous kernel on p150b (tt-metal v0.76.0 baseline).
// bf16-DEST only; never used for fp32 dest. Programs: LREG12 (-2^30), instruction templates 0..2, macro sequence
// slot 4, LoadMacroConfig.Misc (slot 8); no replay slots, no ADDR_MOD_6 (ADDR_MOD_7 only). Scratch: L0, L16.
//
// Core op: Blackhole's SFPSTOCHRND supports round-toward-zero (rnd_mode=2)
// for fp32 -> sign-magnitude int16 — exactly trunc for |x| < 2^15 (saturates
// to +/-32767 above). Every bf16 with exponent >= 15 is already an integer,
// so those lanes are left untouched: the store is predicated on
// (x*x - 2^30) < 0  <=>  |x| < 2^15  (exact: bf16 squares are exact in fp32,
// and 2^30 - x^2 is exact near the boundary).
//
// Per 32-lane vector, one SFPLOADMACRO issues the load and schedules (delays
// measured in subsequently issued vector instructions):
//   +1  MAD   unit: m = L0*L0 + (-2^30)         -> L0   (result at +3)
//   +1  Round unit: s = stochrnd_zero_smag16(x) -> L0   (result at +2)
//   +2  Simple unit: y = cast_sm32_to_fp32(s)   -> L16  (result at +3)
//   +4  Store unit: DST[load addr] = L16  (lane-predicated, load's mod0)
// The two writes to L0 land on different cycles (round lat 1, MAD lat 2), so
// L0 holds x at +1, s at +2, m at +3. Issued stream per vector:
//   [LOADMACRO, NOP, NOP, SETCC(L0<0), ENCC]  = 5 issues/vector,
// with SETCC at +3 reading m and taking effect at +4 (the store tick), and
// ENCC at +4 re-opening all lanes from +5 (the next vector's load).
//
// vs. the previous kernel's 10 issues/vector.
inline void _init_trunc_bf16_fast_()
{
    // LREG12 (sfpi vConstFloatPrgm0 = CREG_IDX_PRGM1) = -2^30, the MAD
    // predicate offset. (LREG11 is sfpi's reserved -1 constant.)
    sfpi::vConstFloatPrgm0 = -1073741824.0f;

    // Instruction templates are captured by executing with lreg_dest = 12+i.
    // Template 0: stochrnd rnd=zero, fp32->smag16, descale=imm8=0.
    //             src_c is overridden to the load target at schedule time.
    TTI_SFP_STOCH_RND(2, 0, 0, 0, 12, sfpi::SFPSTOCHRND_MOD1_FP32_TO_SMAG16 | sfpi::SFPSTOCHRND_MOD1_IMM8);
    // Template 1: cast sm32->fp32 (RNE; exact for <=15-bit magnitudes).
    TTI_SFPCAST(0, 13, sfpi::SFPCAST_MOD1_SM32_TO_FP32_RNE);
    // Template 2: mad: dst = LREG0 * (VB:=load target) + LREG12 (-2^30).
    TTI_SFPMAD(p_sfpu::LREG0, 0, p_sfpu::LREG12, 14, 0);

    // Macro 0 sequence bytes: [store|round|mad|simple], each
    // {bit7: override VB (else VC) with load target, bit6: dest = LREG16,
    //  bits5:3 delay, bits2:0 selector (3 = store, 4+i = template i)}.
    {
        constexpr std::uint32_t simple_bits = 0x40 | (1 << 3) | 5; // CAST, +2, ->L16
        constexpr std::uint32_t mad_bits    = 0x80 | (0 << 3) | 6; // MAD, +1, VB:=VD
        constexpr std::uint32_t round_bits  = 0x00 | (0 << 3) | 4; // STOCHRND, +1
        constexpr std::uint32_t store_bits  = 0x40 | (3 << 3) | 3; // STORE L16, +4
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mad_bits << 8) | simple_bits);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store_bits << 8) | round_bits);
        TTI_SFPCONFIG(0, 4 + 0, 0);
    }

    // Misc: {UsesLoadMod0ForStore for macro 0, delays count issued instrs}.
    TTI_SFPCONFIG(0x710, 8, 1);
}

// One 32-lane vector at DST row offset `off`, in place.
#define TRUNC_FAST_VEC(off)                                              \
    TTI_SFPLOADMACRO(0, 0, ADDR_MOD_7, off);                             \
    TTI_SFPNOP;                                                          \
    TTI_SFPNOP;                                                          \
    TTI_SFPSETCC(0, p_sfpu::LREG0, 0, sfpi::SFPSETCC_MOD1_LREG_LT0);     \
    TTI_SFPENCC(3, 0, 0, sfpi::SFPENCC_MOD1_EI_RI);

// One face (8 dst vectors) in place, at dst offsets 0,2,...,14 relative to the current dst base.
inline void _calculate_trunc_bf16_fast_()
{
    TRUNC_FAST_VEC(0);
    TRUNC_FAST_VEC(2);
    TRUNC_FAST_VEC(4);
    TRUNC_FAST_VEC(6);
    TRUNC_FAST_VEC(8);
    TRUNC_FAST_VEC(10);
    TRUNC_FAST_VEC(12);
    TRUNC_FAST_VEC(14);
}

#undef TRUNC_FAST_VEC

template <bool is_fp32_dest_acc_en>
inline void _init_trunc_()
{
    // Common prologue (config reg + ADDR_MOD_7 + counter reset) is run by the callback init overload.
#ifndef DISABLE_SFPLOADMACRO
    if constexpr (!is_fp32_dest_acc_en)
    {
        _init_trunc_bf16_fast_();
    }
#endif
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, int ITERATIONS = 8>
sfpi_inline void _calculate_trunc_()
{
#ifndef DISABLE_SFPLOADMACRO
    if constexpr (!is_fp32_dest_acc_en && ITERATIONS == 8)
    {
        _calculate_trunc_bf16_fast_();
        return;
    }
#endif
    for (int d = 0; d < ITERATIONS; d++)
    {
        sfpi::dst_reg[0] = _trunc_body_(sfpi::dst_reg[0]);
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
sfpi_inline void _calculate_frac_()
{
    for (int d = 0; d < ITERATIONS; d++)
    {
        sfpi::vFloat x   = sfpi::dst_reg[0];
        sfpi::dst_reg[0] = x - _trunc_body_(x);
        sfpi::dst_reg++;
    }
}

sfpi_inline sfpi::vFloat _round_even_(sfpi::vFloat v)
{
    // Create a temporary copy tmp = abs(v).
    sfpi::vFloat tmp = sfpi::setsgn(v, 0);
    // For all 0 ≤ x < 2**23, x + 2**23 will shift out the fractional part with round-to-nearest-even.
    tmp += 0x1.p23f;
    // Hide SFPNOP; extract exponent.
    sfpi::vInt exp = sfpi::exexp(v);
    // Subtract 2**23 to restore exponent.
    tmp += -0x1.p23f;
    // Hide SFPNOP; check exponent.  If x ≥ 2**23, then there is no fractional part.
    v_if (exp < 23)
    {
        // v.{Exp,Man}=tmp.{Exp,Man}; retaining original sign.
        v = sfpi::copysgn(tmp, v);
    }
    v_endif;
    return v;
}

// Fast bf16 round (half-to-even, decimals == 0) for Blackhole. Origin: llk-bench (LLM-agent-written kernel,
// claude-fable-5, 2026-08), validated exhaustively over all 65,536 bf16 inputs vs the exact golden: max 0 ULP
// (gate <= 0). Measured 95.1 cycles/tile vs 434.1 for the previous kernel on p150b (tt-metal v0.76.0 baseline).
// bf16-DEST only; never used for fp32 dest. Programs: LREG12 (-1.5*2^23), instruction templates 1..2, macro
// sequence slots 4..5, LoadMacroConfig.Misc (slot 8), and math-thread replay slots 0..19 (the 20-instruction face
// body; each face is one REPLAY issue); no ADDR_MOD_6 (ADDR_MOD_7 only). Scratch: L0..L3.
//
// Algorithm (exact for all bf16 inputs):
//   y = x + C            (C = 1.5 * 2^23, fp32 RNE add -> integer round)
//   z = y - C
//   out = stochrnd_rne_fp32_to_bf16(z)   (repairs the lone +-2^48 case,
//                                         exact for every other result)
//
// Implementation uses Blackhole SFPLOADMACRO: one issued instruction per
// vector performs LD + (scheduled) MAD + STOCHRND + STORE on the parallel
// SFPU sub-units; the first add (y = x + C) is a plain SFPADDI issued right
// after each LOADMACRO. 8 vectors/face -> 8 LOADMACRO + 8 SFPADDI + 4 drain
// NOPs = 20 issued instructions per face, recorded once into the replay
// buffer so each face is a single REPLAY issue.
//
// Issue schedule (slot = issued vector-instruction index; UnitDelayKind is
// set to count-instructions so slots are exact):
//   vector v (lreg j = v%4):
//     LOADMACRO @s=2v: LD fires @s   -> L[j] = x
//     ADDI issued @s+1               -> L[j] += C   (y)     (MAD unit, odd)
//     MAD (delay 3) fires @s+4       -> L[j] = 1.0*L[j]-C   (MAD unit, even)
//     ROUND (delay 5) fires @s+6     -> staging = bf16_rne(L[j])
//     STORE (delay 6) fires @s+7     -> dst[LD addr] = staging
//   Plain ADDIs occupy odd slots, macro MADs fire on even slots, so the MAD
//   unit never collides. Tail NOPs tick the delay counters so every STORE
//   fires before the wrapper's SETRWC advances the dest counter.
inline void _round_fast_face_body_()
{
    TTI_SFPLOADMACRO(0, 0, ADDR_MOD_7, 0);  // v0 -> L0
    TTI_SFPADDI(0x4B40, 0, 0);              // y0 = x0 + C
    TTI_SFPLOADMACRO(1, 0, ADDR_MOD_7, 2);  // v1 -> L1
    TTI_SFPADDI(0x4B40, 1, 0);              // y1
    TTI_SFPLOADMACRO(2, 0, ADDR_MOD_7, 4);  // v2 -> L2
    TTI_SFPADDI(0x4B40, 2, 0);              // y2
    TTI_SFPLOADMACRO(3, 0, ADDR_MOD_7, 6);  // v3 -> L3
    TTI_SFPADDI(0x4B40, 3, 0);              // y3
    TTI_SFPLOADMACRO(0, 0, ADDR_MOD_7, 8);  // v4 -> L0
    TTI_SFPADDI(0x4B40, 0, 0);              // y4
    TTI_SFPLOADMACRO(1, 0, ADDR_MOD_7, 10); // v5 -> L1
    TTI_SFPADDI(0x4B40, 1, 0);              // y5
    TTI_SFPLOADMACRO(2, 0, ADDR_MOD_7, 12); // v6 -> L2
    TTI_SFPADDI(0x4B40, 2, 0);              // y6
    TTI_SFPLOADMACRO(7, 0, ADDR_MOD_7, 14); // v7 -> L3 (sequence 1: shorter delays)
    TTI_SFPADDI(0x4B40, 3, 0);              // y7
    TTI_SFPNOP;                             // drain: tick pending counters
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
}

inline void _init_round_bf16_fast_()
{
    // L12 = -C = -1.5 * 2^23 = -12582912.0f (0xCB400000)
    TTI_SFPLOADI(0, 0xA, 0x0000);
    TTI_SFPLOADI(0, 0x8, 0xCB40);
    TTI_SFPCONFIG(0, 12, 0);
    TTI_SFPNOP;

    // Macro instruction template mux[5] (backdoor install via lreg_dest=13):
    //   MAD: dest = 1.0 * VB + L12, VB remapped to the loadmacro lreg.
    TTI_SFPMAD(p_sfpu::LCONST_1, 0, 12, 13, 0);
    // Macro instruction template mux[6] (backdoor install via lreg_dest=14):
    //   STOCHRND: fp32 -> fp16b, non-stochastic nearest; input VC remapped
    //   to the loadmacro lreg.
    TTI_SFP_STOCH_RND(0, 0, 0, 0, 14, 1);

    // Macro sequence register 0:
    //   SIMPLE: skip                                     = 0x00
    //   MAD   : sel 5, delay 3, dest=lreg, loaded->VB    = 0x9D
    //   ROUND : sel 6, delay 5, dest=staging, loaded->VC = 0x6E
    //   STORE : sel 3, delay 6, src=staging              = 0x73
    TTI_SFPLOADI(0, 0xA, 0x9D00);
    TTI_SFPLOADI(0, 0x8, 0x736E);
    TTI_SFPCONFIG(0, 4, 0);
    TTI_SFPNOP;

    // Macro sequence register 1 (used by the last vector of each face):
    // same pipeline, one slot tighter so the face drains a slot earlier.
    //   MAD   : sel 5, delay 2, dest=lreg, loaded->VB    = 0x95
    //   ROUND : sel 6, delay 4, dest=staging, loaded->VC = 0x66
    //   STORE : sel 3, delay 5, src=staging              = 0x6B
    TTI_SFPLOADI(0, 0xA, 0x9500);
    TTI_SFPLOADI(0, 0x8, 0x6B66);
    TTI_SFPCONFIG(0, 5, 0);
    TTI_SFPNOP;

    // Misc: StoreMod0=0 (SRCB fmt), UsesLoadMod0ForStore=0, UnitDelayKind=0xF
    // (delays count issued vector instructions, not raw cycles).
    TTI_SFPCONFIG(0xF00, 8, 1);
    TTI_SFPNOP;

    // Record the 20-instruction face body into the replay buffer (NoExec);
    // each face is then a single REPLAY issue.
    load_replay_buf(0, 20, _round_fast_face_body_);
}

// One face (8 dst vectors) in place, at dst offsets 0,2,...,14 relative to the current dst base.
inline void _calculate_round_bf16_fast_()
{
    lltt::replay(0, 20);
}

template <bool is_fp32_dest_acc_en>
inline void _init_round_()
{
    // Common prologue (config reg + ADDR_MOD_7 + counter reset) is run by the callback init overload.
    // The fast state is programmed whenever bf16 dest is in use; _calculate_round_ only takes the fast path when
    // decimals == 0 at runtime, and the SFPI fallback does not depend on any of this state.
#ifndef DISABLE_SFPLOADMACRO
    if constexpr (!is_fp32_dest_acc_en)
    {
        _init_round_bf16_fast_();
    }
#endif
}

template <bool APPROXIMATE, bool is_fp32_dest_acc_en, int ITERATIONS = 8>
void _calculate_round_(const int decimals)
{
#ifndef DISABLE_SFPLOADMACRO
    if constexpr (!is_fp32_dest_acc_en && ITERATIONS == 8)
    {
        // The fast kernel implements round-half-even to an integer only (validated for decimals == 0).
        if (decimals == 0)
        {
            _calculate_round_bf16_fast_();
            return;
        }
    }
#endif
    const auto exp10i = [](int n)
    {
        if (n > 38) // 38 is max decimal places float32 can store for positive values
        {
            return 1.0F / 0.0F;
        }

        if (n < -45) // 45 is max decimal places float32 can store for negative values
        {
            return 0.0F;
        }

        return PRECOMPUTED_POW10_TABLE[n + 45];
    };

    const sfpi::vFloat coeff   = exp10i(decimals);
    const sfpi::vFloat inverse = exp10i(-decimals);

    for (int d = 0; d < ITERATIONS; ++d)
    {
        sfpi::vFloat v      = sfpi::dst_reg[0];
        sfpi::vFloat result = inverse * _round_even_(v * coeff);
        sfpi::dst_reg[0]    = result;
        sfpi::dst_reg++;
    }
}

// Performs stochastic rounding of values in DST from fp32 to fp16b format.
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
sfpi_inline void _calculate_stochastic_round_()
{
#pragma GCC unroll ITERATIONS
    for (int d = 0; d < ITERATIONS; d++)
    {
        sfpi::vFloat x   = sfpi::dst_reg[0];
        x                = sfpi::convert<sfpi::vFloat16b>(x, sfpi::RoundMode::NearestStochastic);
        sfpi::dst_reg[0] = x;
        sfpi::dst_reg++;
    }
}

} // namespace sfpu
} // namespace ckernel
