// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Round-2 SwiGluMul candidates (perf_experiments/swiglu_epilogue_loadmacro). Same math, same precision grade as
// the op's SwiGluMul (genify_matmul_swiglu_sfpu.hpp): out = silu(g) * u with
//   x = clamp(g, +-87); z = x*c1 + M; s = M - z; f = x*c1 + s; p = ((k3 f + k2) f + k1) f + 1;
//   E = p +int (z << 23); d = E + 1; r = arecip(d); e = 1 - d r; w = g u; q = w r; out = q e + q.
//
// Helper bypass (raw LLK): the op's SwiGluMul is raw SFPI already (no kernel_lib element computes silu(a)*b);
// these candidates go one level lower, to hand-scheduled TTI_SFP* words with explicit LRegs, because the
// SFPI compiler neither schedules across rows nor exposes SFPLOADMACRO. The eltwise_chain is NOT bypassed
// (it still owns init, DEST sync, CB sync, format reconfig and the pack).
//
// impl 1 "tuned": the whole tile (32 SFPU rows, 4 faces) is ONE software-pipelined instruction stream:
//   step(i) = tail(row i) interleaved with head(row i+1), hand-ordered so that no SFPMAD result is consumed
//   by the very next instruction (no auto-stall bubbles) and the two SFPSWAPs are the only 2-cycle issues.
//   Two LReg banks (A = L1..L3, B = L4..L6) alternate by row parity; z (L7) is shared (its row-i lifetime
//   ends before row i+1 defines it); k3 lives in L0. Per row: 22 issues + 2 SWAP cycles = 24 cycles
//   (baseline: same 22 issues + 2 SWAP cycles + ~4 MAD-dependency / face-drain bubbles).
//   K2C: k2 lives in the programmable constant LReg11 (LCONST_neg1), set per tile and restored to -1.0 at the
//   end of the tile -> the per-row k2 SFPLOADI disappears (21 issues + 2).
//   Bit-identical to the baseline element (the same SFPMAD / SFPIADD / SFPARECIP sequence per row).

#pragma once

#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"

#ifdef TRISC_MATH
#include "sfpi.h"
#include "lltt.h"
#include "ckernel_sfpu_sigmoid.h"
#include "llk_math_eltwise_binary_sfpu_macros.h"
#include "llk_math_eltwise_unary_sfpu_params.h"

namespace ckernel::sfpu {
namespace swlm {

constexpr uint32_t LZ = 7, LK3 = 0;
constexpr uint32_t C0 = 9, C1 = 10, CK1 = 12, CC1 = 13, CM = 14, CK2 = 11;  // const LRegs: 0.0, 1.0, k1, -log2e, M, k2
constexpr uint32_t BF16_NEG87 = 0xC2AE, BF16_POS87 = 0x42AE, FP16_K2 = 0x33C1, FP16_K3 = 0x2B15, BF16_NEG1 = 0xBF80;
constexpr uint32_t LD = 0;     // InstrModLoadStore::DEFAULT (resolves to FP32 on 32-bit DEST, BF16 otherwise)
constexpr uint32_t AM = 7;     // ADDR_MOD_7: no auto increment
constexpr uint32_t NEG_A = 1;  // SFPMAD_MOD1_NEGATE_VA
constexpr uint32_t IADD_CC_NONE = 4;
constexpr uint32_t SHFT_IMM_VC = 5;    // SFPSHFT_MOD1_ARG_IMM | ARG_IMM_USE_VC: VD = VC << imm
constexpr uint32_t SWAP_MINMAX = 1;    // VD = min, VC = max
constexpr bool D16 = !DST_ACCUM_MODE;  // 16-bit DEST: round the result to bf16 (RNE) before the store
template <uint32_t REG>
ALWI void round_bf16() {
    if constexpr (D16) {
        TTI_SFP_STOCH_RND(0, 0, 9, REG, REG, 1);  // SFPSTOCHRND fp32 -> bf16, round to nearest even
    }
}

template <int B>
struct Bank {
    static constexpr uint32_t g = B ? 4 : 1, u = B ? 5 : 2, s = B ? 6 : 3;
};
// whole-tile SFPU row R (0..31) -> SFPLOAD/STORE address (face = R / 8, 16 DEST rows per face)
template <int R>
constexpr uint32_t ra() {
    return 16 * (R / 8) + 2 * (R % 8);
}

// ---- standalone head / tail (pipeline fill and drain; hardware auto-stalls cover the MAD hazards) ----
template <int R, int UO>
ALWI void head() {
    using X = Bank<R & 1>;
    TTI_SFPLOAD(X::g, LD, AM, ra<R>());
    TTI_SFPLOAD(X::u, LD, AM, ra<R>() + UO);
    TTI_SFPMAD(X::g, X::u, C0, X::u, 0);  // w = g u
    TTI_SFPLOADI(X::s, sfpi::SFPLOADI_MOD0_FLOATB, BF16_NEG87);
    TTI_SFPSWAP(0, X::g, X::s, SWAP_MINMAX);  // g = max(g, -87)
    TTI_SFPLOADI(X::s, sfpi::SFPLOADI_MOD0_FLOATB, BF16_POS87);
    TTI_SFPSWAP(0, X::s, X::g, SWAP_MINMAX);  // x = min(87, g)
    TTI_SFPMAD(X::g, CC1, CM, LZ, 0);         // z = x c1 + M
    TTI_SFPMAD(LZ, C1, CM, X::s, NEG_A);      // s = M - z
    TTI_SFPMAD(X::g, CC1, X::s, X::g, 0);     // f = x c1 + s
    TTI_SFPSHFT(23, LZ, LZ, SHFT_IMM_VC);     // z <<= 23
}

template <int R, int OO, bool K2C>
ALWI void tail() {
    using X = Bank<R & 1>;
    if constexpr (K2C) {
        TTI_SFPMAD(LK3, X::g, CK2, X::s, 0);
    } else {
        TTI_SFPLOADI(X::s, sfpi::SFPLOADI_MOD0_FLOATA, FP16_K2);
        TTI_SFPMAD(LK3, X::g, X::s, X::s, 0);  // p = k3 f + k2
    }
    TTI_SFPMAD(X::s, X::g, CK1, X::s, 0);     // p = p f + k1
    TTI_SFPMAD(X::s, X::g, C1, X::s, 0);      // p = p f + 1
    TTI_SFPNOP;                               // SFPIADD reads VD: not covered by the MAD auto-stall
    TTI_SFPIADD(0, LZ, X::s, IADD_CC_NONE);   // E = p +int (z << 23)
    TTI_SFPMAD(X::s, C1, C1, X::s, 0);        // d = E + 1
    TTI_SFPARECIP(0, X::s, X::g, 0);          // r = arecip(d)
    TTI_SFPMAD(X::s, X::g, C1, X::s, NEG_A);  // e = 1 - d r
    TTI_SFPMAD(X::u, X::g, C0, X::u, 0);      // q = w r
    TTI_SFPMAD(X::u, X::s, X::u, X::u, 0);    // out = q e + q
    round_bf16<X::u>();
    TTI_SFPSTORE(X::u, LD, AM, ra<R>() + OO);
}

// ---- steady state: tail(row R) [bank X] interleaved with head(row R+1) [bank Y] ----
template <int R, int UO, int OO, bool K2C>
ALWI void step() {
    using X = Bank<R & 1>;
    using Y = Bank<(R + 1) & 1>;
    if constexpr (!K2C) {
        TTI_SFPLOADI(X::s, sfpi::SFPLOADI_MOD0_FLOATA, FP16_K2);  // T1
    }
    TTI_SFPLOAD(Y::g, LD, AM, ra<R + 1>());  // H1
    if constexpr (K2C) {
        TTI_SFPMAD(LK3, X::g, CK2, X::s, 0);  // T2 p = k3 f + k2
    } else {
        TTI_SFPMAD(LK3, X::g, X::s, X::s, 0);  // T2
    }
    TTI_SFPLOAD(Y::u, LD, AM, ra<R + 1>() + UO);                 // H2
    TTI_SFPMAD(X::s, X::g, CK1, X::s, 0);                        // T3 p = p f + k1
    TTI_SFPMAD(Y::g, Y::u, C0, Y::u, 0);                         // H3 w = g u
    TTI_SFPMAD(X::s, X::g, C1, X::s, 0);                         // T4 p = p f + 1
    TTI_SFPLOADI(Y::s, sfpi::SFPLOADI_MOD0_FLOATB, BF16_NEG87);  // H4
    TTI_SFPSWAP(0, Y::g, Y::s, SWAP_MINMAX);                     // H5 g = max(g, -87)
    TTI_SFPIADD(0, LZ, X::s, IADD_CC_NONE);                      // T5 E = p +int z
    TTI_SFPLOADI(Y::s, sfpi::SFPLOADI_MOD0_FLOATB, BF16_POS87);  // H6
    TTI_SFPMAD(X::s, C1, C1, X::s, 0);                           // T6 d = E + 1
    TTI_SFPSWAP(0, Y::s, Y::g, SWAP_MINMAX);                     // H7 x = min(87, g)
    TTI_SFPARECIP(0, X::s, X::g, 0);                             // T7 r = arecip(d)
    TTI_SFPMAD(Y::g, CC1, CM, LZ, 0);                            // H8 z = x c1 + M
    TTI_SFPMAD(X::s, X::g, C1, X::s, NEG_A);                     // T8 e = 1 - d r
    TTI_SFPMAD(LZ, C1, CM, Y::s, NEG_A);                         // H9 s = M - z
    TTI_SFPMAD(X::u, X::g, C0, X::u, 0);                         // T9 q = w r
    TTI_SFPMAD(Y::g, CC1, Y::s, Y::g, 0);                        // H10 f = x c1 + s
    TTI_SFPMAD(X::u, X::s, X::u, X::u, 0);                       // T10 out = q e + q
    TTI_SFPSHFT(23, LZ, LZ, SHFT_IMM_VC);                        // H11 z <<= 23
    round_bf16<X::u>();                                          // (16-bit DEST only)
    TTI_SFPSTORE(X::u, LD, AM, ra<R>() + OO);                    // T11
}

template <int UO, int OO, bool K2C, int... I>
ALWI void steps(std::integer_sequence<int, I...>) {
    (step<I, UO, OO, K2C>(), ...);
}

// One whole tile: gate at the DEST base, up at +UO, out at +OO (SFPLOAD address units; one tile = 64).
template <int UO, int OO, bool K2C>
inline void swiglu_tuned_tile() {
    TTI_SFPLOADI(LK3, sfpi::SFPLOADI_MOD0_FLOATA, FP16_K3);
    if constexpr (K2C) {
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_FLOATA, FP16_K2);
        TTI_SFPCONFIG(0, CK2, 0);
        TTI_SFPNOP;
        TTI_SFPLOADI(LK3, sfpi::SFPLOADI_MOD0_FLOATA, FP16_K3);
    }
    head<0, UO>();
    steps<UO, OO, K2C>(std::make_integer_sequence<int, 31>{});
    tail<31, OO, K2C>();
    if constexpr (K2C) {
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_FLOATB, BF16_NEG1);  // restore LCONST_neg1
        TTI_SFPCONFIG(0, CK2, 0);
        TTI_SFPNOP;
    }
}

// =====================================================================================================
// impl 0 "lm": SFPLOADMACRO pipeline. Same front (clamp, z, s, f, polynomial, exponent add, d, r, e) as
// "tuned"; the back end is reformulated so that two loads double as macros that each carry extra work:
//   q = g r      <- MG = SFPLOADMACRO(macro 0) re-loads g into the row's g LReg and schedules the SFPMAD
//                   q = Lr * g (template 0) on the MAD sub-unit, concurrent with the next regular issue;
//   V = q e + q  (regular SFPMAD into Lv);
//   out = V u    <- MU = SFPLOADMACRO(macro 1) loads u into Lr (free by then) and schedules the SFPMAD
//                   L16 = Lv * u (template 1) and the SFPSTORE of L16 to u's own DEST address (store sub-unit),
//                   both concurrent with the NEXT step's first regular issues.
// The w = g u multiply, the q multiply and the SFPSTORE leave the issue stream: 19 issues + 2 SWAP cycles per
// row (tuned_k2c: 21 + 2). Output lands in the UP slot (D1); the chain packs D1. Precision: the same 3
// roundings in the product chain ((g r)(1 + e)) u instead of ((g u) r)(1 + e).
// Registers: bank A g=L1 s=L2, bank B g=L3 s=L4; z=L5; r=L6 (then u); V=L7; k3=L0; k2=LReg11 (per tile).
// Timing (instruction-count delays, Misc.UnitDelayKind = all): every scheduled op lands on a cycle whose
// regular issue is NOT on the same sub-unit (a scheduled op silently discards a same-unit regular issue), and
// every consumer of a scheduled SFPMAD result is >= 2 cycles later (no auto-stall for macro ops).
namespace lmk {
constexpr uint32_t LZ = 5, LR = 6, LV = 7, LK3 = 0;
template <int B>
struct Bank {
    static constexpr uint32_t g = B ? 3 : 1, s = B ? 4 : 2;
};
// SFPLOADMACRO: lreg_ind = macro << 2 | VD[1:0]; address low bit = VD[2] (base addresses are even).
template <uint32_t MACRO, uint32_t VD, uint32_t ADDR>
ALWI void loadmacro() {
    TTI_SFPLOADMACRO((MACRO << 2) | (VD & 3), LD, AM, ADDR | (VD >> 2));
}

template <int R, int UO>
ALWI void head() {
    using X = Bank<R & 1>;
    TTI_SFPLOAD(X::g, LD, AM, ra<R>());
    TTI_SFPLOADI(X::s, sfpi::SFPLOADI_MOD0_FLOATB, BF16_NEG87);
    TTI_SFPSWAP(0, X::g, X::s, SWAP_MINMAX);
    TTI_SFPLOADI(X::s, sfpi::SFPLOADI_MOD0_FLOATB, BF16_POS87);
    TTI_SFPSWAP(0, X::s, X::g, SWAP_MINMAX);
    TTI_SFPMAD(X::g, CC1, CM, LZ, 0);
    TTI_SFPMAD(LZ, C1, CM, X::s, NEG_A);
    TTI_SFPMAD(X::g, CC1, X::s, X::g, 0);
    TTI_SFPSHFT(23, LZ, LZ, SHFT_IMM_VC);
}

template <int R, int UO>
ALWI void tail() {
    using X = Bank<R & 1>;
    TTI_SFPNOP;  // the previous step's out MAD executes on this cycle: a regular SFPMAD here would be discarded
    TTI_SFPMAD(LK3, X::g, CK2, X::s, 0);
    TTI_SFPMAD(X::s, X::g, CK1, X::s, 0);
    TTI_SFPMAD(X::s, X::g, C1, X::s, 0);
    TTI_SFPNOP;
    TTI_SFPIADD(0, LZ, X::s, IADD_CC_NONE);
    TTI_SFPMAD(X::s, C1, C1, X::s, 0);
    TTI_SFPARECIP(0, X::s, LR, 0);
    TTI_SFPMAD(X::s, LR, C1, X::s, NEG_A);
    loadmacro<0, X::g, ra<R>()>();  // g -> X.g, q = Lr g at +1
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPMAD(X::g, X::s, X::g, LV, 0);  // V = q e + q
    loadmacro<1, LR, ra<R>() + UO>();     // u -> Lr, L16 = V u at +1, (round at +3,) store L16 at +3 (+4)
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
}

// tail(row R) [bank X] interleaved with head(row R+1) [bank Y]; cycle numbers in the comments (SWAP = 2).
template <int R, int UO>
ALWI void step() {
    using X = Bank<R & 1>;
    using Y = Bank<(R + 1) & 1>;
    TTI_SFPLOAD(Y::g, LD, AM, ra<R + 1>());                      // c0  H1   [prev out MAD]
    TTI_SFPMAD(LK3, X::g, CK2, X::s, 0);                         // c1  T2   p = k3 f + k2
    TTI_SFPLOADI(Y::s, sfpi::SFPLOADI_MOD0_FLOATB, BF16_NEG87);  // c2  H4   [prev store L16]
    TTI_SFPMAD(X::s, X::g, CK1, X::s, 0);                        // c3  T3
    TTI_SFPSWAP(0, Y::g, Y::s, SWAP_MINMAX);                     // c4  H5   g = max(g, -87)
    TTI_SFPMAD(X::s, X::g, C1, X::s, 0);                         // c6  T4
    TTI_SFPLOADI(Y::s, sfpi::SFPLOADI_MOD0_FLOATB, BF16_POS87);  // c7  H6
    TTI_SFPIADD(0, LZ, X::s, IADD_CC_NONE);                      // c8  T5   E = p +int z
    TTI_SFPSWAP(0, Y::s, Y::g, SWAP_MINMAX);                     // c9  H7   x = min(87, g)
    TTI_SFPMAD(X::s, C1, C1, X::s, 0);                           // c11 T6   d = E + 1
    TTI_SFPMAD(Y::g, CC1, CM, LZ, 0);                            // c12 H8   z = x c1 + M
    TTI_SFPARECIP(0, X::s, LR, 0);                               // c13 T7   r = arecip(d)
    TTI_SFPMAD(LZ, C1, CM, Y::s, NEG_A);                         // c14 H9   s = M - z
    TTI_SFPMAD(X::s, LR, C1, X::s, NEG_A);                       // c15 T8   e = 1 - d r
    loadmacro<0, X::g, ra<R>()>();                               // c16 MG   g -> X.g; q = Lr g @c17
    TTI_SFPSHFT(23, LZ, LZ, SHFT_IMM_VC);                        // c17 H11  z <<= 23   [q MAD]
    TTI_SFPMAD(Y::g, CC1, Y::s, Y::g, 0);                        // c18 H10  f = x c1 + s
    TTI_SFPMAD(X::g, X::s, X::g, LV, 0);                         // c19 V    V = q e + q
    loadmacro<1, LR, ra<R>() + UO>();                            // c20 MU   u -> Lr; L16 = V u @c21; st @c23
}

// lm2: the g load of row R+1 is itself a macro (MG0) that schedules the first clamp SFPSWAP(VC = L7 = -87,
// VD = g, mode 9: VD = max) onto the simple sub-unit at a cycle whose regular issue is an SFPLOADI (MAD idle)
// and whose next regular issue is an SFPMAD (simple / round idle) -- the swap leaves the issue stream and its
// stall cycle disappears: 18 issues + 1 SWAP cycle = 19 cycles per row (lm: 21). L7 is free between the
// previous out MAD (c0) and V (c17), so it doubles as the clamp temp.
template <int R, int UO>
ALWI void step2() {
    using X = Bank<R & 1>;
    using Y = Bank<(R + 1) & 1>;
    loadmacro<2, Y::g, ra<R + 1>()>();                           // c0  MG0  g -> Y.g; swap(L7, Y.g) @c4 [prev out MAD]
    TTI_SFPMAD(LK3, X::g, CK2, X::s, 0);                         // c1  T2   p = k3 f + k2
    TTI_SFPLOADI(LV, sfpi::SFPLOADI_MOD0_FLOATB, BF16_NEG87);    // c2  H4   L7 = -87   [prev store / round]
    TTI_SFPMAD(X::s, X::g, CK1, X::s, 0);                        // c3  T3
    TTI_SFPLOADI(Y::s, sfpi::SFPLOADI_MOD0_FLOATB, BF16_POS87);  // c4  H6   [macro swap: g = max(g, -87)]
    TTI_SFPMAD(X::s, X::g, C1, X::s, 0);                         // c5  T4   (simple/round idle for the swap)
    TTI_SFPSWAP(0, Y::s, Y::g, SWAP_MINMAX);                     // c6  H7   x = min(87, g) (2 cycles)
    TTI_SFPIADD(0, LZ, X::s, IADD_CC_NONE);                      // c8  T5   E = p +int z
    TTI_SFPMAD(X::s, C1, C1, X::s, 0);                           // c9  T6   d = E + 1
    TTI_SFPMAD(Y::g, CC1, CM, LZ, 0);                            // c10 H8   z = x c1 + M
    TTI_SFPARECIP(0, X::s, LR, 0);                               // c11 T7   r = arecip(d)
    TTI_SFPMAD(LZ, C1, CM, Y::s, NEG_A);                         // c12 H9   s = M - z
    TTI_SFPMAD(X::s, LR, C1, X::s, NEG_A);                       // c13 T8   e = 1 - d r
    loadmacro<0, X::g, ra<R>()>();                               // c14 MG   g -> X.g; q = Lr g @c15
    TTI_SFPSHFT(23, LZ, LZ, SHFT_IMM_VC);                        // c15 H11  z <<= 23   [q MAD]
    TTI_SFPMAD(Y::g, CC1, Y::s, Y::g, 0);                        // c16 H10  f = x c1 + s
    TTI_SFPMAD(X::g, X::s, X::g, LV, 0);                         // c17 V    V = q e + q
    loadmacro<1, LR, ra<R>() + UO>();                            // c18 MU   u -> Lr; L16 = V u @c0'; st @c2'
}

template <int UO, bool SW, int... I>
ALWI void steps(std::integer_sequence<int, I...>) {
    if constexpr (SW) {
        (step2<I, UO>(), ...);
    } else {
        (step<I, UO>(), ...);
    }
}

template <int UO, bool SW>
inline void swiglu_lm_tile() {
    TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_FLOATA, FP16_K2);
    TTI_SFPCONFIG(0, CK2, 0);
    TTI_SFPNOP;
    TTI_SFPLOADI(LK3, sfpi::SFPLOADI_MOD0_FLOATA, FP16_K3);
    head<0, UO>();
    steps<UO, SW>(std::make_integer_sequence<int, 31>{});
    tail<31, UO>();
    TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_FLOATB, BF16_NEG1);  // restore LCONST_neg1
    TTI_SFPCONFIG(0, CK2, 0);
    TTI_SFPNOP;
}

// ---- fp32 grade (fp32 output on 32-bit DEST): degree-4 2^f (k4 in L0, k3 per-row SFPLOADI, k2 in LReg11,
// k1 in LReg12) and two Newton steps (e0 in the row's g LReg, which is free between T4 and MG) ----
constexpr uint32_t FP16_FK3 = 0x2B28, FP16_FK4 = 0x20E9, FK2_HI = 0x3E76, FK2_LO = 0x0154;

template <int R, int UO>
ALWI void tailf() {
    using X = Bank<R & 1>;
    TTI_SFPNOP;  // previous step's out MAD executes on this cycle
    TTI_SFPLOADI(X::s, sfpi::SFPLOADI_MOD0_FLOATA, FP16_FK3);
    TTI_SFPMAD(LK3, X::g, X::s, X::s, 0);  // p = k4 f + k3
    TTI_SFPMAD(X::s, X::g, CK2, X::s, 0);  // p = p f + k2
    TTI_SFPMAD(X::s, X::g, CK1, X::s, 0);  // p = p f + k1
    TTI_SFPMAD(X::s, X::g, C1, X::s, 0);   // p = p f + 1
    TTI_SFPNOP;
    TTI_SFPIADD(0, LZ, X::s, IADD_CC_NONE);
    TTI_SFPMAD(X::s, C1, C1, X::s, 0);      // d
    TTI_SFPARECIP(0, X::s, LR, 0);          // r0
    TTI_SFPMAD(X::s, LR, C1, X::g, NEG_A);  // e0 = 1 - d r0
    TTI_SFPMAD(LR, X::g, LR, LR, 0);        // r1 = r0 e0 + r0
    TTI_SFPMAD(X::s, LR, C1, X::s, NEG_A);  // e1 = 1 - d r1
    loadmacro<0, X::g, ra<R>()>();          // g -> X.g, q = r1 g at +1
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPMAD(X::g, X::s, X::g, LV, 0);  // V = q e1 + q
    loadmacro<1, LR, ra<R>() + UO>();
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
}

template <int R, int UO>
ALWI void step2f() {
    using X = Bank<R & 1>;
    using Y = Bank<(R + 1) & 1>;
    TTI_SFPLOADI(X::s, sfpi::SFPLOADI_MOD0_FLOATA, FP16_FK3);    // c0  T1   [prev out MAD]
    TTI_SFPMAD(LK3, X::g, X::s, X::s, 0);                        // c1  T2   p = k4 f + k3
    loadmacro<2, Y::g, ra<R + 1>()>();                           // c2  MG0  g -> Y.g; swap @c6 [prev store]
    TTI_SFPMAD(X::s, X::g, CK2, X::s, 0);                        // c3  T2b  p = p f + k2
    TTI_SFPLOADI(LV, sfpi::SFPLOADI_MOD0_FLOATB, BF16_NEG87);    // c4  H4   L7 = -87
    TTI_SFPMAD(X::s, X::g, CK1, X::s, 0);                        // c5  T3   p = p f + k1
    TTI_SFPLOADI(Y::s, sfpi::SFPLOADI_MOD0_FLOATB, BF16_POS87);  // c6  H6   [macro swap]
    TTI_SFPMAD(X::s, X::g, C1, X::s, 0);                         // c7  T4   p = p f + 1
    TTI_SFPSWAP(0, Y::s, Y::g, SWAP_MINMAX);                     // c8  H7   (2 cycles)
    TTI_SFPIADD(0, LZ, X::s, IADD_CC_NONE);                      // c10 T5
    TTI_SFPMAD(X::s, C1, C1, X::s, 0);                           // c11 T6   d
    TTI_SFPMAD(Y::g, CC1, CM, LZ, 0);                            // c12 H8   z
    TTI_SFPARECIP(0, X::s, LR, 0);                               // c13 T7   r0
    TTI_SFPMAD(X::s, LR, C1, X::g, NEG_A);                       // c14 T8a  e0 = 1 - d r0
    TTI_SFPMAD(LZ, C1, CM, Y::s, NEG_A);                         // c15 H9   s
    TTI_SFPMAD(LR, X::g, LR, LR, 0);                             // c16 T8b  r1 = r0 e0 + r0
    loadmacro<0, X::g, ra<R>()>();                               // c17 MG   g -> X.g; q = r1 g @c18
    TTI_SFPSHFT(23, LZ, LZ, SHFT_IMM_VC);                        // c18 H11  [q MAD]
    TTI_SFPMAD(X::s, LR, C1, X::s, NEG_A);                       // c19 T8c  e1 = 1 - d r1
    TTI_SFPMAD(Y::g, CC1, Y::s, Y::g, 0);                        // c20 H10  f
    TTI_SFPMAD(X::g, X::s, X::g, LV, 0);                         // c21 V    V = q e1 + q
    loadmacro<1, LR, ra<R>() + UO>();                            // c22 MU
}

template <int UO, int... I>
ALWI void stepsf(std::integer_sequence<int, I...>) {
    (step2f<I, UO>(), ...);
}

template <int UO>
inline void swiglu_lm_tile_f() {
    TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_USHORT, FK2_LO);
    TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, FK2_HI);
    TTI_SFPCONFIG(0, CK2, 0);
    TTI_SFPNOP;
    TTI_SFPLOADI(LK3, sfpi::SFPLOADI_MOD0_FLOATA, FP16_FK4);
    head<0, UO>();
    stepsf<UO>(std::make_integer_sequence<int, 31>{});
    tailf<31, UO>();
    TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_FLOATB, BF16_NEG1);  // restore LCONST_neg1
    TTI_SFPCONFIG(0, CK2, 0);
    TTI_SFPNOP;
}

// Load-macro state (init): templates via the VD >= 12 backdoor, sequences, Misc.
template <bool SW>
inline void swiglu_lm_init() {
    TTI_SFPMAD(LR, 0, C0, 12, 0);  // InstructionTemplate[0]: VD = Lr * VB(macro VD) + 0
    TTI_SFPMAD(LV, 0, C0, 13, 0);  // InstructionTemplate[1]: VD = Lv * VB(macro VD) + 0
    // Sequence bits per sub-unit: [2:0] 4+template / 3 store, [5:3] delay, 0x40 VD=L16, 0x80 VB=macro VD.
    {  // macro 0 (MG): MAD template 0, delay 0, in place
        constexpr uint32_t simple_bits = 0, mad_bits = 0x80 | (0 << 3) | 4, round_bits = 0, store_bits = 0;
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mad_bits << 8) | simple_bits);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store_bits << 8) | round_bits);
        TTI_SFPCONFIG(0, 4 + 0, 0);
    }
    if constexpr (D16) {
        TTI_SFP_STOCH_RND(0, 0, 9, 0, 14, 1);  // InstructionTemplate[2]: L16 = bf16_rne(macro VD)
    }
    {  // macro 1 (MU): MAD template 1 -> L16, delay 0; store L16 at delay 2. 16-bit DEST: MAD in place, round
       // (template 2) -> L16 at delay 2, store L16 at delay 3.
        constexpr uint32_t simple_bits = 0, mad_bits = 0x80 | (D16 ? 0 : 0x40) | (0 << 3) | 5,
                           round_bits = D16 ? (0x40 | (2 << 3) | 6) : 0, store_bits = 0x40 | ((D16 ? 3 : 2) << 3) | 3;
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mad_bits << 8) | simple_bits);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store_bits << 8) | round_bits);
        TTI_SFPCONFIG(0, 4 + 1, 0);
    }
    if constexpr (SW) {
        TTI_SFPSWAP(0, LV, 15, 9);  // InstructionTemplate[3]: VD(macro) = max(L7, VD), L7 = min
        // macro 2 (MG0): simple template 3, VB = macro VD (VC stays L7), delay 3 (instruction-counted)
        constexpr uint32_t simple_bits = 0x80 | (3 << 3) | 7, mad_bits = 0, round_bits = 0, store_bits = 0;
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mad_bits << 8) | simple_bits);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store_bits << 8) | round_bits);
        TTI_SFPCONFIG(0, 4 + 2, 0);
    }
    TTI_SFPCONFIG(0xff0, 8, 1);  // Misc: UsesLoadMod0ForStore = all, UnitDelayKind = WaitForElapsedInstructions
    TTI_SFPNOP;
}
}  // namespace lmk
// =====================================================================================================
// impl 4 "lm3": lm2 with FIXED registers (no bank alternation) and DEST addresses relative to the Dst RWC,
// which the last instruction of every step (MU, ADDR_MOD_6: dest += 2) advances by one SFPU row. Every
// steady-state step is then the SAME 19 (bf16 grade) / 22 (fp32 grade) instruction words: the first 16 are
// recorded once into the SFPU half of the replay buffer (slots 0..15; the FPU's matmul replay owns 16..31)
// and replayed, the rest are issued directly -> 4 (7) TRISC-pushed instructions per row instead of 19 (22).
// lm2 measured fast in isolation but its fully unrolled TRISC stream (~2.5 KB of .ttinsn per tile) ran up to
// 1.9x slower on the cores of the whole op whose L1 is busiest with outgoing NoC writes.
//   Registers: k3 (k4) L0, x L1, s L2, p/d/e L3, f (and e0) L4, z L5, r/u L6, -87/q/V L7.
//   The only cross-row value, f, is written by H10 straight into L4, which the next step's tail reads.
namespace lm3 {
using lmk::FK2_HI;
using lmk::FK2_LO;
using lmk::FP16_FK3;
using lmk::FP16_FK4;
constexpr uint32_t K = 0, H = 1, HS = 2, TS = 3, F = 4, Z = 5, LR = 6, L7 = 7;
constexpr uint32_t AM_INC = 6;  // ADDR_MOD_6: dest += 2 (one SFPU row), set by the init
template <uint32_t MACRO, uint32_t VD, uint32_t ADDR, uint32_t AMOD = AM>
ALWI void loadmacro() {
    TTI_SFPLOADMACRO((MACRO << 2) | (VD & 3), LD, AMOD, ADDR | (VD >> 2));
}

ALWI void head() {  // row 0 (RWC base 0)
    TTI_SFPLOAD(H, LD, AM, 0);
    TTI_SFPLOADI(HS, sfpi::SFPLOADI_MOD0_FLOATB, BF16_NEG87);
    TTI_SFPSWAP(0, H, HS, SWAP_MINMAX);
    TTI_SFPLOADI(HS, sfpi::SFPLOADI_MOD0_FLOATB, BF16_POS87);
    TTI_SFPSWAP(0, HS, H, SWAP_MINMAX);
    TTI_SFPMAD(H, CC1, CM, Z, 0);
    TTI_SFPMAD(Z, C1, CM, HS, NEG_A);
    TTI_SFPMAD(H, CC1, HS, F, 0);
    TTI_SFPSHFT(23, Z, Z, SHFT_IMM_VC);
}

template <bool FP32>
ALWI void tail(uint32_t) {  // row 31 (RWC base 62)
    TTI_SFPNOP;             // the previous step's out MAD executes on this cycle
    if constexpr (FP32) {
        TTI_SFPLOADI(TS, sfpi::SFPLOADI_MOD0_FLOATA, FP16_FK3);
        TTI_SFPMAD(K, F, TS, TS, 0);
        TTI_SFPMAD(TS, F, CK2, TS, 0);
    } else {
        TTI_SFPMAD(K, F, CK2, TS, 0);
    }
    TTI_SFPMAD(TS, F, CK1, TS, 0);
    TTI_SFPMAD(TS, F, C1, TS, 0);
    TTI_SFPNOP;
    TTI_SFPIADD(0, Z, TS, IADD_CC_NONE);
    TTI_SFPMAD(TS, C1, C1, TS, 0);
    TTI_SFPARECIP(0, TS, LR, 0);
    if constexpr (FP32) {
        TTI_SFPMAD(TS, LR, C1, F, NEG_A);  // e0
        TTI_SFPMAD(LR, F, LR, LR, 0);      // r1
    }
    TTI_SFPMAD(TS, LR, C1, TS, NEG_A);
    loadmacro<0, L7, 0>();
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPMAD(L7, TS, L7, L7, 0);
    loadmacro<1, LR, 64>();
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
}

// bf16 grade step, part 1 (16 words, recorded / replayed) and part 2 (3 words, issued).
ALWI void step_b_part1() {
    loadmacro<2, H, 2>();                                      // c0  MG0  g(R+1) -> x; swap(L7, x) @c4
    TTI_SFPMAD(K, F, CK2, TS, 0);                              // c1  T2
    TTI_SFPLOADI(L7, sfpi::SFPLOADI_MOD0_FLOATB, BF16_NEG87);  // c2  H4
    TTI_SFPMAD(TS, F, CK1, TS, 0);                             // c3  T3
    TTI_SFPLOADI(HS, sfpi::SFPLOADI_MOD0_FLOATB, BF16_POS87);  // c4  H6  [macro swap]
    TTI_SFPMAD(TS, F, C1, TS, 0);                              // c5  T4
    TTI_SFPSWAP(0, HS, H, SWAP_MINMAX);                        // c6  H7
    TTI_SFPIADD(0, Z, TS, IADD_CC_NONE);                       // c8  T5
    TTI_SFPMAD(TS, C1, C1, TS, 0);                             // c9  T6
    TTI_SFPMAD(H, CC1, CM, Z, 0);                              // c10 H8
    TTI_SFPARECIP(0, TS, LR, 0);                               // c11 T7
    TTI_SFPMAD(Z, C1, CM, HS, NEG_A);                          // c12 H9
    TTI_SFPMAD(TS, LR, C1, TS, NEG_A);                         // c13 T8
    loadmacro<0, L7, 0>();                                     // c14 MG   g(R) -> L7; q @c15
    TTI_SFPSHFT(23, Z, Z, SHFT_IMM_VC);                        // c15 H11  [q]
}
ALWI void step_b_part2() {
    TTI_SFPMAD(H, CC1, HS, F, 0);    // c16 H10  f(R+1) -> L4
    TTI_SFPMAD(L7, TS, L7, L7, 0);   // c17 V
    loadmacro<1, LR, 64, AM_INC>();  // c18 MU   u(R) -> L6; out @c0', store @c2'; RWC += 2
}

// fp32 grade step: 16 + 6 words.
ALWI void step_f_part1() {
    TTI_SFPLOADI(TS, sfpi::SFPLOADI_MOD0_FLOATA, FP16_FK3);    // c0  T1
    TTI_SFPMAD(K, F, TS, TS, 0);                               // c1  T2
    loadmacro<2, H, 2>();                                      // c2  MG0  swap @c6
    TTI_SFPMAD(TS, F, CK2, TS, 0);                             // c3  T2b
    TTI_SFPLOADI(L7, sfpi::SFPLOADI_MOD0_FLOATB, BF16_NEG87);  // c4  H4
    TTI_SFPMAD(TS, F, CK1, TS, 0);                             // c5  T3
    TTI_SFPLOADI(HS, sfpi::SFPLOADI_MOD0_FLOATB, BF16_POS87);  // c6  H6  [macro swap]
    TTI_SFPMAD(TS, F, C1, TS, 0);                              // c7  T4
    TTI_SFPSWAP(0, HS, H, SWAP_MINMAX);                        // c8  H7
    TTI_SFPIADD(0, Z, TS, IADD_CC_NONE);                       // c10 T5
    TTI_SFPMAD(TS, C1, C1, TS, 0);                             // c11 T6
    TTI_SFPMAD(H, CC1, CM, Z, 0);                              // c12 H8
    TTI_SFPARECIP(0, TS, LR, 0);                               // c13 T7
    TTI_SFPMAD(TS, LR, C1, F, NEG_A);                          // c14 T8a  e0 -> L4
    TTI_SFPMAD(Z, C1, CM, HS, NEG_A);                          // c15 H9
    TTI_SFPMAD(LR, F, LR, LR, 0);                              // c16 T8b  r1
}
ALWI void step_f_part2() {
    loadmacro<0, L7, 0>();               // c17 MG   q @c18
    TTI_SFPSHFT(23, Z, Z, SHFT_IMM_VC);  // c18 H11  [q]
    TTI_SFPMAD(TS, LR, C1, TS, NEG_A);   // c19 T8c  e1
    TTI_SFPMAD(H, CC1, HS, F, 0);        // c20 H10
    TTI_SFPMAD(L7, TS, L7, L7, 0);       // c21 V
    loadmacro<1, LR, 64, AM_INC>();      // c22 MU
}

template <bool FP32>
inline void swiglu_lm3_tile() {
    if constexpr (FP32) {
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_USHORT, FK2_LO);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, FK2_HI);
    } else {
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_FLOATA, FP16_K2);
    }
    TTI_SFPCONFIG(0, CK2, 0);
    TTI_SFPNOP;
    TTI_SFPLOADI(K, sfpi::SFPLOADI_MOD0_FLOATA, FP32 ? FP16_FK4 : FP16_K3);
    head();
    lltt::record<lltt::Exec>(0, 16);
    if constexpr (FP32) {
        step_f_part1();
        step_f_part2();
    } else {
        step_b_part1();
        step_b_part2();
    }
#pragma GCC unroll 0
    for (int r = 1; r < 31; r++) {
        lltt::replay(0, 16);
        if constexpr (FP32) {
            step_f_part2();
        } else {
            step_b_part2();
        }
    }
    tail<FP32>(0);
    TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_FLOATB, BF16_NEG1);  // restore LCONST_neg1
    TTI_SFPCONFIG(0, CK2, 0);
    TTI_SFPNOP;
}

inline void swiglu_lm3_init() {
    lmk::swiglu_lm_init<true>();  // same templates / sequences as lm2 (L7 = V = swap temp, LR = r / u)
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 2}}.set(ADDR_MOD_6);
}
}  // namespace lm3
}  // namespace swlm
}  // namespace ckernel::sfpu
#endif  // TRISC_MATH

namespace compute_kernel_lib {

// impl 1: tuned (K2 via SFPLOADI), impl 2: tuned + k2 in LReg11. Gate D0, up D1, out D0. bf16 grade, 32-bit
// DEST only; other configs fall back to the op's element.
template <uint32_t impl, bool fp32_grade>
struct SwiGluLM : BinaryOp<SwiGluLM<impl, fp32_grade>, Dst::D0, Dst::D1, Dst::D0> {
    static constexpr bool raw = (!fp32_grade && impl <= 4) || (fp32_grade && (impl == 3 || impl == 4));
    static constexpr bool lm = raw && (impl == 0 || impl == 3 || impl == 4);
    static constexpr Dst out_slot = lm ? Dst::D1 : Dst::D0;  // lm / lm2 store to the up slot
    static ALWI void init() {
        SwiGluMul<fp32_grade>::init();
        if constexpr (lm && impl == 4) {
            MATH((sfpu::swlm::lm3::swiglu_lm3_init()));
        } else if constexpr (lm) {
            MATH((sfpu::swlm::lmk::swiglu_lm_init<impl == 3>()));
        }
    }
    static ALWI void exec_impl(uint32_t slot_offset) {
        if constexpr (lm && impl == 4) {
            MATH((_llk_math_eltwise_unary_sfpu_params_(
                sfpu::swlm::lm3::swiglu_lm3_tile<fp32_grade>, to_u32(Dst::D0) + slot_offset, VectorMode::None)));
        } else if constexpr (lm && fp32_grade) {
            MATH((_llk_math_eltwise_unary_sfpu_params_(
                sfpu::swlm::lmk::swiglu_lm_tile_f<64>, to_u32(Dst::D0) + slot_offset, VectorMode::None)));
        } else if constexpr (lm) {
            MATH((_llk_math_eltwise_unary_sfpu_params_(
                sfpu::swlm::lmk::swiglu_lm_tile < 64, impl == 3 >, to_u32(Dst::D0) + slot_offset, VectorMode::None)));
        } else if constexpr (raw) {
            MATH((_llk_math_eltwise_unary_sfpu_params_(
                sfpu::swlm::swiglu_tuned_tile < 64, 0, impl == 2 >, to_u32(Dst::D0) + slot_offset, VectorMode::None)));
        } else {
            SwiGluMul<fp32_grade>::exec_impl(slot_offset);
        }
    }
};

}  // namespace compute_kernel_lib
