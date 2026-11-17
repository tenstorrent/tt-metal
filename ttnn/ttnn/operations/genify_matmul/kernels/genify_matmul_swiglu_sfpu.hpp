// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Fused swiglu epilogue element for eltwise_chain: out = silu(gate) * up in ONE SFPU pass over two DEST slots,
// as a hand-scheduled, SFPLOADMACRO-pipelined instruction stream over the whole tile (perf round 2,
// perf_experiments/swiglu_epilogue_loadmacro "lm2": isolated focus epilogue 61.8 -> 46.6 us, fp32 out
// 75.7 -> 54.9 us; same math and precision as the round-1 element, outputs equal except 2 / 86016 sub-1e-37
// elements on the bf16 focus block).
//
//   eltwise_chain(shape,
//       CopyTile<gate ..., Dst::D0>{...},
//       CopyTile<up ...,   Dst::D1>{...},
//       SwiGluMul<fp32_grade>{},
//       PackTile<..., SwiGluMul<fp32_grade>::out_slot>{...});   // the result lands in the UP slot (D1)
//
// RAW-LLK, justified by measurement (Perf 2) -- do NOT "fix" it back to SFPI / a helper: the SFPI element
// (round 1) measured 61.8 us on the isolated PERF FOCUS 1 epilogue, this stream 46.6 us (whole op, with the tail
// write split: 418.3 -> 391.9 us together with the head ramp), identical PCC / rel-RMS / max-ULP.
// Helper bypass (raw LLK, capability): kernel_lib has no silu(a)*b element; the element is TTI_SFP* words with
// explicit LRegs because neither SFPI nor any LLK exposes SFPLOADMACRO scheduling or cross-row software
// pipelining. The chain is NOT bypassed (it owns init, DEST sync, CB sync, format reconfig, pack).
// Blackhole only (SFPARECIP, SFPLOADMACRO sequences, SFPSWAP mode 9).
//
// Math (unchanged from round 1): x = clamp(g, +-87); z = x*c1 + M (c1 = -log2 e, M = 1.5*2^23); s = M - z;
// f = x*c1 + s; p = 2^f polynomial (bf16 grade: degree 3; fp32 grade: degree 4); E = p +int (z << 23);
// d = E + 1; r = arecip(d) + Newton (bf16: 1 step folded into the product; fp32: 2 steps);
// out = ((g r)(1 + e)) u -- the same 3 product roundings as ((g u) r)(1 + e).
//
// Schedule (one SFPU row = 32 lanes; 32 rows per tile; rows R and R+1 overlap: tail(R) || head(R+1)):
//   * whole tile = ONE stream (no per-face drain), two LReg banks alternating by row parity
//     (A: g L1, s L2; B: g L3, s L4), z L5, r / u L6, -87 / V L7, k3 (k4) L0, k2 in LReg11 (restored to -1.0
//     at the end of every tile), k1 / c1 / M in LReg12..14;
//   * three loads double as SFPLOADMACROs:
//       MG0 (g of row R+1)  schedules the first clamp SFPSWAP(VC = L7 = -87, VD = g, mode 9 -> VD = max) on the
//                           simple sub-unit 4 issues later, on a cycle whose regular issue is an SFPLOADI (MAD
//                           idle) followed by an SFPMAD (simple / round idle);
//       MG  (g of row R, re-loaded) schedules q = r * g on the MAD sub-unit on the next cycle (regular issue:
//                           SFPSHFT);
//       MU  (u of row R)    schedules L16 = V * u (MAD) and the SFPSTORE of L16 to u's own DEST address
//                           (16-bit DEST: MAD in place, SFPSTOCHRND -> L16 on the round sub-unit, then store),
//                           all concurrent with the next step's first issues (SFPLOAD / SFPLOADI);
//   * bf16 grade: 18 issues + 1 SWAP cycle = 19 cycles per row (round-1 element: 22 issues + 2 SWAP cycles +
//     ~4 MAD-dependency / face-drain bubbles = ~28); fp32 grade: 22 + 1 = 23 cycles per row (was ~35).
// Hazard rules this schedule obeys (BH ISA, SFPLOADMACRO / SFPMAD / SFPSWAP pages):
//   - a macro-scheduled op silently DISCARDS a regular op issued to the same sub-unit on the same cycle;
//   - there is no auto-stall for macro-scheduled results: consumers sit >= 2 cycles after a scheduled SFPMAD;
//   - SFPIADD reads VD without the MAD auto-stall: it never directly follows the SFPMAD that wrote its VD;
//   - delays are instruction-counted (Misc.UnitDelayKind = all); the tile ends with 5 SFPNOPs so the last
//     scheduled MAD / round / store drain.
// Program state owned between init and the last exec (do not interleave other SFPU code): LoadMacroConfig
// (templates 0..3, sequences 0..2, Misc), LReg12..14 constants; LReg11 is set / restored per tile.

#pragma once

#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"

#ifdef TRISC_MATH
#include "sfpi.h"
#include "ckernel_sfpu_sigmoid.h"  // math::reset_counters / p_setrwc via cmath_common
#include "llk_math_eltwise_binary_sfpu_macros.h"
#include "llk_math_eltwise_unary_sfpu_params.h"

namespace ckernel::sfpu::swiglu_lm {

constexpr float NEG_LOG2E = -1.4426950408889634f;
constexpr float MAGIC = 12582912.0f;  // 1.5 * 2^23
// bf16 grade, degree 3 (k1 in LReg12, k2 in LReg11, k3 in L0)
constexpr float B_K1 = 0.6932584643363953f;
constexpr uint32_t FP16_B_K2 = 0x33C1, FP16_B_K3 = 0x2B15;  // 0.2423095703125, 0.055328369140625
// fp32 grade, degree 4 (k1 in LReg12, k2 in LReg11, k3 per-row SFPLOADI, k4 in L0)
constexpr float F_K1 = 0.6931241750717163f;
constexpr uint32_t F_K2_HI = 0x3E76, F_K2_LO = 0x0154;      // 0.2402394413948059
constexpr uint32_t FP16_F_K3 = 0x2B28, FP16_F_K4 = 0x20E9;  // 0.055908203125, 0.00959014892578125
constexpr uint32_t BF16_NEG87 = 0xC2AE, BF16_POS87 = 0x42AE, BF16_NEG1 = 0xBF80;

constexpr uint32_t C0 = 9, C1 = 10, CK2 = 11, CK1 = 12, CC1 = 13, CM = 14;  // const LRegs
constexpr uint32_t LK = 0, LZ = 5, LR = 6, L7 = 7;
constexpr uint32_t LD = 0;  // InstrModLoadStore::DEFAULT: FP32 on 32-bit DEST, BF16 on 16-bit DEST
constexpr uint32_t AM = 7;  // ADDR_MOD_7 (no increment; set by the SFPU init)
constexpr uint32_t NEG_A = 1, IADD_CC_NONE = 4, SHFT_IMM_VC = 5, SWAP_MINMAX = 1;
constexpr int UO = 64;  // up slot (D1) - gate slot (D0), in SFPLOAD address units
constexpr bool D16 = !DST_ACCUM_MODE;

template <int B>
struct Bank {
    static constexpr uint32_t g = B ? 3 : 1, s = B ? 4 : 2;
};
// whole-tile SFPU row R (0..31): address 2R (face R/8 at 16 * (R/8), row pair 2 * (R%8) inside it)
template <int R>
constexpr uint32_t ra() {
    return 2 * R;
}
// SFPLOADMACRO: lreg_ind = macro << 2 | VD[1:0]; the address low bit carries VD[2] (DEST bases are even).
template <uint32_t MACRO, uint32_t VD, uint32_t ADDR>
ALWI void loadmacro() {
    TTI_SFPLOADMACRO((MACRO << 2) | (VD & 3), LD, AM, ADDR | (VD >> 2));
}

// row 0: pipeline fill (regular issues; hardware auto-stalls cover the MAD hazards)
template <int R>
ALWI void head() {
    using X = Bank<R & 1>;
    TTI_SFPLOAD(X::g, LD, AM, ra<R>());
    TTI_SFPLOADI(X::s, sfpi::SFPLOADI_MOD0_FLOATB, BF16_NEG87);
    TTI_SFPSWAP(0, X::g, X::s, SWAP_MINMAX);  // g = max(g, -87)
    TTI_SFPLOADI(X::s, sfpi::SFPLOADI_MOD0_FLOATB, BF16_POS87);
    TTI_SFPSWAP(0, X::s, X::g, SWAP_MINMAX);  // x = min(87, g)
    TTI_SFPMAD(X::g, CC1, CM, LZ, 0);         // z = x c1 + M
    TTI_SFPMAD(LZ, C1, CM, X::s, NEG_A);      // s = M - z
    TTI_SFPMAD(X::g, CC1, X::s, X::g, 0);     // f = x c1 + s
    TTI_SFPSHFT(23, LZ, LZ, SHFT_IMM_VC);     // z <<= 23
}

// row 31: pipeline drain
template <int R, bool FP32>
ALWI void tail() {
    using X = Bank<R & 1>;
    TTI_SFPNOP;  // the previous step's out MAD executes on this cycle: a regular SFPMAD here would be discarded
    if constexpr (FP32) {
        TTI_SFPLOADI(X::s, sfpi::SFPLOADI_MOD0_FLOATA, FP16_F_K3);
        TTI_SFPMAD(LK, X::g, X::s, X::s, 0);  // p = k4 f + k3
        TTI_SFPMAD(X::s, X::g, CK2, X::s, 0);
    } else {
        TTI_SFPMAD(LK, X::g, CK2, X::s, 0);  // p = k3 f + k2
    }
    TTI_SFPMAD(X::s, X::g, CK1, X::s, 0);
    TTI_SFPMAD(X::s, X::g, C1, X::s, 0);
    TTI_SFPNOP;                              // SFPIADD reads VD: not covered by the MAD auto-stall
    TTI_SFPIADD(0, LZ, X::s, IADD_CC_NONE);  // E = p +int z
    TTI_SFPMAD(X::s, C1, C1, X::s, 0);       // d = E + 1
    TTI_SFPARECIP(0, X::s, LR, 0);           // r = arecip(d)
    if constexpr (FP32) {
        TTI_SFPMAD(X::s, LR, C1, X::g, NEG_A);  // e0 = 1 - d r0
        TTI_SFPMAD(LR, X::g, LR, LR, 0);        // r1 = r0 e0 + r0
    }
    TTI_SFPMAD(X::s, LR, C1, X::s, NEG_A);  // e = 1 - d r
    loadmacro<0, X::g, ra<R>()>();          // g -> X.g; q = r g at +1
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPMAD(X::g, X::s, X::g, L7, 0);  // V = q e + q
    loadmacro<1, LR, ra<R>() + UO>();     // u -> L6; L16 = V u; (round;) store L16 -> u's address
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
}

// bf16 grade steady state: tail(row R) [bank X] || head(row R+1) [bank Y]; cycle in the comments (SWAP = 2).
template <int R>
ALWI void step_b() {
    using X = Bank<R & 1>;
    using Y = Bank<(R + 1) & 1>;
    loadmacro<2, Y::g, ra<R + 1>()>();                           // c0  MG0  g -> Y.g; swap(L7, Y.g) @c4 [prev out MAD]
    TTI_SFPMAD(LK, X::g, CK2, X::s, 0);                          // c1  p = k3 f + k2
    TTI_SFPLOADI(L7, sfpi::SFPLOADI_MOD0_FLOATB, BF16_NEG87);    // c2  L7 = -87       [prev store / round]
    TTI_SFPMAD(X::s, X::g, CK1, X::s, 0);                        // c3  p = p f + k1
    TTI_SFPLOADI(Y::s, sfpi::SFPLOADI_MOD0_FLOATB, BF16_POS87);  // c4                 [macro swap: g = max(g, -87)]
    TTI_SFPMAD(X::s, X::g, C1, X::s, 0);                         // c5  p = p f + 1
    TTI_SFPSWAP(0, Y::s, Y::g, SWAP_MINMAX);                     // c6  x = min(87, g) (2 cycles)
    TTI_SFPIADD(0, LZ, X::s, IADD_CC_NONE);                      // c8  E = p +int z
    TTI_SFPMAD(X::s, C1, C1, X::s, 0);                           // c9  d = E + 1
    TTI_SFPMAD(Y::g, CC1, CM, LZ, 0);                            // c10 z = x c1 + M
    TTI_SFPARECIP(0, X::s, LR, 0);                               // c11 r = arecip(d)
    TTI_SFPMAD(LZ, C1, CM, Y::s, NEG_A);                         // c12 s = M - z
    TTI_SFPMAD(X::s, LR, C1, X::s, NEG_A);                       // c13 e = 1 - d r
    loadmacro<0, X::g, ra<R>()>();                               // c14 MG   g -> X.g; q = r g @c15
    TTI_SFPSHFT(23, LZ, LZ, SHFT_IMM_VC);                        // c15 z <<= 23       [q MAD]
    TTI_SFPMAD(Y::g, CC1, Y::s, Y::g, 0);                        // c16 f = x c1 + s
    TTI_SFPMAD(X::g, X::s, X::g, L7, 0);                         // c17 V = q e + q
    loadmacro<1, LR, ra<R>() + UO>();                            // c18 MU   u -> L6; L16 = V u @c0'; store @c2'
}

// fp32 grade steady state (degree-4 polynomial, two Newton steps; e0 lives in the row's g LReg, free between
// the last polynomial MAD and MG).
template <int R>
ALWI void step_f() {
    using X = Bank<R & 1>;
    using Y = Bank<(R + 1) & 1>;
    TTI_SFPLOADI(X::s, sfpi::SFPLOADI_MOD0_FLOATA, FP16_F_K3);   // c0  [prev out MAD]
    TTI_SFPMAD(LK, X::g, X::s, X::s, 0);                         // c1  p = k4 f + k3
    loadmacro<2, Y::g, ra<R + 1>()>();                           // c2  MG0  swap @c6  [prev store]
    TTI_SFPMAD(X::s, X::g, CK2, X::s, 0);                        // c3  p = p f + k2
    TTI_SFPLOADI(L7, sfpi::SFPLOADI_MOD0_FLOATB, BF16_NEG87);    // c4
    TTI_SFPMAD(X::s, X::g, CK1, X::s, 0);                        // c5  p = p f + k1
    TTI_SFPLOADI(Y::s, sfpi::SFPLOADI_MOD0_FLOATB, BF16_POS87);  // c6  [macro swap]
    TTI_SFPMAD(X::s, X::g, C1, X::s, 0);                         // c7  p = p f + 1
    TTI_SFPSWAP(0, Y::s, Y::g, SWAP_MINMAX);                     // c8  (2 cycles)
    TTI_SFPIADD(0, LZ, X::s, IADD_CC_NONE);                      // c10 E
    TTI_SFPMAD(X::s, C1, C1, X::s, 0);                           // c11 d
    TTI_SFPMAD(Y::g, CC1, CM, LZ, 0);                            // c12 z
    TTI_SFPARECIP(0, X::s, LR, 0);                               // c13 r0
    TTI_SFPMAD(X::s, LR, C1, X::g, NEG_A);                       // c14 e0 = 1 - d r0
    TTI_SFPMAD(LZ, C1, CM, Y::s, NEG_A);                         // c15 s
    TTI_SFPMAD(LR, X::g, LR, LR, 0);                             // c16 r1 = r0 e0 + r0
    loadmacro<0, X::g, ra<R>()>();                               // c17 MG   q = r1 g @c18
    TTI_SFPSHFT(23, LZ, LZ, SHFT_IMM_VC);                        // c18 [q MAD]
    TTI_SFPMAD(X::s, LR, C1, X::s, NEG_A);                       // c19 e1 = 1 - d r1
    TTI_SFPMAD(Y::g, CC1, Y::s, Y::g, 0);                        // c20 f
    TTI_SFPMAD(X::g, X::s, X::g, L7, 0);                         // c21 V = q e1 + q
    loadmacro<1, LR, ra<R>() + UO>();                            // c22 MU
}

template <bool FP32, int... I>
ALWI void steps(std::integer_sequence<int, I...>) {
    if constexpr (FP32) {
        (step_f<I>(), ...);
    } else {
        (step_b<I>(), ...);
    }
}

// One whole tile: gate at the DEST base (D0), up at +64 (D1); the result overwrites up.
template <bool FP32>
inline void swiglu_tile() {
    if constexpr (FP32) {
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_USHORT, F_K2_LO);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, F_K2_HI);
    } else {
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_FLOATA, FP16_B_K2);
    }
    TTI_SFPCONFIG(0, CK2, 0);  // LReg11 = k2
    TTI_SFPNOP;
    TTI_SFPLOADI(LK, sfpi::SFPLOADI_MOD0_FLOATA, FP32 ? FP16_F_K4 : FP16_B_K3);
    head<0>();
    steps<FP32>(std::make_integer_sequence<int, 31>{});
    tail<31, FP32>();
    TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_FLOATB, BF16_NEG1);  // restore LCONST_neg1 (core-wide -1.0)
    TTI_SFPCONFIG(0, CK2, 0);
    TTI_SFPNOP;
}

template <uint32_t MACRO, uint32_t SIMPLE, uint32_t MAD, uint32_t ROUND, uint32_t STORE>
ALWI void sequence() {
    TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (MAD << 8) | SIMPLE);
    TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (STORE << 8) | ROUND);
    TTI_SFPCONFIG(0, 4 + MACRO, 0);
}

template <bool FP32>
inline void swiglu_init() {
    math::reset_counters(p_setrwc::SET_ABD_F);
    sfpi::vConstFloatPrgm0 = FP32 ? F_K1 : B_K1;  // LReg12
    sfpi::vConstFloatPrgm1 = NEG_LOG2E;           // LReg13
    sfpi::vConstFloatPrgm2 = MAGIC;               // LReg14
    // InstructionTemplate[i] via the VD >= 12 backdoor
    TTI_SFPMAD(LR, 0, C0, 12, 0);  // [0] VD = L6 * VB(macro VD) + 0          (q = r g)
    TTI_SFPMAD(L7, 0, C0, 13, 0);  // [1] VD = L7 * VB(macro VD) + 0          (out = V u)
    if constexpr (D16) {
        TTI_SFP_STOCH_RND(0, 0, 9, 0, 14, 1);  // [2] VD = bf16_rne(VC = macro VD)
    }
    TTI_SFPSWAP(0, L7, 15, 9);  // [3] VD(macro) = max(L7, VD), L7 = min     (g = max(g, -87))
    // Sequence bits per sub-unit: [2:0] 4 + template / 3 store, [5:3] delay, 0x40 VD = L16, 0x80 VB = macro VD.
    sequence<0, 0, 0x80 | (0 << 3) | 4, 0, 0>();  // MG:  q MAD in place, delay 0
    if constexpr (D16) {                          // MU:  MAD in place, round -> L16 @2, store L16 @3
        sequence<1, 0, 0x80 | (0 << 3) | 5, 0x40 | (2 << 3) | 6, 0x40 | (3 << 3) | 3>();
    } else {  //                                     MU:  MAD -> L16, store L16 @2
        sequence<1, 0, 0x80 | 0x40 | (0 << 3) | 5, 0, 0x40 | (2 << 3) | 3>();
    }
    sequence<2, 0x80 | (3 << 3) | 7, 0, 0, 0>();  // MG0: swap (VC stays L7), delay 3
    TTI_SFPCONFIG(0xff0, 8, 1);                   // Misc: store uses load Mod0; all delays instruction-counted
    TTI_SFPNOP;
}

}  // namespace ckernel::sfpu::swiglu_lm
#endif  // TRISC_MATH

namespace compute_kernel_lib {

// out (= Dst::D1, the up slot) = silu(D0) * D1. fp32_grade: the op's silu_fp32_precision && DST_ACCUM_MODE.
template <bool fp32_grade>
struct SwiGluMul : BinaryOp<SwiGluMul<fp32_grade>, Dst::D0, Dst::D1, Dst::D1> {
    static constexpr Dst out_slot = Dst::D1;
    static ALWI void init() { MATH((SFPU_BINARY_INIT_FN_NO_ARGS(unused, sfpu::swiglu_lm::swiglu_init<fp32_grade>))); }
    static ALWI void exec_impl(uint32_t slot_offset) {
        MATH((_llk_math_eltwise_unary_sfpu_params_(
            sfpu::swiglu_lm::swiglu_tile<fp32_grade>, to_u32(Dst::D0) + slot_offset, VectorMode::None)));
    }
};

}  // namespace compute_kernel_lib
