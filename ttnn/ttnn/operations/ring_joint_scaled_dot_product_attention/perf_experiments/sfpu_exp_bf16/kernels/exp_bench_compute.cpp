// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// sfpu_exp_bf16 bench: the ring-joint SDPA softmax exp phases, isolated.
//   MODE 0 (exp_block):     P = exp(s (S - m)) in place on a Sq x Sk score block (BinaryFpu Sub bcast col ->
//                           ScaledExp -> PackTile, Strided), exactly the op's chain.
//   MODE 1 (rescale exp):   a = exp(s (m_acc - m_blk)) over Sq tiles (BinaryFpu Sub -> ScaledExp -> PackTile).
// VARIANT selects the ScaledExp implementation (all at bf16 DEST, approx=false, same exp_21f math):
//   0 base       : stock exp_tile<false, true>(scale)                         (the op today)
//   1 trim       : stock TTI body minus the upper clamp (LOADI 255 + SWAP)     (bit-exact for x*s <= 88)
//   2 split      : RND/CAST int-frac split (11 instr/row), one row at a time
//   3 split2     : split, two rows interleaved per replay body
//   4 fold2      : split2 with the scale folded into the 1/ln2 MAD constant (10 instr/row; not bit-exact)
//   5 noexp      : no exp at all (ablation: the Sub + pack floor)
//
// RAW-LLK JUSTIFICATION (variants 1-4): bypasses exp_tile<false,true>() / _sfpu_exp_21f_bf16_tti_. The stock
// body spends 7 SIMPLE-unit ops per row on the float->fixed int/frac split (EXEXP, EXMAN, SHFT, EXMAN9) and the
// [0,255] clamp (LOADI+SWAP, GT+AND). For x <= 0 (softmax after the max subtraction) the same split is one
// SFP_STOCH_RND(round-to-zero, fp32->uint16) = floor(t) (saturates t<=0 / -inf to 0), one SFPCAST and one MAD
// (phi = t - floor(t), exact); the poly coefficients are rescaled by exact powers of two (c1*2^23, c2*2^46) so
// the Horner evaluation is bit-identical; SFPSETEXP(mod 0) takes floor(t) as the biased exponent.
#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/cb_api.h"
#include "api/compute/eltwise_unary/exp.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/math.hpp"
#include "scaled_exp_bf16.hpp"  // drop-in ScaledExp (variant 11)

using namespace compute_kernel_lib;

#ifdef TRISC_MATH
namespace exp_bench {
using namespace ckernel;
using namespace ckernel::sfpu;

constexpr uint32_t C1_BITS = 0x33a85adau;             // stock c1 = 7.839635e-08
constexpr uint32_t C2_BITS = 0x27aca418u;             // stock c2 = 4.791750e-15
constexpr uint32_t C1P_BITS = C1_BITS + (23u << 23);  // c1 * 2^23 (exact)
constexpr uint32_t C2P_BITS = C2_BITS + (46u << 23);  // c2 * 2^46 (exact)

constexpr float bf16_to_float(uint32_t b) { return __builtin_bit_cast(float, b << 16); }
template <uint32_t scale_bf16>
constexpr uint32_t k_bits() {  // fp32(s / ln2), s = the bf16 scale
    return __builtin_bit_cast(
        uint32_t, static_cast<float>(static_cast<double>(bf16_to_float(scale_bf16)) * 1.4426950408889634));
}

// Constants for the split variants. Stock exp_tile_init<false>() already did SFPU config reg, ADDR_MOD_6 (dest
// +2) / ADDR_MOD_7 (+0), L12 = 1/ln2, L13 = c2. We override:
//   L12 = 1/ln2 (split/split2) or s/ln2 (fold2), L13 = c2*2^46, L14 = c1*2^23, L6 = 127.0, L7 = c0 (fp16a 0x3c02).
template <uint32_t scale_bf16, bool FOLD>
inline void split_init() {
    if constexpr (FOLD) {
        constexpr uint32_t kb = k_bits<scale_bf16>();
        TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_UPPER, (kb >> 16) & 0xFFFF);
        TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_LOWER, kb & 0xFFFF);
        TTI_SFPCONFIG(0, p_sfpu::LREG12, 0);
    }
    TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_UPPER, (C2P_BITS >> 16) & 0xFFFF);
    TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_LOWER, C2P_BITS & 0xFFFF);
    TTI_SFPCONFIG(0, p_sfpu::LREG13, 0);
    TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_UPPER, (C1P_BITS >> 16) & 0xFFFF);
    TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_LOWER, C1P_BITS & 0xFFFF);
    TTI_SFPCONFIG(0, p_sfpu::LREG14, 0);
}

ALWI void split_consts_lregs() {
    TTI_SFPLOADI(p_sfpu::LREG6, sfpi::SFPLOADI_MOD0_FLOATB, 0x42fe);  // 127.0
    TTI_SFPLOADI(p_sfpu::LREG7, sfpi::SFPLOADI_MOD0_FLOATA, 0x3c02);  // c0 (stock fp16a value)
}

constexpr unsigned RND_ZERO = 2;
constexpr unsigned STOCH_FP32_TO_UINT16 = 6;
constexpr unsigned MAD_NEGATE_VA = 1;

// ---- variant 1: stock body without the upper clamp ------------------------------------------------------------
template <uint32_t scale_bf16>
inline void trim_face() {
    TTI_SFPLOADI(p_sfpu::LREG5, sfpi::SFPLOADI_MOD0_FLOATB, 0x42fe);
    TTI_SFPLOADI(p_sfpu::LREG6, sfpi::SFPLOADI_MOD0_UPPER, 0x33a8);
    TTI_SFPLOADI(p_sfpu::LREG6, sfpi::SFPLOADI_MOD0_LOWER, 0x5ada);
    TTI_SFPLOADI(p_sfpu::LREG7, sfpi::SFPLOADI_MOD0_FLOATA, 0x3c02);
    constexpr int BODY_LEN = 15;
    TTI_REPLAY(0, BODY_LEN, 1, 1);
    TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_7, 0);
    TTI_SFPMULI(scale_bf16, p_sfpu::LREG0, 0);
    TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG12, p_sfpu::LREG5, p_sfpu::LREG3, 0);
    TTI_SFPEXEXP(0, p_sfpu::LREG3, p_sfpu::LREG1, 0);
    TTI_SFPEXMAN(0, p_sfpu::LREG3, p_sfpu::LREG0, 0);
    TTI_SFPSHFT(0, p_sfpu::LREG1, p_sfpu::LREG0, 0);
    TTI_SFPEXMAN(0, p_sfpu::LREG0, p_sfpu::LREG1, sfpi::SFPEXMAN_MOD1_PAD9);
    TTI_SFPCAST(p_sfpu::LREG1, p_sfpu::LREG1, 0);
    TTI_SFPMAD(p_sfpu::LREG1, p_sfpu::LREG13, p_sfpu::LREG6, p_sfpu::LREG2, 0);
    TTI_SFPGT(0, p_sfpu::LCONST_0, p_sfpu::LREG3, 8);
    TTI_SFPMAD(p_sfpu::LREG2, p_sfpu::LREG1, p_sfpu::LREG7, p_sfpu::LREG1, 0);
    TTI_SFPAND(p_sfpu::LREG0, p_sfpu::LREG3, p_sfpu::LREG0, 1);
    TTI_SFPSETEXP(0, p_sfpu::LREG1, p_sfpu::LREG0, 2);
    TTI_SFP_STOCH_RND(
        sfpi::SFPSTOCHRND_RND_EVEN,
        0,
        p_sfpu::LREG0,
        p_sfpu::LREG0,
        p_sfpu::LREG0,
        sfpi::SFPSTOCHRND_MOD1_FP32_TO_FP16B);
    TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_6, 0);
#pragma GCC unroll 8
    for (int i = 1; i < 8; i++) {
        TTI_REPLAY(0, BODY_LEN, 0, 0);
    }
}

// ---- split body for one row in registers (a, n, f) --------------------------------------------------------------
// t = (x*s)/ln2 + 127 ; n = trunc_u16(t) ; f = t - float(n) ; p = (f*c2' + c1')*f + c0 ; y = setexp(p, n) ; bf16 RNE
#define SPLIT_LOAD(A, OFF) TTI_SFPLOAD(A, InstrModLoadStore::DEFAULT, ADDR_MOD_7, OFF)
#define SPLIT_MULI(A) TTI_SFPMULI(scale_bf16, A, 0)
#define SPLIT_T(A) TTI_SFPMAD(A, p_sfpu::LREG12, p_sfpu::LREG6, A, 0)
#define SPLIT_N(A, N) TTI_SFP_STOCH_RND(RND_ZERO, 0, 0, A, N, STOCH_FP32_TO_UINT16)
#define SPLIT_NF(N, F) TTI_SFPCAST(N, F, 0)
#define SPLIT_F(A, F) TTI_SFPMAD(F, p_sfpu::LCONST_1, A, F, MAD_NEGATE_VA)
#define SPLIT_P1(A, F) TTI_SFPMAD(F, p_sfpu::LREG13, p_sfpu::LREG14, A, 0)
#define SPLIT_P2(A, F) TTI_SFPMAD(A, F, p_sfpu::LREG7, A, 0)
#define SPLIT_Y(A, N) TTI_SFPSETEXP(0, A, N, 0)
#define SPLIT_RND(N) TTI_SFP_STOCH_RND(sfpi::SFPSTOCHRND_RND_EVEN, 0, N, N, N, sfpi::SFPSTOCHRND_MOD1_FP32_TO_FP16B)
#define SPLIT_STORE(N) TTI_SFPSTORE(N, InstrModLoadStore::DEFAULT, ADDR_MOD_6, 0)

template <uint32_t scale_bf16>
inline void split_face() {
    split_consts_lregs();
    constexpr int BODY_LEN = 11;
    TTI_REPLAY(0, BODY_LEN, 1, 1);
    SPLIT_LOAD(p_sfpu::LREG0, 0);
    SPLIT_MULI(p_sfpu::LREG0);
    SPLIT_T(p_sfpu::LREG0);
    SPLIT_N(p_sfpu::LREG0, p_sfpu::LREG1);
    SPLIT_NF(p_sfpu::LREG1, p_sfpu::LREG2);
    SPLIT_F(p_sfpu::LREG0, p_sfpu::LREG2);
    SPLIT_P1(p_sfpu::LREG0, p_sfpu::LREG2);
    SPLIT_P2(p_sfpu::LREG0, p_sfpu::LREG2);
    SPLIT_Y(p_sfpu::LREG0, p_sfpu::LREG1);
    SPLIT_RND(p_sfpu::LREG1);
    SPLIT_STORE(p_sfpu::LREG1);
#pragma GCC unroll 8
    for (int i = 1; i < 8; i++) {
        TTI_REPLAY(0, BODY_LEN, 0, 0);
    }
}

// Two rows per body: row A in (L0, L1, L2) at dest +0, row B in (L3, L4, L5) at dest +2. A's store advances the
// dest base by 2 (ADDR_MOD_6), so B stores at +0 of the new base and advances again: net +4 per body.
// GUARD: 0 = none (t <= 0 / -inf lanes are WRONG: the u16 conversion of a negative t is not 0),
//        1 = t = max(t, 0) via SFPSWAP against LCONST_0 (min half discarded into the read-only constant),
//        2 = baseline-style mask: n &= (t > 0) via SFPGT + SFPAND.
#define SPLIT_CLAMP(A) TTI_SFPSWAP(0, A, p_sfpu::LCONST_0, sfpi::SFPSWAP_MOD1_VEC_MIN_MAX)
#define SPLIT_GT(A) TTI_SFPGT(0, p_sfpu::LCONST_0, A, 8)
#define SPLIT_AND(A, N) TTI_SFPAND(0, A, N, 0)
template <uint32_t scale_bf16, bool FOLD, int GUARD>
inline void split2_face() {
    split_consts_lregs();
    constexpr int BODY_LEN = (FOLD ? 20 : 22) + (GUARD == 1 ? 2 : 0) + (GUARD == 2 ? 4 : 0);
    TTI_REPLAY(0, BODY_LEN, 1, 1);
    SPLIT_LOAD(p_sfpu::LREG0, 0);
    SPLIT_LOAD(p_sfpu::LREG3, 2);
    if constexpr (!FOLD) {
        SPLIT_MULI(p_sfpu::LREG0);
        SPLIT_MULI(p_sfpu::LREG3);
    }
    SPLIT_T(p_sfpu::LREG0);
    SPLIT_T(p_sfpu::LREG3);
    if constexpr (GUARD == 1) {
        SPLIT_CLAMP(p_sfpu::LREG0);
        SPLIT_CLAMP(p_sfpu::LREG3);
    }
    SPLIT_N(p_sfpu::LREG0, p_sfpu::LREG1);
    SPLIT_N(p_sfpu::LREG3, p_sfpu::LREG4);
    SPLIT_NF(p_sfpu::LREG1, p_sfpu::LREG2);
    SPLIT_NF(p_sfpu::LREG4, p_sfpu::LREG5);
    SPLIT_F(p_sfpu::LREG0, p_sfpu::LREG2);
    SPLIT_F(p_sfpu::LREG3, p_sfpu::LREG5);
    if constexpr (GUARD == 2) {
        SPLIT_GT(p_sfpu::LREG0);
        SPLIT_GT(p_sfpu::LREG3);
        SPLIT_AND(p_sfpu::LREG0, p_sfpu::LREG1);
        SPLIT_AND(p_sfpu::LREG3, p_sfpu::LREG4);
    }
    SPLIT_P1(p_sfpu::LREG0, p_sfpu::LREG2);
    SPLIT_P1(p_sfpu::LREG3, p_sfpu::LREG5);
    SPLIT_P2(p_sfpu::LREG0, p_sfpu::LREG2);
    SPLIT_P2(p_sfpu::LREG3, p_sfpu::LREG5);
    SPLIT_Y(p_sfpu::LREG0, p_sfpu::LREG1);
    SPLIT_Y(p_sfpu::LREG3, p_sfpu::LREG4);
    SPLIT_RND(p_sfpu::LREG1);
    SPLIT_RND(p_sfpu::LREG4);
    SPLIT_STORE(p_sfpu::LREG1);
    SPLIT_STORE(p_sfpu::LREG4);
#pragma GCC unroll 4
    for (int i = 1; i < 4; i++) {
        TTI_REPLAY(0, BODY_LEN, 0, 0);
    }
}

// ---- variant 10: SFPLOADMACRO-pipelined fold2c (8 issue slots / row, 2 register sets, whole tile streamed) -------
// Per row: LM_A = [load x; MAD t = K*x + 127 (macro); SWAP t = max(t, 0) (macro)], explicit RND_n, CAST, MAD phi,
// MAD p1, MAD p2, SETEXP, LM_B = [dummy load into phi; L16 = bf16_rne(y) (macro); store L16 (macro)].
// Rows are 8 slots apart; the schedule (kernels/lmexp_stream.h, generated by gen.py) keeps every residue mod 8 on one
// issued instruction and every sub-unit (load/simple/MAD/round/store) on at most one op per slot.
#include "lmexp_stream.h"
constexpr uint32_t LM_DM = 0, LM_DS = 2, LM_DR = 0, LM_DT = 2;
template <uint32_t scale_bf16>
inline void lm_init() {
    // Instruction templates (backdoor install: VD = 12 + slot; nothing executes).
    TTI_SFPMAD(p_sfpu::LREG12, 0, p_sfpu::LREG6, 12, 0);  // T0: VD = L12 * VD + L6
    TTI_SFPSWAP(0, p_sfpu::LCONST_0, 13, 9);              // T1: VD = max(VD, 0)
    TTI_SFP_STOCH_RND(sfpi::SFPSTOCHRND_RND_EVEN, 0, 0, p_sfpu::LREG1, 14, sfpi::SFPSTOCHRND_MOD1_FP32_TO_FP16B);  // T2
    TTI_SFP_STOCH_RND(sfpi::SFPSTOCHRND_RND_EVEN, 0, 0, p_sfpu::LREG4, 15, sfpi::SFPSTOCHRND_MOD1_FP32_TO_FP16B);  // T3
    {  // seq 0 (LM_A): MAD T0 @dM (VB := VD), SIMPLE T1 @dS (VC kept = LCONST_0)
        constexpr uint32_t simple_bits = 0x80 | (LM_DS << 3) | (4 + 1);
        constexpr uint32_t mad_bits = 0x80 | (LM_DM << 3) | (4 + 0);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mad_bits << 8) | simple_bits);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, 0);
        TTI_SFPCONFIG(0, 4 + 0, 0);
    }
    {  // seq 1 (LM_B, set X): ROUND T2 @dR -> L16 (VC kept = L1), STORE L16 @dT
        constexpr uint32_t round_bits = 0x80 | 0x40 | (LM_DR << 3) | (4 + 2);
        constexpr uint32_t store_bits = 0x40 | (LM_DT << 3) | 3;
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, 0);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store_bits << 8) | round_bits);
        TTI_SFPCONFIG(0, 4 + 1, 0);
    }
    {  // seq 2 (LM_B, set Y): ROUND T3 @dR -> L16 (VC kept = L4), STORE L16 @dT
        constexpr uint32_t round_bits = 0x80 | 0x40 | (LM_DR << 3) | (4 + 3);
        constexpr uint32_t store_bits = 0x40 | (LM_DT << 3) | 3;
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, 0);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store_bits << 8) | round_bits);
        TTI_SFPCONFIG(0, 4 + 2, 0);
    }
    // Misc: UsesLoadMod0ForStore = all, UnitDelayKind = WaitForElapsedInstructions for all units.
    TTI_SFPCONFIG(0xff0, 8, 1);
    split_init<scale_bf16, true>();  // L12 = s/ln2, L13 = c2*2^46, L14 = c1*2^23 (after the template installs)
}

inline void lm_tile(uint32_t dst_index) {
    _llk_math_eltwise_sfpu_start_(dst_index);
    split_consts_lregs();  // L6 = 127, L7 = c0
    LMEXP_PROLOGUE();
    TTI_REPLAY(0, LMEXP_BODY_LEN, 1, 1);
    LMEXP_BODY();
#pragma GCC unroll 0
    for (int k = 1; k < LMEXP_NBODIES; ++k) {
        TTI_REPLAY(0, LMEXP_BODY_LEN, 0, 0);
    }
    LMEXP_EPILOGUE();
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
    _llk_math_eltwise_sfpu_done_();
}
}  // namespace exp_bench
#endif

// ---- the ScaledExp functors (UnaryOp: static init() + exec_impl(slot)) ------------------------------------------
template <uint32_t scale_bf16>
struct ScaledExpBase : UnaryOp<ScaledExpBase<scale_bf16>, Dst::D0> {
    static ALWI void init() { exp_tile_init<false>(); }
    static ALWI void exec_impl(uint32_t slot_offset) {
        exp_tile<false, true>(slot_offset, VectorMode::RC, static_cast<uint16_t>(scale_bf16));
    }
};

template <uint32_t scale_bf16>
struct ScaledExpTrim : UnaryOp<ScaledExpTrim<scale_bf16>, Dst::D0> {
    static ALWI void init() { exp_tile_init<false>(); }
    static ALWI void exec_impl(uint32_t slot_offset) {
        MATH((_llk_math_eltwise_unary_sfpu_params_(exp_bench::trim_face<scale_bf16>, slot_offset, VectorMode::RC)));
    }
};

template <uint32_t scale_bf16>
struct ScaledExpSplit : UnaryOp<ScaledExpSplit<scale_bf16>, Dst::D0> {
    static ALWI void init() {
        exp_tile_init<false>();
        MATH((exp_bench::split_init<scale_bf16, false>()));
    }
    static ALWI void exec_impl(uint32_t slot_offset) {
        MATH((_llk_math_eltwise_unary_sfpu_params_(exp_bench::split_face<scale_bf16>, slot_offset, VectorMode::RC)));
    }
};

template <uint32_t scale_bf16, bool FOLD, int GUARD>
struct ScaledExpSplit2 : UnaryOp<ScaledExpSplit2<scale_bf16, FOLD, GUARD>, Dst::D0> {
    static ALWI void init() {
        exp_tile_init<false>();
        MATH((exp_bench::split_init<scale_bf16, FOLD>()));
    }
    static ALWI void exec_impl(uint32_t slot_offset) {
        MATH((_llk_math_eltwise_unary_sfpu_params_(
            exp_bench::split2_face<scale_bf16, FOLD, GUARD>, slot_offset, VectorMode::RC)));
    }
};

template <uint32_t scale_bf16>
struct ScaledExpLM : UnaryOp<ScaledExpLM<scale_bf16>, Dst::D0> {
    static ALWI void init() {
        exp_tile_init<false>();
        MATH((exp_bench::lm_init<scale_bf16>()));
    }
    static ALWI void exec_impl(uint32_t slot_offset) { MATH((exp_bench::lm_tile(slot_offset))); }
};

template <uint32_t variant, uint32_t scale_bf16>
struct Pick;
template <uint32_t s>
struct Pick<10, s> {
    using T = ScaledExpLM<s>;
};
template <uint32_t s>
struct Pick<11, s> {
    using T = ::ScaledExp<s>;
};
template <uint32_t s>
struct Pick<0, s> {
    using T = ScaledExpBase<s>;
};
template <uint32_t s>
struct Pick<1, s> {
    using T = ScaledExpTrim<s>;
};
template <uint32_t s>
struct Pick<2, s> {
    using T = ScaledExpSplit<s>;
};
template <uint32_t s>
struct Pick<3, s> {
    using T = ScaledExpSplit2<s, false, 0>;
};
template <uint32_t s>
struct Pick<4, s> {
    using T = ScaledExpSplit2<s, true, 0>;
};
template <uint32_t s>
struct Pick<6, s> {
    using T = ScaledExpSplit2<s, false, 1>;
};
template <uint32_t s>
struct Pick<7, s> {
    using T = ScaledExpSplit2<s, true, 1>;
};
template <uint32_t s>
struct Pick<8, s> {
    using T = ScaledExpSplit2<s, false, 2>;
};
template <uint32_t s>
struct Pick<9, s> {
    using T = ScaledExpSplit2<s, true, 2>;
};

ALWI void inplace_fence(uint32_t cb, uint32_t n) {
    cb_pop_front(cb, n);
    cb_reserve_back(cb, n);
    cb_push_back(cb, n);
    cb_wait_front(cb, n);
}

void kernel_main() {
    constexpr uint32_t VARIANT = get_compile_time_arg_val(0);
    constexpr uint32_t MODE = get_compile_time_arg_val(1);
    constexpr uint32_t Sq = get_compile_time_arg_val(2);
    constexpr uint32_t Sk = get_compile_time_arg_val(3);
    constexpr uint32_t scale_bits = get_compile_time_arg_val(4);
    constexpr bool NOEXP = VARIANT == 5;
    using Exp = typename Pick<(NOEXP ? 0 : VARIANT), scale_bits>::T;

    constexpr uint32_t cb_scores = 0, cb_max_blk = 1, cb_max_acc = 2, cb_out = 16;

    if constexpr (MODE == 0) {
        compute_kernel_hw_startup(cb_scores, cb_max_blk, cb_scores);
        // The score shard is resident; publish it from the compute kernel's own pack side (as the op's matmul
        // does) so the in-place fence below sees consistent CB counters.
        cb_reserve_back(cb_scores, Sq * Sk);
        cb_push_back(cb_scores, Sq * Sk);
        cb_wait_front(cb_scores, Sq * Sk);
        constexpr auto S = input(
            cb_scores,
            WaitPolicy::None,
            PopPolicy::None,
            InputTileMapping::Block,
            DataFormatReconfig::Enabled,
            TileAddressing::Strided);
        constexpr auto M =
            input(cb_max_blk, BroadcastDim::Col, WaitPolicy::Upfront, PopPolicy::None, InputTileMapping::Col);
        constexpr auto O = output(
            cb_scores, ReservePolicy::None, PushPolicy::None, DataFormatReconfig::Enabled, TileAddressing::Strided);
        if constexpr (NOEXP) {
            eltwise_chain(
                IterationShape::grid(Sq, Sk),
                BinaryFpu<BinaryFpuOp::Sub, S, M>{StridedTileRange{0, Sk}},
                PackTile<O>{StridedTileRange{0, Sk}});
        } else {
            eltwise_chain(
                IterationShape::grid(Sq, Sk),
                BinaryFpu<BinaryFpuOp::Sub, S, M>{StridedTileRange{0, Sk}},
                Exp{},
                PackTile<O>{StridedTileRange{0, Sk}});
        }
        inplace_fence(cb_scores, Sq * Sk);
    } else {
        compute_kernel_hw_startup(cb_max_acc, cb_max_blk, cb_out);
        constexpr auto A = input(cb_max_acc);
        constexpr auto B = input(cb_max_blk, WaitPolicy::Upfront, PopPolicy::None, InputTileMapping::Block);
        if constexpr (NOEXP) {
            eltwise_chain(IterationShape::tiles(Sq), BinaryFpu<BinaryFpuOp::Sub, A, B>{}, PackTile<output(cb_out)>{});
        } else {
            eltwise_chain(
                IterationShape::tiles(Sq), BinaryFpu<BinaryFpuOp::Sub, A, B>{}, Exp{}, PackTile<output(cb_out)>{});
        }
    }
}
