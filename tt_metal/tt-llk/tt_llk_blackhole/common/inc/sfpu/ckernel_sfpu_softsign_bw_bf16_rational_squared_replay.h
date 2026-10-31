// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// The certified body is 18 slots, or 20 with the exponent-FF NaN class tail. Fusing the gradient
// multiply (FUSE_GRAD_SINGLE_ROUNDING) adds 6 more and retires the separate gradient pass: the two
// products, their gap, and the flushed-reciprocal guard (setcc, move, gap, endcc): 24 slots, or 26.
// A fused body thus records past slot 15, into the slots TT-Metal's LLK gives the math thread's FPU
// replays (replay_buf_offset = 16 in cmath_common.h; matmul, reduce and transpose_dest record there in
// their own init). The body records its slots again on every tile, and the unary-backward program that
// calls it (copy_tile, this call, pack_tile) runs no FPU replay; a later program's FPU init records its
// own slots before it replays them.
#include <cstdint>
constexpr std::uint32_t kSquaredAbsDenominatorCertifiedSlots = 18u;
constexpr std::uint32_t kSquaredAbsDenominatorBodySlots      = kSquaredAbsDenominatorCertifiedSlots + 6u;
constexpr bool kSquaredAbsDenominatorTtiReplay =
    (TT_TARGET_BH_BF16_SQUARED_ABS_DENOMINATOR_TTI_BODY_SLOTS == kSquaredAbsDenominatorCertifiedSlots) && tti_replay_fits(kSquaredAbsDenominatorBodySlots);

// Arm the existing BH load-macro RNE/store tail for this anonymous replay.
// T1 captures the same FP32->BF16 RNE used by the plain body; macro sequence
// zero fires it at issue+1 and the unchanged-format store at issue+3.  The
// template issue is backdoor-only; no programmable constant beyond the
// reciprocal helper's immutable 2.0 (Prgm0/L12) is consumed.
inline void squared_abs_denominator_tti_lm_init()
{
#if !defined(DISABLE_SFPLOADMACRO)
    TTI_SFP_STOCH_RND(SFPSTOCHRND_RND_EVEN, 0, 0, ckernel::p_sfpu::LREG0, 13, SFPSTOCHRND_MOD1_FP32_TO_FP16B);
    TTI_SFPLOADI(0, SFPLOADI_MOD0_LOWER, 0x0000); // mad | simple idle
    TTI_SFPLOADI(0, SFPLOADI_MOD0_UPPER, 0x1385); // store | round
    TTI_SFPCONFIG(0, 4, 0);
    TTI_SFPCONFIG(0x110, 8, 1);
#endif
}

inline void squared_abs_denominator_tti_row();

inline void squared_abs_denominator_tti_replay_tile()
{
    constexpr std::uint32_t kBodySlots = kSquaredAbsDenominatorBodySlots;
    static_assert(kSquaredAbsDenominatorTtiReplay);
    // Each row stores through the hold mode; the DEST counter steps after it, outside the record. A
    // load-macro's store keeps the row its issue computed.
#if defined(DISABLE_SFPLOADMACRO)
    // The same rows without the load macro or the replay buffer: identical results.
#pragma GCC unroll 1
    for (int row = 0; row < 32; ++row)
    {
        squared_abs_denominator_tti_row();
        TTI_INCRWC(0, 2, 0, 0);
    }
#endif
#if !defined(DISABLE_SFPLOADMACRO)
    TT_REPLAY_ADDRMOD_ENTER();
    ckernel::load_replay_buf<ckernel::Exec>(0, (kBodySlots), [&] {  // record + execute element 0
        squared_abs_denominator_tti_row();
        TTI_INCRWC(0, 2, 0, 0);
    });
#pragma GCC unroll 32
    for (int row = 1; row < 32; ++row)
    {
        TTI_REPLAY(0, kBodySlots, 0, 0);
        TTI_INCRWC(0, 2, 0, 0);
    }
#endif
    TT_REPLAY_TILE_EPILOGUE();
}

// One row: the body the replay records.
inline void squared_abs_denominator_tti_row()
{
    TTI_SFPLOAD(ckernel::p_sfpu::LREG0, 0, TT_ADDRMOD_DST_HOLD, 0); // x
    TTI_SFPNOP;                                                     // conservative replay load-consumer gap
    TTI_SFPSETSGN(0, ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG1,
                  1); // magnitude = |x| (no clamp: certificate-proved unnecessary)
    TTI_SFPMAD(ckernel::p_sfpu::LCONST_1, ckernel::p_sfpu::LREG1, ckernel::p_sfpu::LCONST_1, ckernel::p_sfpu::LREG7,
               0);                                                   // den = magnitude + 1.0
    TTI_SFPLOAD(ckernel::p_sfpu::LREG5, 0, TT_ADDRMOD_DST_HOLD, 64); // grad (dest tile + 1), in the den gap
    TTI_SFPARECIP(0, 7, 2, 0);                                       // reciprocal seed -> L2
    TTI_SFPNOP;                                                      // reciprocal-seed hazard gap
    TTI_SFPMAD(ckernel::p_sfpu::LREG7, ckernel::p_sfpu::LREG2, ckernel::p_sfpu::LREG12, ckernel::p_sfpu::LREG3,
               2); // t = den*r0 - 2.0
    TTI_SFPNOP;    // first-residual hazard gap
    TTI_SFPMAD(ckernel::p_sfpu::LREG3, ckernel::p_sfpu::LREG2, ckernel::p_sfpu::LCONST_0, ckernel::p_sfpu::LREG4,
               3); // y1 = r0*-t - 0
    TTI_SFPNOP;
    TTI_SFPSETCC(0, 3, 0, 0); // CC: first residual < 0 (NaN guard)
    TTI_SFPMAD(ckernel::p_sfpu::LREG7, ckernel::p_sfpu::LREG4, ckernel::p_sfpu::LREG12, ckernel::p_sfpu::LREG1,
               1); // t2 = den*y1 - 2.0 (predicated)
    TTI_SFPNOP;
    TTI_SFPMAD(ckernel::p_sfpu::LREG4, ckernel::p_sfpu::LREG1, ckernel::p_sfpu::LCONST_0, ckernel::p_sfpu::LREG2,
               2);            // reciprocal = y1*-t2 - 0 (predicated)
    TTI_SFPENCC(3, 0, 0, 10); // CC-balanced replay body
    // (reciprocal * grad) * reciprocal is formed in FP32 and rounded to BF16 once, by the load-macro
    // tail below; scaling before squaring keeps a normal product out of the FTZ squared reciprocal.
    TTI_SFPMUL(ckernel::p_sfpu::LREG2, ckernel::p_sfpu::LREG5, ckernel::p_sfpu::LCONST_0, ckernel::p_sfpu::LREG6,
               0); // reciprocal * grad
    TTI_SFPNOP;
    // Where the reciprocal flushed to 0 (|x| from 2^126, or an infinite or NaN x) an infinite grad would
    // make 0 * inf = NaN: reciprocal * grad is 0 there for any grad, as the composite's flushed factor is.
    TTI_SFPSETCC(0, ckernel::p_sfpu::LREG2, 0, sfpi::SFPSETCC_MOD1_LREG_EQ0); // CC: reciprocal == 0
    TTI_SFPMOV(0, ckernel::p_sfpu::LCONST_0, ckernel::p_sfpu::LREG6, 0);      // reciprocal * grad = +0
    TTI_SFPNOP;                                                               // move-consumer gap
    TTI_SFPENCC(3, 0, 0, 10);                                                 // CC-balanced replay body
    TTI_SFPMUL(ckernel::p_sfpu::LREG6, ckernel::p_sfpu::LREG2, ckernel::p_sfpu::LCONST_0, ckernel::p_sfpu::LREG0,
               0); // grad * reciprocal * reciprocal
#if defined(DISABLE_SFPLOADMACRO)
    TTI_SFPNOP; // result-MUL hazard gap
    TTI_SFP_STOCH_RND(SFPSTOCHRND_RND_EVEN, 0, ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG4, SFPSTOCHRND_MOD1_FP32_TO_FP16B);
    TTI_SFPSTORE(ckernel::p_sfpu::LREG4, 0, TT_ADDRMOD_DST_HOLD, 0);
#endif
#if !defined(DISABLE_SFPLOADMACRO)
    // The load-macro itself is the result-MUL hazard gap.  Captured T1 RNE
    // fires at issue+1, then the fixed store writes the same destination
    // row at issue+3. L4 is dead after the second Newton update.
    TTI_SFPLOADMACRO((0 << 2) | (ckernel::p_sfpu::LREG4 & 3), 0, TT_ADDRMOD_DST_HOLD, (ckernel::p_sfpu::LREG4 >> 2));
#endif
}
