// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>

#include "ckernel_addrmod.h"
#include "ckernel_instr_params.h"
#include "ckernel_ops.h"
// Inverted dependency, kept on purpose: the sfpi leaves below reuse the metal exp helpers
// (_float_to_int32_for_exp_21f_, PolynomialEvaluator), which live one layer up in
// hw/ckernels/blackhole/metal/llk_api/llk_sfpu/. The TTI kernel further down needs nothing from it.
#include "ckernel_sfpu_exp.h"
#include "sfpi.h"
#include "sfpu/ckernel_sfpu_load_config.h"

namespace ckernel
{
namespace sfpu
{

//**************************************************************
// sfpi leaves (vFloat -> vFloat), used by the compute API on the PACK thread
// (api/compute/experimental/sdpa.h: non_approx_exp_mul_prev), where the two inputs
// arrive in LREGs rather than in DEST and no dst_reg walk exists to replay.
//**************************************************************

template <bool is_fp32_dest_acc_en>
sfpi_inline sfpi::vFloat _sfpu_exp_21f_bf16_lower_clamp_only_(sfpi::vFloat val)
{
    constexpr float ONE_LN2 = 1.4426950216293334961f;
    sfpi::vFloat xlog2      = (val * ONE_LN2 + 127.f);

    // Lower clamp only (xlog2 >= 0). Upper clamp is dead when val <= 0 (see file header).
    // One SFPSWAP via sfpi::max, unlike sfpi::vec_min_max's swap through a second operand.
    xlog2 = sfpi::max(xlog2, 0.0f);

    sfpi::vInt z = _float_to_int32_for_exp_21f_(xlog2);

    sfpi::vInt exponential_part = exexp(sfpi::as<sfpi::vFloat>(z), sfpi::ExponentMode::Biased);
    sfpi::vMag fractional_part  = sfpi::exman(sfpi::as<sfpi::vFloat>(z));

    sfpi::vFloat frac = sfpi::convert<sfpi::vFloat>(fractional_part, sfpi::RoundMode::NearestAway);
    frac              = PolynomialEvaluator::eval(frac, 1.0017248f, 7.839635491371155e-08f, 4.791750143340323e-15f);

    sfpi::vFloat y = sfpi::setexp(frac, exponential_part);

    if constexpr (!is_fp32_dest_acc_en)
    {
        y = sfpi::convert<sfpi::vFloat16b>(y, sfpi::RoundMode::NearestAway);
    }

    return y;
}

template <bool SCALE_EN, bool is_fp32_dest_acc_en>
sfpi_inline sfpi::vFloat _ckernel_sfpu_exp_accurate_upper_unclamped_(sfpi::vFloat val, const std::uint32_t exp_base_scale_factor)
{
    static_assert(!is_fp32_dest_acc_en, "upper-unclamped exp variant implemented for bf16 dest only");
    if constexpr (SCALE_EN)
    {
        val = val * sfpi::sFloat16b(exp_base_scale_factor);
    }
    return _sfpu_exp_21f_bf16_lower_clamp_only_<is_fp32_dest_acc_en>(val);
}

//**************************************************************
// TTI kernel (DEST face walk, replayed): the upper-unclamped twin of the shared
// _sfpu_exp_21f_bf16_tti_ -- that kernel's body verbatim minus its SFPLOADI(255) + SFPSWAP
// upper clamp, which is dead for val <= 0. Same algorithm and constants as the sfpi leaf above.
//**************************************************************

/**
 * @brief One-time setup for _calculate_sdpa_exp_unclamped_.
 *
 * Programs what the replayed body reads but never writes:
 *   - ADDR_MOD_6: dest += 2 on the body's SFPSTORE, so each replay lands on the next 4x8 slot
 *     without a per-row INCRWC.
 *   - LREG12 = 1/ln2 (0x3fb8aa3b), LREG13 = c2 = 4.791750143340323e-15f (0x27aca418): the two
 *     iteration-invariant constants that do not fit the four free LREGs. Same registers and
 *     values as exp_init<false> programs for the shared clamped TTI exp, so the two kernels
 *     can share a thread's SFPU state.
 *
 * Call once after the invariant SFPU init (SFPCONFIG + ADDR_MOD_7); re-call after any op init
 * that reprograms LREG12/13 (exp_init<true>, sfpu_reciprocal_init, log_init, ...) or ADDR_MOD_6.
 */
inline void _init_sdpa_exp_unclamped_()
{
    addr_mod_t {
        .srca = {.incr = 0},
        .srcb = {.incr = 0},
        .dest = {.incr = 2},
    }
        .set(ADDR_MOD_6);

    _sfpu_load_config32_(p_sfpu::LREG12, 0x3fb8, 0xaa3b); // 1/ln2 = 1.4426950216293334961f
    _sfpu_load_config32_(p_sfpu::LREG13, 0x27ac, 0xa418); // c2    = 4.791750143340323e-15f
}

/**
 * @brief exp(val * scale) over ITERATIONS consecutive 4x8 DEST slots, bf16 DEST, upper clamp removed.
 *
 * Contract: identical to _sfpu_exp_21f_bf16_lower_clamp_only_ for every bf16 input --
 * exp() over val*scale in [-88.03, 88.7] (~3 fp32 ULP before the bf16 rounding), +0 below the
 * surviving lower clamp (val*scale <= -88.03, including -Inf), and the SAME unclamped
 * wrap above 88.7 (the float->int step wraps; exp overflows bf16 there anyway). Callers are
 * expected to feed val <= 0 (post max-subtraction logits); the upper clamp is dead there.
 *
 * Register use inside the recorded body:
 *   LREG0  val -> scaled val -> int part (float encoding of 2^floor) -> result
 *   LREG1  exexp(xlog2) -> frac (int) -> frac (float) -> polynomial result
 *   LREG2  polynomial accumulator
 *   LREG3  xlog2 = val/ln2 + 127 -> lower-clamp mask (xlog2 > 0)
 *   LREG5  127.0 (bf16), LREG6 c1 = 7.839635491371155e-08f, LREG7 c0 = 1.0017248f -- loaded once per call
 *   LREG12 1/ln2, LREG13 c2 -- programmed by _init_sdpa_exp_unclamped_
 *
 * Scheduling (BlackholeA0 ISA, SFPMAD.md): SFPMAD results take 2 cycles and the scoreboard
 * stalls a dependent reader on the next cycle -- except for the listed errata consumers, among
 * them every min/max form of SFPSWAP. That is why the lower clamp is the shared TTI kernel's
 * SFPGT mask + SFPAND (both 1-cycle, scoreboard-visible, and slotted into the two SFPMAD
 * latency windows so nothing stalls) and not the sfpi leaf's max(xlog2, 0) SFPSWAP, which would
 * need an explicit gap after the xlog2 SFPMAD and forces a one-cycle bubble of its own.
 *
 * The body is recorded into replay slot 0 on the first iteration and replayed ITERATIONS-1
 * times: 15 SFPU instructions per slot with SCALE_EN (14 without), one RISC issue per replay,
 * against 27 (26) for the compiler-scheduled sfpi leaf in a dst_reg loop.
 *
 * @tparam SCALE_EN    multiply the input by exp_base_scale_factor (bf16 immediate) first
 * @tparam ITERATIONS  4x8 slots per call; 8 covers a 16x16 face
 * @param exp_base_scale_factor  scale as a raw bf16 bit pattern; ignored when !SCALE_EN
 * @note Requires _init_sdpa_exp_unclamped_ (ADDR_MOD_6, LREG12/13) and clobbers replay slot 0.
 * @note bf16 DEST only: the fp32->bf16 round-to-nearest before the store is unconditional.
 */
template <bool SCALE_EN, int ITERATIONS>
inline void _calculate_sdpa_exp_unclamped_(const std::uint16_t exp_base_scale_factor)
{
    // Iteration-invariant constants that fit the free LREGs: loaded once, outside the replay body.
    TTI_SFPLOADI(p_sfpu::LREG5, sfpi::SFPLOADI_MOD0_FLOATB, 0x42fe); // 127.0
    TTI_SFPLOADI(p_sfpu::LREG6, sfpi::SFPLOADI_MOD0_UPPER, 0x33a8);  // c1 = 7.839635491371155e-08f
    TTI_SFPLOADI(p_sfpu::LREG6, sfpi::SFPLOADI_MOD0_LOWER, 0x5ada);
    TTI_SFPLOADI(p_sfpu::LREG7, sfpi::SFPLOADI_MOD0_UPPER, 0x3f80); // c0 = 1.0017248f, exact fp32
    TTI_SFPLOADI(p_sfpu::LREG7, sfpi::SFPLOADI_MOD0_LOWER, 0x3885); //   (the sfpi leaf's value, not the fp16 0x3c02)

    // Instructions between the record below and the replay loop; must match exactly.
    //   SFPLOAD, SFPMAD, SFPEXEXP, SFPEXMAN, SFPSHFT, SFPEXMAN, SFPCAST,
    //   SFPMAD, SFPGT, SFPMAD, SFPAND, SFPSETEXP, SFP_STOCH_RND, SFPSTORE     = 14
    //   + SFPMULI when SCALE_EN                                               = 15
    constexpr std::uint32_t BODY_LEN = 14 + (SCALE_EN ? 1 : 0);

    TTI_REPLAY(0, BODY_LEN, 1, 1); // record slot 0 while executing

    // val = dst_reg[0]
    TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_7, 0);

    if constexpr (SCALE_EN)
    {
        // val *= scale (bf16 immediate), as sfpi::sFloat16b(scale) * val lowers. The TT_ form
        // builds the instruction word on the RISC (one li+sw per call, recorded once) so a scale
        // that is not a compile-time constant still compiles; TTI_SFPMULI needs an "n" immediate.
        TT_SFPMULI(exp_base_scale_factor, p_sfpu::LREG0, 0);
    }

    // xlog2 = val * (1/ln2) + 127.0, kept in LREG3 for the mask below. No upper clamp: the
    // shared kernel's SFPLOADI(255) + SFPSWAP go here; they are dead for val <= 0.
    TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG12, p_sfpu::LREG5, p_sfpu::LREG3, 0);

    // _float_to_int32_for_exp_21f_: z = exman8(xlog2) << exexp(xlog2), i.e. xlog2 * 2^23 as an integer
    // whose float encoding is 2^floor(val/ln2) with the fractional part in the mantissa bits.
    // SFPEXEXP reads the SFPMAD result one cycle later; the scoreboard sees that read and stalls.
    TTI_SFPEXEXP(0, p_sfpu::LREG3, p_sfpu::LREG1, sfpi::SFPEXEXP_MOD1_DEBIAS); // LREG1 = exexp(xlog2)
    TTI_SFPEXMAN(0, p_sfpu::LREG3, p_sfpu::LREG0, sfpi::SFPEXMAN_MOD1_PAD8);   // LREG0 = exman8(xlog2)
    TTI_SFPSHFT(0, p_sfpu::LREG1, p_sfpu::LREG0, 0);                           // LREG0 <<= LREG1 (z)

    // frac = float(exman9(z)), round to nearest even
    TTI_SFPEXMAN(0, p_sfpu::LREG0, p_sfpu::LREG1, sfpi::SFPEXMAN_MOD1_PAD9);
    TTI_SFPCAST(p_sfpu::LREG1, p_sfpu::LREG1, sfpi::SFPCAST_MOD1_SM32_TO_FP32_RNE);

    // Horner: poly = (frac * c2 + c1) * frac + c0, with the lower-clamp mask built in the first
    // SFPMAD's latency window and applied in the second's.
    TTI_SFPMAD(p_sfpu::LREG1, p_sfpu::LREG13, p_sfpu::LREG6, p_sfpu::LREG2, 0);

    // Lower clamp, part 1: LREG3 = (xlog2 > 0) ? ~0 : 0. SFPGT SET_VD writes -1 where VC (0) is
    // smaller than VD (xlog2) in the sign-magnitude total order, so -Inf, -NaN and any xlog2 <= 0
    // give 0; +NaN gives ~0 and flows through unclamped, as it does through the sfpi max().
    constexpr unsigned SFPGT_MOD1_SET_VD = 8;
    TTI_SFPGT(0, p_sfpu::LCONST_0, p_sfpu::LREG3, SFPGT_MOD1_SET_VD);

    TTI_SFPMAD(p_sfpu::LREG2, p_sfpu::LREG1, p_sfpu::LREG7, p_sfpu::LREG1, 0);

    // Lower clamp, part 2: zero the int part where xlog2 <= 0. SETEXP then yields an fp32 with a
    // zero exponent field, which SFP_STOCH_RND (FP32->BF16) turns into +0 -- the same +0 the sfpi
    // leaf reaches through max(xlog2, 0) -> z = 0.
    constexpr unsigned SFPAND_MOD1_USE_VB = 1;
    TTI_SFPAND(p_sfpu::LREG0, p_sfpu::LREG3, p_sfpu::LREG0, SFPAND_MOD1_USE_VB);

    // y = setexp(poly, exponent field of z): recombine 2^floor * 2^frac. SFPSETEXP_MOD1_CPY takes
    // the exponent from lreg_dest's own encoding, so the biased exexp(z) the sfpi leaf extracts
    // explicitly is not needed as a separate instruction.
    TTI_SFPSETEXP(0, p_sfpu::LREG1, p_sfpu::LREG0, sfpi::SFPSETEXP_MOD1_CPY);

    // Round fp32 -> bf16 to nearest before the 16-bit store truncates.
    TTI_SFP_STOCH_RND(sfpi::SFPSTOCHRND_RND_NEAREST, 0, p_sfpu::LREG0, p_sfpu::LREG0, p_sfpu::LREG0, sfpi::SFPSTOCHRND_MOD1_FP32_TO_FP16B);

    // dst_reg[0] = y; dst_reg++ (ADDR_MOD_6: dest += 2)
    TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_6, 0);

#pragma GCC unroll 8
    for (int i = 1; i < ITERATIONS; i++)
    {
        TTI_REPLAY(0, BODY_LEN, 0, 0);
    }
}

} // namespace sfpu
} // namespace ckernel
