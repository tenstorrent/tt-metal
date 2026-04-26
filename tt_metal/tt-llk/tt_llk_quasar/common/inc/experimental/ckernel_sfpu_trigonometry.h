// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
// AI-generated — run_id: 2026-04-24_trigonometry_quasar_6e898d97
#pragma once

#include <cstdint>

#include "ckernel_ops.h"
#include "ckernel_trisc_common.h"
#include "cmath_common.h"
#include "sfpi.h"

namespace ckernel
{
namespace sfpu
{
// fp16b bit patterns (upper 16 bits of the corresponding fp32) used as
// SFPLOADI/SFPMULI/SFPADDI immediates in MOD0_FLOATB mode.
constexpr std::uint32_t FP16B_INV_PI = 0x3EA2; // 1/pi                ~= 0.31831
constexpr std::uint32_t FP16B_PI     = 0x4049; // pi                  ~= 3.1406
constexpr std::uint32_t FP16B_HALF   = 0x3F00; // 0.5
constexpr std::uint32_t FP16B_TWO    = 0x4000; // 2.0

// Round-to-nearest-even bias constants for the float-to-int snap:
// adding +1.5*2^23 pushes the integer part into the mantissa's
// representable range so IEEE RNE resolves ties to even, then subtracting
// 1.5*2^23 leaves the exact nearest integer. Exact in fp16b for |y| < 2^22,
// which covers the sin/cos argument-reduction domain.
constexpr std::uint32_t FP16B_RNE_BIAS_POS = 0x4B40; // +1.5 * 2^23
constexpr std::uint32_t FP16B_RNE_BIAS_NEG = 0xCB40; // -1.5 * 2^23

// SFPSETCC imm12 bit 11 = 1: "treat LREG as FP32" (test sign via the MSB of
// the fp32 bit-pattern, not via two's-complement of int32). Used by the
// sine/cosine parity check to conditionally negate the result.
constexpr std::uint32_t SFPSETCC_IMM_FP32_TEST = 0x800;

// FP32 exponent bias — used to convert the biased (0..255) exponent read by
// SFPEXEXP back to a true integer exponent (-127..128), and as the bias that
// SFPSETEXP's imm12 forces when normalizing a mantissa into [1, 2).
constexpr std::uint32_t FP32_EXP_BIAS = 127;

// Log polynomial (3rd-order minimax on mantissa m in [1, 2)):
//   log(m) ~ ((A*m + B)*m + C)*m + D
// Whole log(x) = (exp_f - 127) * ln2 + log(m), where exp_f is the raw
// biased exponent cast to fp32 and m is setexp(x, 127).
constexpr std::uint32_t FP16B_LOG_EXP_DEBIAS_NEG = 0xC2FE; // -127.0 (added to exp_f to debias)
constexpr std::uint32_t FP16B_LOG_COEFF_A        = 0x3E10; //  0.14081
constexpr std::uint32_t FP16B_LOG_COEFF_B        = 0xBF5E; // -0.86883
constexpr std::uint32_t FP16B_LOG_COEFF_C        = 0x4014; //  2.28790
constexpr std::uint32_t FP16B_LOG_COEFF_D        = 0xBFCB; // -1.58710
constexpr std::uint32_t FP16B_LN2                = 0x3F31; //  ln(2) ~= 0.693147

// Sine Maclaurin coefficients: sin(z) = z + SIN_C3*z^3 + SIN_C5*z^5 + SIN_C7*z^7
//                              [+ SIN_C9*z^9 + SIN_C11*z^11 when !APPROXIMATION_MODE]
// c_k = (-1)^((k-1)/2) / k!
constexpr std::uint32_t FP16B_SIN_C3  = 0xBE2B; // -1/6          ~= -0.16666
constexpr std::uint32_t FP16B_SIN_C5  = 0x3C08; //  1/120        ~=  0.008333
constexpr std::uint32_t FP16B_SIN_C7  = 0xB950; // -1/5040       ~= -1.984e-4
constexpr std::uint32_t FP16B_SIN_C9  = 0x3638; //  1/362880     ~=  2.756e-6
constexpr std::uint32_t FP16B_SIN_C11 = 0xB2D7; // -1/39916800   ~= -2.505e-8

// Cosine Maclaurin coefficients: cos(z) = 1 + COS_C2*z^2 + COS_C4*z^4 + COS_C6*z^6
//                                [+ COS_C8*z^8 + COS_C10*z^10 when !APPROXIMATION_MODE]
// c_k = (-1)^(k/2) / k!
constexpr std::uint32_t FP16B_COS_C2  = 0xBF00; // -1/2          = -0.5
constexpr std::uint32_t FP16B_COS_C4  = 0x3D2B; //  1/24         ~=  0.04166
constexpr std::uint32_t FP16B_COS_C6  = 0xBAB6; // -1/720        ~= -0.001388
constexpr std::uint32_t FP16B_COS_C8  = 0x37D0; //  1/40320      ~=  2.48e-5
constexpr std::uint32_t FP16B_COS_C10 = 0xB494; // -1/3628800    ~= -2.755e-7

// Placeholder init — no LUT/constant pre-loads; every coefficient fits in SFPLOADI/SFPMULI/SFPADDI immediates
inline void _init_trigonometry_()
{
}

// Inline 3rd-order log polynomial: log(base in src) -> dst via scratch LREGs.
// Path avoids SFPCAST mod 2/3 by using biased exponent + fp offset.
// Scratch LREGs used: LREG_exp_f (exponent as float), LREG_x (mantissa in [1,2)), LREG_tmp (polynomial accumulator).
inline void _log_body_inline_(
    const std::uint32_t src, const std::uint32_t dst, const std::uint32_t exp_f, const std::uint32_t x_reg, const std::uint32_t tmp_reg)
{
    // biased exponent (0..255)
    TTI_SFPEXEXP(src, exp_f, sfpi::SFPEXEXP_MOD1_NODEBIAS);
    // int32 (unsigned) -> fp32 RNE
    TTI_SFPCAST(exp_f, exp_f, sfpi::SFPCAST_MOD1_INT32_TO_FP32_RNE);
    // exp_f -= 127.0
    TTI_SFPLOADI(tmp_reg, sfpi::SFPLOADI_MOD0_FLOATB, FP16B_LOG_EXP_DEBIAS_NEG);
    TTI_SFPADD(p_sfpu::LCONST_1, exp_f, tmp_reg, exp_f, 0 /* mod1 */);

    // Normalize mantissa: x = setexp(src, 127) -> x in [1, 2)
    // InstrMod=1 selects exponent from Imm12; Mod=0 reads exp from VD itself (wrong).
    TTI_SFPSETEXP(FP32_EXP_BIAS, src, x_reg, 1 /* mod1: exp from imm12 */);

    // Polynomial: series = ((x*A + B)*x + C)*x + D
    TTI_SFPLOADI(tmp_reg, sfpi::SFPLOADI_MOD0_FLOATB, FP16B_LOG_COEFF_A);
    TTI_SFPLOADI(dst, sfpi::SFPLOADI_MOD0_FLOATB, FP16B_LOG_COEFF_B); // B into running accumulator
    TTI_SFPMAD(tmp_reg, x_reg, dst, dst, 0 /* mod1 */);               // dst = x*A + B
    TTI_SFPLOADI(tmp_reg, sfpi::SFPLOADI_MOD0_FLOATB, FP16B_LOG_COEFF_C);
    TTI_SFPMAD(dst, x_reg, tmp_reg, dst, 0 /* mod1 */); // dst = dst*x + C
    TTI_SFPLOADI(tmp_reg, sfpi::SFPLOADI_MOD0_FLOATB, FP16B_LOG_COEFF_D);
    TTI_SFPMAD(dst, x_reg, tmp_reg, dst, 0 /* mod1 */); // dst = dst*x + D (= series)

    // log(base) = exp_f * ln(2) + series
    TTI_SFPLOADI(tmp_reg, sfpi::SFPLOADI_MOD0_FLOATB, FP16B_LN2);
    TTI_SFPMAD(exp_f, tmp_reg, dst, dst, 0 /* mod1 */); // dst = exp_f * ln2 + series
}

// -----------------------------------------------------------------------------
// sine: y = (1/pi)*x; whole = round(y); frac = y - whole; z = pi*frac;
//        out = z + sum(c_k * z^(2k+1)); negate out if whole is odd.
// -----------------------------------------------------------------------------
template <bool APPROXIMATION_MODE>
inline void _calculate_sine_sfp_rows_()
{
    TTI_SFPLOAD(p_sfpu::LREG0, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /* done */, 0 /* dest_reg */); // x

    TTI_SFPMULI(FP16B_INV_PI, p_sfpu::LREG0, 0 /* mod1 */); // y = (1/pi) * x  (fp16b 0.31831)

    // whole_float = round_to_nearest_even(y): add then subtract 1.5 * 2^23
    TTI_SFPMOV(p_sfpu::LREG0, p_sfpu::LREG1, 0 /* mod1 */);
    TTI_SFPADDI(FP16B_RNE_BIAS_POS, p_sfpu::LREG1, 0 /* mod1 */); // +1.5*2^23
    TTI_SFPADDI(FP16B_RNE_BIAS_NEG, p_sfpu::LREG1, 0 /* mod1 */); // -1.5*2^23 -> rounded to nearest even

    // frac = y - whole_float ; z = pi * frac
    TTI_SFPMAD(p_sfpu::LCONST_neg1, p_sfpu::LREG1, p_sfpu::LREG0, p_sfpu::LREG0, 0 /* mod1 */);
    TTI_SFPMULI(FP16B_PI, p_sfpu::LREG0, 0 /* mod1 */); // z = pi * frac (fp16b pi ~= 3.1406)

    // Maclaurin: out = z; tmp = z; (tmp *= z*z; out += coeff * tmp) per term
    TTI_SFPMOV(p_sfpu::LREG0, p_sfpu::LREG2, 0 /* mod1 */); // out
    TTI_SFPMOV(p_sfpu::LREG0, p_sfpu::LREG3, 0 /* mod1 */); // tmp

    // term SIN_C3 * z^3
    TTI_SFPMUL(p_sfpu::LREG3, p_sfpu::LREG0, p_sfpu::LCONST_0, p_sfpu::LREG3, 0 /* mod1 */);
    TTI_SFPMUL(p_sfpu::LREG3, p_sfpu::LREG0, p_sfpu::LCONST_0, p_sfpu::LREG3, 0 /* mod1 */);
    TTI_SFPLOADI(p_sfpu::LREG4, sfpi::SFPLOADI_MOD0_FLOATB, FP16B_SIN_C3);
    TTI_SFPMAD(p_sfpu::LREG4, p_sfpu::LREG3, p_sfpu::LREG2, p_sfpu::LREG2, 0 /* mod1 */);

    // term SIN_C5 * z^5
    TTI_SFPMUL(p_sfpu::LREG3, p_sfpu::LREG0, p_sfpu::LCONST_0, p_sfpu::LREG3, 0 /* mod1 */);
    TTI_SFPMUL(p_sfpu::LREG3, p_sfpu::LREG0, p_sfpu::LCONST_0, p_sfpu::LREG3, 0 /* mod1 */);
    TTI_SFPLOADI(p_sfpu::LREG4, sfpi::SFPLOADI_MOD0_FLOATB, FP16B_SIN_C5);
    TTI_SFPMAD(p_sfpu::LREG4, p_sfpu::LREG3, p_sfpu::LREG2, p_sfpu::LREG2, 0 /* mod1 */);

    // term SIN_C7 * z^7
    TTI_SFPMUL(p_sfpu::LREG3, p_sfpu::LREG0, p_sfpu::LCONST_0, p_sfpu::LREG3, 0 /* mod1 */);
    TTI_SFPMUL(p_sfpu::LREG3, p_sfpu::LREG0, p_sfpu::LCONST_0, p_sfpu::LREG3, 0 /* mod1 */);
    TTI_SFPLOADI(p_sfpu::LREG4, sfpi::SFPLOADI_MOD0_FLOATB, FP16B_SIN_C7);
    TTI_SFPMAD(p_sfpu::LREG4, p_sfpu::LREG3, p_sfpu::LREG2, p_sfpu::LREG2, 0 /* mod1 */);

    if constexpr (!APPROXIMATION_MODE)
    {
        // term SIN_C9 * z^9
        TTI_SFPMUL(p_sfpu::LREG3, p_sfpu::LREG0, p_sfpu::LCONST_0, p_sfpu::LREG3, 0 /* mod1 */);
        TTI_SFPMUL(p_sfpu::LREG3, p_sfpu::LREG0, p_sfpu::LCONST_0, p_sfpu::LREG3, 0 /* mod1 */);
        TTI_SFPLOADI(p_sfpu::LREG4, sfpi::SFPLOADI_MOD0_FLOATB, FP16B_SIN_C9);
        TTI_SFPMAD(p_sfpu::LREG4, p_sfpu::LREG3, p_sfpu::LREG2, p_sfpu::LREG2, 0 /* mod1 */);

        // term SIN_C11 * z^11
        TTI_SFPMUL(p_sfpu::LREG3, p_sfpu::LREG0, p_sfpu::LCONST_0, p_sfpu::LREG3, 0 /* mod1 */);
        TTI_SFPMUL(p_sfpu::LREG3, p_sfpu::LREG0, p_sfpu::LCONST_0, p_sfpu::LREG3, 0 /* mod1 */);
        TTI_SFPLOADI(p_sfpu::LREG4, sfpi::SFPLOADI_MOD0_FLOATB, FP16B_SIN_C11);
        TTI_SFPMAD(p_sfpu::LREG4, p_sfpu::LREG3, p_sfpu::LREG2, p_sfpu::LREG2, 0 /* mod1 */);
    }

    // Parity check on whole_float in LREG1: parity = whole - 2 * round(whole/2)
    // Use SFPMULI with fp16b(0.5) / fp16b(2.0) to halve/double — avoids DIVP2 imm-sign ambiguity.
    TTI_SFPMOV(p_sfpu::LREG1, p_sfpu::LREG5, 0 /* mod1 */);       // LREG5 = whole
    TTI_SFPMULI(FP16B_HALF, p_sfpu::LREG5, 0 /* mod1 */);         // LREG5 *= 0.5
    TTI_SFPADDI(FP16B_RNE_BIAS_POS, p_sfpu::LREG5, 0 /* mod1 */); // round whole/2 to nearest even
    TTI_SFPADDI(FP16B_RNE_BIAS_NEG, p_sfpu::LREG5, 0 /* mod1 */);
    TTI_SFPMULI(FP16B_TWO, p_sfpu::LREG5, 0 /* mod1 */);                                        // LREG5 *= 2.0
    TTI_SFPMAD(p_sfpu::LCONST_neg1, p_sfpu::LREG5, p_sfpu::LREG1, p_sfpu::LREG5, 0 /* mod1 */); // parity = whole - 2*round(whole/2)

    // Negate out if parity != 0 (treat LREG5 as FP32 via Imm12[11]=1, bit-encoded as 0x800).
    TTI_SFPSETCC(SFPSETCC_IMM_FP32_TEST, p_sfpu::LREG5, sfpi::SFPSETCC_MOD1_LREG_NE0);
    TTI_SFPMOV(p_sfpu::LREG2, p_sfpu::LREG2, 1 /* mod1: flip sign */);
    TTI_SFPENCC(0 /* imm12 */, 0 /* mod1 */);

    TTI_SFPSTORE(p_sfpu::LREG2, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /* done */, 0 /* dest_reg */);
}

template <bool APPROXIMATION_MODE>
inline void _calculate_sine_(const int iterations)
{
#pragma GCC unroll 8
    for (int d = 0; d < iterations; d++)
    {
        _calculate_sine_sfp_rows_<APPROXIMATION_MODE>();
        ckernel::math::_incr_counters_<0x0, 0x0, ckernel::math::SFP_ROWS, 0x0>();
    }
}

// -----------------------------------------------------------------------------
// cosine: same argument reduction; out = 1 + sum(c_k * z^(2k)), negate if odd whole.
// -----------------------------------------------------------------------------
template <bool APPROXIMATION_MODE>
inline void _calculate_cosine_sfp_rows_()
{
    TTI_SFPLOAD(p_sfpu::LREG0, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /* done */, 0 /* dest_reg */); // x

    TTI_SFPMULI(FP16B_INV_PI, p_sfpu::LREG0, 0 /* mod1 */); // y = (1/pi) * x

    TTI_SFPMOV(p_sfpu::LREG0, p_sfpu::LREG1, 0 /* mod1 */);
    TTI_SFPADDI(FP16B_RNE_BIAS_POS, p_sfpu::LREG1, 0 /* mod1 */);
    TTI_SFPADDI(FP16B_RNE_BIAS_NEG, p_sfpu::LREG1, 0 /* mod1 */); // whole_float

    TTI_SFPMAD(p_sfpu::LCONST_neg1, p_sfpu::LREG1, p_sfpu::LREG0, p_sfpu::LREG0, 0 /* mod1 */); // frac
    TTI_SFPMULI(FP16B_PI, p_sfpu::LREG0, 0 /* mod1 */);                                         // z = pi * frac

    // out = 1.0; tmp = z*z
    TTI_SFPMOV(p_sfpu::LCONST_1, p_sfpu::LREG2, 0 /* mod1 */);
    TTI_SFPMUL(p_sfpu::LREG0, p_sfpu::LREG0, p_sfpu::LCONST_0, p_sfpu::LREG3, 0 /* mod1 */); // tmp = z^2

    // term COS_C2 * z^2
    TTI_SFPLOADI(p_sfpu::LREG4, sfpi::SFPLOADI_MOD0_FLOATB, FP16B_COS_C2);
    TTI_SFPMAD(p_sfpu::LREG4, p_sfpu::LREG3, p_sfpu::LREG2, p_sfpu::LREG2, 0 /* mod1 */);

    // term COS_C4 * z^4
    TTI_SFPMUL(p_sfpu::LREG3, p_sfpu::LREG0, p_sfpu::LCONST_0, p_sfpu::LREG3, 0 /* mod1 */);
    TTI_SFPMUL(p_sfpu::LREG3, p_sfpu::LREG0, p_sfpu::LCONST_0, p_sfpu::LREG3, 0 /* mod1 */);
    TTI_SFPLOADI(p_sfpu::LREG4, sfpi::SFPLOADI_MOD0_FLOATB, FP16B_COS_C4);
    TTI_SFPMAD(p_sfpu::LREG4, p_sfpu::LREG3, p_sfpu::LREG2, p_sfpu::LREG2, 0 /* mod1 */);

    // term COS_C6 * z^6
    TTI_SFPMUL(p_sfpu::LREG3, p_sfpu::LREG0, p_sfpu::LCONST_0, p_sfpu::LREG3, 0 /* mod1 */);
    TTI_SFPMUL(p_sfpu::LREG3, p_sfpu::LREG0, p_sfpu::LCONST_0, p_sfpu::LREG3, 0 /* mod1 */);
    TTI_SFPLOADI(p_sfpu::LREG4, sfpi::SFPLOADI_MOD0_FLOATB, FP16B_COS_C6);
    TTI_SFPMAD(p_sfpu::LREG4, p_sfpu::LREG3, p_sfpu::LREG2, p_sfpu::LREG2, 0 /* mod1 */);

    if constexpr (!APPROXIMATION_MODE)
    {
        // term COS_C8 * z^8
        TTI_SFPMUL(p_sfpu::LREG3, p_sfpu::LREG0, p_sfpu::LCONST_0, p_sfpu::LREG3, 0 /* mod1 */);
        TTI_SFPMUL(p_sfpu::LREG3, p_sfpu::LREG0, p_sfpu::LCONST_0, p_sfpu::LREG3, 0 /* mod1 */);
        TTI_SFPLOADI(p_sfpu::LREG4, sfpi::SFPLOADI_MOD0_FLOATB, FP16B_COS_C8);
        TTI_SFPMAD(p_sfpu::LREG4, p_sfpu::LREG3, p_sfpu::LREG2, p_sfpu::LREG2, 0 /* mod1 */);

        // term COS_C10 * z^10
        TTI_SFPMUL(p_sfpu::LREG3, p_sfpu::LREG0, p_sfpu::LCONST_0, p_sfpu::LREG3, 0 /* mod1 */);
        TTI_SFPMUL(p_sfpu::LREG3, p_sfpu::LREG0, p_sfpu::LCONST_0, p_sfpu::LREG3, 0 /* mod1 */);
        TTI_SFPLOADI(p_sfpu::LREG4, sfpi::SFPLOADI_MOD0_FLOATB, FP16B_COS_C10);
        TTI_SFPMAD(p_sfpu::LREG4, p_sfpu::LREG3, p_sfpu::LREG2, p_sfpu::LREG2, 0 /* mod1 */);
    }

    // Parity check on whole_float (LREG1) — use SFPMULI for halve/double (no DIVP2 sign ambiguity).
    TTI_SFPMOV(p_sfpu::LREG1, p_sfpu::LREG5, 0 /* mod1 */);
    TTI_SFPMULI(FP16B_HALF, p_sfpu::LREG5, 0 /* mod1 */); // *= 0.5
    TTI_SFPADDI(FP16B_RNE_BIAS_POS, p_sfpu::LREG5, 0 /* mod1 */);
    TTI_SFPADDI(FP16B_RNE_BIAS_NEG, p_sfpu::LREG5, 0 /* mod1 */);
    TTI_SFPMULI(FP16B_TWO, p_sfpu::LREG5, 0 /* mod1 */); // *= 2.0
    TTI_SFPMAD(p_sfpu::LCONST_neg1, p_sfpu::LREG5, p_sfpu::LREG1, p_sfpu::LREG5, 0 /* mod1 */);

    TTI_SFPSETCC(SFPSETCC_IMM_FP32_TEST, p_sfpu::LREG5, sfpi::SFPSETCC_MOD1_LREG_NE0);
    TTI_SFPMOV(p_sfpu::LREG2, p_sfpu::LREG2, 1 /* mod1: flip sign */);
    TTI_SFPENCC(0 /* imm12 */, 0 /* mod1 */);

    TTI_SFPSTORE(p_sfpu::LREG2, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /* done */, 0 /* dest_reg */);
}

template <bool APPROXIMATION_MODE>
inline void _calculate_cosine_(const int iterations)
{
#pragma GCC unroll 8
    for (int d = 0; d < iterations; d++)
    {
        _calculate_cosine_sfp_rows_<APPROXIMATION_MODE>();
        ckernel::math::_incr_counters_<0x0, 0x0, ckernel::math::SFP_ROWS, 0x0>();
    }
}

// -----------------------------------------------------------------------------
// acosh(x) = log(x + sqrt(x^2 - 1)); NaN for x < 1 (propagates through sqrt).
// -----------------------------------------------------------------------------
template <bool APPROXIMATION_MODE>
inline void _calculate_acosh_sfp_rows_()
{
    TTI_SFPLOAD(p_sfpu::LREG0, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /* done */, 0 /* dest_reg */); // x

    // x^2 - 1
    TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG0, p_sfpu::LCONST_neg1, p_sfpu::LREG1, 0 /* mod1 */);
    // sqrt(x^2 - 1) -> NaN when negative
    TTI_SFPNONLINEAR(p_sfpu::LREG1, p_sfpu::LREG1, p_sfpnonlinear::SQRT_MODE);
    // x + sqrt(x^2 - 1)
    TTI_SFPADD(p_sfpu::LCONST_1, p_sfpu::LREG1, p_sfpu::LREG0, p_sfpu::LREG1, 0 /* mod1 */);

    // log(LREG1) -> LREG2; LREG3..LREG5 scratch
    _log_body_inline_(p_sfpu::LREG1, p_sfpu::LREG2, p_sfpu::LREG3, p_sfpu::LREG4, p_sfpu::LREG5);

    TTI_SFPSTORE(p_sfpu::LREG2, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /* done */, 0 /* dest_reg */);
}

template <bool APPROXIMATION_MODE>
inline void _calculate_acosh_(const int iterations)
{
#pragma GCC unroll 8
    for (int d = 0; d < iterations; d++)
    {
        _calculate_acosh_sfp_rows_<APPROXIMATION_MODE>();
        ckernel::math::_incr_counters_<0x0, 0x0, ckernel::math::SFP_ROWS, 0x0>();
    }
}

// -----------------------------------------------------------------------------
// asinh(x) = sign(x) * log(|x| + sqrt(x^2 + 1))
// -----------------------------------------------------------------------------
template <bool APPROXIMATION_MODE>
inline void _calculate_asinh_sfp_rows_()
{
    TTI_SFPLOAD(p_sfpu::LREG0, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /* done */, 0 /* dest_reg */); // x

    TTI_SFPABS(p_sfpu::LREG0, p_sfpu::LREG1, sfpi::SFPABS_MOD1_FLOAT); // |x|
    // x^2 + 1
    TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG0, p_sfpu::LCONST_1, p_sfpu::LREG2, 0 /* mod1 */);
    // sqrt(x^2 + 1)
    TTI_SFPNONLINEAR(p_sfpu::LREG2, p_sfpu::LREG2, p_sfpnonlinear::SQRT_MODE);
    // |x| + sqrt(x^2 + 1)
    TTI_SFPADD(p_sfpu::LCONST_1, p_sfpu::LREG1, p_sfpu::LREG2, p_sfpu::LREG2, 0 /* mod1 */);

    // log(...) -> LREG3; LREG4..LREG6 scratch (LREG7 reserved)
    _log_body_inline_(p_sfpu::LREG2, p_sfpu::LREG3, p_sfpu::LREG4, p_sfpu::LREG5, p_sfpu::LREG6);

    // Sign restoration: negate result when x < 0
    TTI_SFPSETCC(0 /* imm12 */, p_sfpu::LREG0, sfpi::SFPSETCC_MOD1_LREG_LT0);
    TTI_SFPMOV(p_sfpu::LREG3, p_sfpu::LREG3, 1 /* mod1: flip sign */);
    TTI_SFPENCC(0 /* imm12 */, 0 /* mod1 */);

    TTI_SFPSTORE(p_sfpu::LREG3, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /* done */, 0 /* dest_reg */);
}

template <bool APPROXIMATION_MODE>
inline void _calculate_asinh_(const int iterations)
{
#pragma GCC unroll 8
    for (int d = 0; d < iterations; d++)
    {
        _calculate_asinh_sfp_rows_<APPROXIMATION_MODE>();
        ckernel::math::_incr_counters_<0x0, 0x0, ckernel::math::SFP_ROWS, 0x0>();
    }
}

// -----------------------------------------------------------------------------
// atanh(x) = 0.5 * log((1 + x) / (1 - x))
// -----------------------------------------------------------------------------
template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en>
inline void _calculate_atanh_sfp_rows_()
{
    TTI_SFPLOAD(p_sfpu::LREG0, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /* done */, 0 /* dest_reg */); // x

    // num = 1 + x
    TTI_SFPADD(p_sfpu::LCONST_1, p_sfpu::LCONST_1, p_sfpu::LREG0, p_sfpu::LREG1, 0 /* mod1 */);
    // den = 1 - x  => (-1)*x + 1
    TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LCONST_neg1, p_sfpu::LCONST_1, p_sfpu::LREG2, 0 /* mod1 */);
    // recip(den) -> LREG3 (Quasar RECIP is always approximate; fp16b round-trip branch deferred)
    TTI_SFPNONLINEAR(p_sfpu::LREG2, p_sfpu::LREG3, p_sfpnonlinear::RECIP_MODE);

    // num / den
    TTI_SFPMUL(p_sfpu::LREG1, p_sfpu::LREG3, p_sfpu::LCONST_0, p_sfpu::LREG1, 0 /* mod1 */);

    // log(num/den) -> LREG2; LREG3..LREG5 scratch
    _log_body_inline_(p_sfpu::LREG1, p_sfpu::LREG2, p_sfpu::LREG3, p_sfpu::LREG4, p_sfpu::LREG5);

    // res = 0.5 * log(...)
    TTI_SFPMULI(FP16B_HALF, p_sfpu::LREG2, 0 /* mod1 */);

    TTI_SFPSTORE(p_sfpu::LREG2, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /* done */, 0 /* dest_reg */);
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en>
inline void _calculate_atanh_(const int iterations)
{
#pragma GCC unroll 8
    for (int d = 0; d < iterations; d++)
    {
        _calculate_atanh_sfp_rows_<APPROXIMATION_MODE, is_fp32_dest_acc_en>();
        ckernel::math::_incr_counters_<0x0, 0x0, ckernel::math::SFP_ROWS, 0x0>();
    }
}

} // namespace sfpu
} // namespace ckernel
