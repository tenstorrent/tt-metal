// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "sfpi.h"

namespace ckernel
{
namespace sfpu
{

/**
 * @brief Constants of the ln(x) approximation used by _calculate_log_body_ / _init_log_.
 *
 * Minimax cubic for ln(x) on x in [1, 2), Horner form x * (x * (x * A + B) + C) + D,
 * max |error| over [1, 2] = 4.4e-04. _init_log_ programs LN2, A and B into vConstFloatPrgm0-2;
 * C and D do not fit the three program registers and are passed to the body by the caller.
 *
 * Bind them to a vFloat *outside* the row loop: sfpi materialises a float literal with an
 * SFPLOADI pair at every use, so a literal inside the loop costs two instructions per row,
 * while a vFloat bound before the loop is loop-invariant and stays in a free LREG
 * (Blackhole SFPU review, G09-P4).
 */
struct LogPoly
{
    static constexpr float LN2 = 0.69314718f;
    static constexpr float A   = 0.10968964f;
    static constexpr float B   = -0.72910421f;
    static constexpr float C   = 2.11263230f;
    static constexpr float D   = -1.49277612f;
};

/**
 * @brief Constants of the ln(x) approximation used by _calculate_log_body_no_init_, which owns
 * no program constant register: x * (x * (x * A - B) + C) - D, 3rd order polynomial determined
 * using rminimax over [1, 2]. Written with the subtractions of the original so that the emitted
 * SFPMADs are unchanged. See LogPoly for why callers bind these outside their row loop.
 */
struct LogPolyNoInit
{
    static constexpr float LN2 = 0.692871f;
    static constexpr float A   = 0x2.44734p-4f;
    static constexpr float B   = 0xd.e712ap-4f;
    static constexpr float C   = 0x2.4f5388p+0f;
    static constexpr float D   = 0x1.952992p+0f;
};

/**
 * @brief ln(in) from the cubic on the mantissa plus exponent * ln2, before the ln(0) = -inf fixup.
 *
 * Requires _init_log_ to have programmed vConstFloatPrgm0-2; @p c and @p d are LogPoly::C / D
 * bound by the caller outside its row loop.
 */
sfpi_inline sfpi::vFloat _calculate_log_series_(const sfpi::vFloat in, const sfpi::vFloat c, const sfpi::vFloat d)
{
    ////////////////////////////
    // "normalize to calculation range"
    ////////////////////////////
    sfpi::vFloat x = setexp(in, 127); // set exp to exp bias (put in range of 1-2)

    ////////////////////////////
    // Minimax cubic approximation of ln(x) on x in [1, 2), Horner form:
    // x * (x * (x * A + B) + C) + D, see LogPoly
    ////////////////////////////
    sfpi::vFloat a             = sfpi::vConstFloatPrgm1;
    sfpi::vFloat b             = sfpi::vConstFloatPrgm2;
    sfpi::vFloat series_result = x * (x * (x * a + b) + c) + d;

    ////////////////////////////
    // Convert exponent to float
    ////////////////////////////
    auto exp = sfpi::convert<sfpi::vSMag>(sfpi::exexp(in));

    sfpi::vFloat expf      = sfpi::convert<sfpi::vFloat>(exp, sfpi::RoundMode::Nearest);
    sfpi::vFloat vConstLn2 = sfpi::vConstFloatPrgm0;
    return expf * vConstLn2 + series_result; // exp correction: ln(1+x) + exp*ln(2)
}

/**
 * @brief One row of ln(x) in place in Dest.
 *
 * @param c       LogPoly::C, bound by the caller outside its row loop.
 * @param d       LogPoly::D, bound by the caller outside its row loop.
 * @param dst_idx Dest tile index of the row.
 */
sfpi_inline void _calculate_log_body_(const sfpi::vFloat c, const sfpi::vFloat d, const std::uint32_t dst_idx = 0)
{
    // size of each tile in Dest is 64/SFP_DESTREG_STRIDE = 32 rows when using sfpi to load/store
    constexpr std::uint32_t dst_tile_size_sfpi = 32;

    sfpi::vFloat in     = sfpi::dst_reg[dst_idx * dst_tile_size_sfpi];
    sfpi::vFloat result = _calculate_log_series_(in, c, d);

    ////////////////////////////
    // Base case when input is 0. ln(0) = -inf
    // `== 0.0F` lowers to a bitwise SFPSETCC zero test, so -0.0 is not selected here (G09-C1).
    // Measured on Blackhole: no caller can deliver one (XLOGY's `in1 < 0.0f` guard returns NaN
    // for -0.0 first, lgamma reflects x < 0.5, digamma only logs x > 102), and sfpi::is_zero
    // would cost one more instruction on every row for nothing observable.
    ////////////////////////////
    v_if (in == 0.0F)
    { // Reload for register pressure
        result = -std::numeric_limits<float>::infinity();
    }
    v_endif;

    sfpi::dst_reg[dst_idx * dst_tile_size_sfpi] = result;
}

/**
 * @brief One row of log_base(x) = ln(x) * base_scale in place in Dest.
 *
 * @param base_scale 1/ln(base), bound by the caller outside its row loop.
 * The ln(0) = -inf fixup is applied after the scaling.
 */
sfpi_inline void _calculate_log_with_base_body_(
    const sfpi::vFloat c, const sfpi::vFloat d, const sfpi::vFloat base_scale, const std::uint32_t dst_idx = 0)
{
    // size of each tile in Dest is 64/SFP_DESTREG_STRIDE = 32 rows when using sfpi to load/store
    constexpr std::uint32_t dst_tile_size_sfpi = 32;

    sfpi::vFloat in     = sfpi::dst_reg[dst_idx * dst_tile_size_sfpi];
    sfpi::vFloat result = _calculate_log_series_(in, c, d) * base_scale;

    ////////////////////////////
    // Base case when input is 0. ln(0) = -inf
    ////////////////////////////
    v_if (in == 0.0F)
    {
        result = -std::numeric_limits<float>::infinity();
    }
    v_endif;

    sfpi::dst_reg[dst_idx * dst_tile_size_sfpi] = result;
}

/**
 * @brief ln(base) without any program constant register.
 *
 * @param ln2 LogPolyNoInit::LN2, bound by the caller outside its row loop.
 * @param d   LogPolyNoInit::D (subtracted), bound by the caller outside its row loop.
 *
 * Only these two of the five constants are taken from the caller: a loop that also runs
 * sfpu_reciprocal_iter (lgamma) has two LREGs to spare, not five. The other three are
 * materialised at their use.
 */
sfpi_inline sfpi::vFloat _calculate_log_body_no_init_(sfpi::vFloat base, const sfpi::vFloat ln2, const sfpi::vFloat d)
{
    // Normalize base to calculation range
    sfpi::vFloat x = setexp(base, 127); // set exp to exp bias (put base in range of 1-2)

    // 3rd order polynomial approx - determined using rminimax over [1,2], see LogPolyNoInit
    sfpi::vFloat series_result = x * (x * (x * LogPolyNoInit::A - LogPolyNoInit::B) + LogPolyNoInit::C) - d;

    // Convert exponent to float
    auto exp          = sfpi::convert<sfpi::vSMag>(sfpi::exexp(base));
    sfpi::vFloat expf = sfpi::convert<sfpi::vFloat>(exp, sfpi::RoundMode::Nearest);

    // De-normalize to original range
    sfpi::vFloat log_result = expf * ln2 + series_result; // exp correction: ln(1+x) + exp*ln(2)

    // Base case when input is 0. ln(0) = -inf
    v_if (base == 0.0f)
    {
        log_result = -std::numeric_limits<float>::infinity();
    }
    v_endif;

    return log_result;
}

/**
 * @brief ln(base) with every constant materialised at its use, for a caller whose loop has no
 * LREG to spare (digamma's piecewise rational already holds all eight).
 *
 * A twin of the form above rather than a wrapper around it: a vFloat built at the call site is
 * live across the whole body and does not fit, a literal at its use costs registers only there.
 */
sfpi_inline sfpi::vFloat _calculate_log_body_no_init_(sfpi::vFloat base)
{
    // Normalize base to calculation range
    sfpi::vFloat x = setexp(base, 127); // set exp to exp bias (put base in range of 1-2)

    // 3rd order polynomial approx - determined using rminimax over [1,2], see LogPolyNoInit
    sfpi::vFloat series_result = x * (x * (x * LogPolyNoInit::A - LogPolyNoInit::B) + LogPolyNoInit::C) - LogPolyNoInit::D;

    // Convert exponent to float
    auto exp          = sfpi::convert<sfpi::vSMag>(sfpi::exexp(base));
    sfpi::vFloat expf = sfpi::convert<sfpi::vFloat>(exp, sfpi::RoundMode::Nearest);

    // De-normalize to original range
    sfpi::vFloat vConstLn2  = LogPolyNoInit::LN2;
    sfpi::vFloat log_result = expf * vConstLn2 + series_result; // exp correction: ln(1+x) + exp*ln(2)

    // Base case when input is 0. ln(0) = -inf
    v_if (base == 0.0f)
    {
        log_result = -std::numeric_limits<float>::infinity();
    }
    v_endif;

    return log_result;
}

template <bool APPROXIMATION_MODE, bool HAS_BASE_SCALING, int ITERATIONS>
inline void _calculate_log_(const int iterations, std::uint32_t log_base_scale_factor)
{
    const sfpi::vFloat c = LogPoly::C;
    const sfpi::vFloat d = LogPoly::D;
    if constexpr (HAS_BASE_SCALING)
    {
        const sfpi::vFloat base_scale = sfpi::sFloat16a(log_base_scale_factor);
#pragma GCC unroll 8
        for (int i = 0; i < iterations; i++)
        {
            _calculate_log_with_base_body_(c, d, base_scale);
            sfpi::dst_reg++;
        }
    }
    else
    {
#pragma GCC unroll 8
        for (int i = 0; i < iterations; i++)
        {
            _calculate_log_body_(c, d);
            sfpi::dst_reg++;
        }
    }
}

template <bool APPROXIMATION_MODE>
inline void _init_log_()
{
    sfpi::vConstFloatPrgm0 = LogPoly::LN2;
    sfpi::vConstFloatPrgm1 = LogPoly::A;
    sfpi::vConstFloatPrgm2 = LogPoly::B;
}

} // namespace sfpu
} // namespace ckernel
