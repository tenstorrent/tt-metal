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

// Biased fp32 exponent of 1.0: setexp(x, FP32_EXP_BIAS) rescales x into [1, 2).
constexpr std::uint32_t FP32_EXP_BIAS = 127;

/**
 * @brief Constants of the ln(x) approximation used by _calculate_log_body_ / _init_log_.
 *
 * Minimax cubic for ln(x) on x in [1, 2), Horner form x * (x * (x * A + B) + C) + D,
 * max |error| over [1, 2] = 4.4e-04. _init_log_ programs LN2, A and B into vConstFloatPrgm0-2;
 * C and D do not fit the three program registers and are passed to the body by the caller.
 *
 * Bind C and D to a vFloat *outside* the row loop. sfpi 7.83.0 materialises an fp32 literal
 * that is not exact in 16 bits with an SFPLOADI pair at every use, so inside the loop it costs
 * two instructions per row, while a vFloat bound before the loop is loop-invariant and stays in
 * a free LREG. Only such literals are affected: 0.0f and 1.0f come from constant registers, and
 * a bf16/fp16-exact literal is a single SFPLOADI.
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
 * no program constant register.
 *
 * x * (x * (x * A - B) + C) - D, 3rd order polynomial determined using rminimax over [1, 2].
 * B and D are subtracted, not added, so the SFPMAD sequence stays fixed. LN2 is deliberately
 * the coarse 0.692871 (about 4e-4 below ln 2), not LogPoly::LN2: lgamma, digamma and POW
 * results are defined by it, so "correcting" it changes their output. See LogPoly for why
 * callers bind these outside their row loop.
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
 * @param in: Input value.
 * @param c: LogPoly::C, bound by the caller outside its row loop.
 * @param d: LogPoly::D, bound by the caller outside its row loop.
 * @note Call @ref _init_log_ before this function; it reads vConstFloatPrgm0-2.
 */
sfpi_inline sfpi::vFloat _calculate_log_series_(const sfpi::vFloat in, const sfpi::vFloat c, const sfpi::vFloat d)
{
    ////////////////////////////
    // "normalize to calculation range"
    ////////////////////////////
    sfpi::vFloat x = setexp(in, FP32_EXP_BIAS); // rescale in to [1, 2)

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
 * The zero test is bitwise (`== 0.0F` lowers to an SFPSETCC zero test), so -0.0 is not mapped
 * to -inf: it takes the polynomial path and comes out finite (about -92.5). Of the two callers,
 * XLOGY's `in1 < 0.0f` guard turns -0.0 into NaN before it gets here; _calculate_log_ feeds
 * raw Dest rows, so there ln(-0.0) is finite. sfpi::is_zero would fix that for one more
 * instruction on every row.
 *
 * @param c: LogPoly::C, bound by the caller outside its row loop.
 * @param d: LogPoly::D, bound by the caller outside its row loop.
 * @param dst_idx: Dest tile index of the row.
 * @note Call @ref _init_log_ before this function.
 */
sfpi_inline void _calculate_log_body_(const sfpi::vFloat c, const sfpi::vFloat d, const std::uint32_t dst_idx = 0)
{
    // size of each tile in Dest is 64/SFP_DESTREG_STRIDE = 32 rows when using sfpi to load/store
    constexpr std::uint32_t dst_tile_size_sfpi = 32;

    sfpi::vFloat in     = sfpi::dst_reg[dst_idx * dst_tile_size_sfpi];
    sfpi::vFloat result = _calculate_log_series_(in, c, d);

    ////////////////////////////
    // Base case when input is 0. ln(0) = -inf (bitwise test, -0.0 is not selected; see above)
    ////////////////////////////
    v_if (in == 0.0F)
    {
        result = -std::numeric_limits<float>::infinity();
    }
    v_endif;

    sfpi::dst_reg[dst_idx * dst_tile_size_sfpi] = result;
}

/**
 * @brief One row of log_base(x) = ln(x) * base_scale in place in Dest.
 *
 * The ln(0) = -inf fixup is applied after the scaling, so log_base(0) is -inf for every
 * base_scale. Same bitwise zero test as _calculate_log_body_: -0.0 is not mapped to -inf.
 *
 * @param c: LogPoly::C, bound by the caller outside its row loop.
 * @param d: LogPoly::D, bound by the caller outside its row loop.
 * @param base_scale: 1/ln(base), bound by the caller outside its row loop.
 * @param dst_idx: Dest tile index of the row.
 * @note Call @ref _init_log_ before this function.
 */
sfpi_inline void _calculate_log_with_base_body_(
    const sfpi::vFloat c, const sfpi::vFloat d, const sfpi::vFloat base_scale, const std::uint32_t dst_idx = 0)
{
    // size of each tile in Dest is 64/SFP_DESTREG_STRIDE = 32 rows when using sfpi to load/store
    constexpr std::uint32_t dst_tile_size_sfpi = 32;

    sfpi::vFloat in     = sfpi::dst_reg[dst_idx * dst_tile_size_sfpi];
    sfpi::vFloat result = _calculate_log_series_(in, c, d) * base_scale;

    ////////////////////////////
    // Base case when input is 0. ln(0) = -inf (bitwise test, -0.0 is not selected)
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
 * Only ln2 and D are taken from the caller: a loop that also runs sfpu_reciprocal_iter (lgamma)
 * has two LREGs to spare, not five, so A, B and C are materialised at their use. The two are
 * template-typed so that a caller with no LREG to spare at all (digamma, whose piecewise
 * rational already holds all eight) passes the plain float constants and has them materialised
 * at their use as well; a vFloat built at the call site would be live across the whole body.
 *
 * The zero test is bitwise, so -0.0 is not mapped to -inf. No caller delivers one: lgamma
 * reflects x < 0.5 to 1 - x first, and digamma only logs x > 102.
 *
 * @tparam Ln2T: sfpi::vFloat (held in an LREG) or float (materialised at use).
 * @tparam DT: sfpi::vFloat (held in an LREG) or float (materialised at use).
 * @param base: Input value.
 * @param ln2: LogPolyNoInit::LN2.
 * @param d: LogPolyNoInit::D (subtracted).
 */
template <typename Ln2T, typename DT>
sfpi_inline sfpi::vFloat _calculate_log_body_no_init_(sfpi::vFloat base, const Ln2T ln2, const DT d)
{
    // Normalize base to calculation range
    sfpi::vFloat x = setexp(base, FP32_EXP_BIAS); // rescale base to [1, 2)

    // 3rd order polynomial approx - determined using rminimax over [1,2], see LogPolyNoInit
    sfpi::vFloat series_result = x * (x * (x * LogPolyNoInit::A - LogPolyNoInit::B) + LogPolyNoInit::C) - d;

    // Convert exponent to float
    auto exp          = sfpi::convert<sfpi::vSMag>(sfpi::exexp(base));
    sfpi::vFloat expf = sfpi::convert<sfpi::vFloat>(exp, sfpi::RoundMode::Nearest);

    // De-normalize to original range
    sfpi::vFloat log_result = expf * ln2 + series_result; // exp correction: ln(1+x) + exp*ln(2)

    // Base case when input is 0. ln(0) = -inf (bitwise test, -0.0 is not selected)
    v_if (base == 0.0f)
    {
        log_result = -std::numeric_limits<float>::infinity();
    }
    v_endif;

    return log_result;
}

/**
 * @brief ln(base) with every constant materialised at its use, for a caller whose loop has no
 * LREG to spare.
 *
 * @param base: Input value.
 */
sfpi_inline sfpi::vFloat _calculate_log_body_no_init_(sfpi::vFloat base)
{
    return _calculate_log_body_no_init_(base, LogPolyNoInit::LN2, LogPolyNoInit::D);
}

/**
 * @brief ln(x), or log_base(x) when HAS_BASE_SCALING, on iterations rows of Dest in place.
 *
 * LogPoly::C and D (and the base scale) are bound once here and held in LREGs across the row
 * loop. -0.0 is not mapped to -inf: the body's zero test is bitwise, so ln(-0.0) comes out
 * finite.
 *
 * @tparam APPROXIMATION_MODE: Unused; the same cubic serves both modes.
 * @tparam HAS_BASE_SCALING: Multiply ln(x) by the base scale, values = <true/false>
 * @tparam ITERATIONS: Unused; the row count is the runtime iterations argument.
 * @param iterations: Number of Dest rows to process.
 * @param log_base_scale_factor: 1/ln(base) as fp16a bits; read only when HAS_BASE_SCALING.
 * @note Call @ref _init_log_ before this function.
 */
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

/**
 * @brief Program vConstFloatPrgm0-2 with LogPoly::LN2, A and B for _calculate_log_body_.
 *
 * @tparam APPROXIMATION_MODE: Unused; the same constants serve both modes.
 */
template <bool APPROXIMATION_MODE>
inline void _init_log_()
{
    sfpi::vConstFloatPrgm0 = LogPoly::LN2;
    sfpi::vConstFloatPrgm1 = LogPoly::A;
    sfpi::vConstFloatPrgm2 = LogPoly::B;
}

} // namespace sfpu
} // namespace ckernel
