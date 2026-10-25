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
 * @brief Constants of the ln(x) cubic used by _calculate_log_body_ / _init_log_.
 *
 * Minimax cubic for ln(x) on [1, 2), x * (x * (x * A + B) + C) + D, max |error| 4.4e-04.
 * _init_log_ holds LN2, A and B in vConstFloatPrgm0-2. C and D are bound by the caller to a
 * vFloat outside its row loop so they stay in an LREG: sfpi 7.83.0 reloads a non-16-bit-exact
 * fp32 literal with an SFPLOADI pair at every use.
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
 * @brief Constants of the ln(x) cubic used by _calculate_log_body_no_init_, which owns no
 * program constant register.
 *
 * x * (x * (x * A - B) + C) - D, rminimax over [1, 2]; B and D are subtracted so the SFPMAD
 * sequence stays fixed. LN2 is deliberately the coarse 0.692871: lgamma, digamma and POW
 * results depend on it.
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
 * The zero test is bitwise, so -0.0 is not mapped to -inf and _calculate_log_, the only
 * caller, returns a finite value for it. A caller that holds its operand in a register uses
 * @ref _calculate_log_body_on_reg_ instead, whose biased-exponent test maps +-0 and the
 * denormals to -inf.
 *
 * @param c: LogPoly::C, bound outside the caller's row loop.
 * @param d: LogPoly::D, bound outside the caller's row loop.
 * @param dst_idx: Dest tile index of the row.
 * @note Call @ref _init_log_ first.
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
 * The ln(0) = -inf fixup is applied after the scaling. Same bitwise zero test as
 * _calculate_log_body_, so -0.0 is not mapped to -inf.
 *
 * @param c: LogPoly::C, bound outside the caller's row loop.
 * @param d: LogPoly::D, bound outside the caller's row loop.
 * @param base_scale: 1/ln(base), bound outside the caller's row loop.
 * @param dst_idx: Dest tile index of the row.
 * @note Call @ref _init_log_ first.
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
 * @brief ln(in) of a value already held in a register, with no Dest load or store.
 *
 * Same arithmetic as _calculate_log_body_, for a caller that holds its operand in a register
 * and would otherwise store it to Dest, run the in-place body and reload the result (two
 * SFPSTOREs and two SFPLOADs per row).
 *
 * The -inf lane is every input whose biased exponent is 0: +-0 *and* the denormals. That is
 * what the Dest round trip it replaces produces: SFPSTORE flushes a denormal to zero (measured
 * on Blackhole, fp32 and bf16 Dest), so the in-place body only ever sees 0 there. An
 * `in == 0.0F` test would instead return a finite value for a denormal (about -88.0..-87.3
 * for a positive one: setexp gives 1.m and exexp the raw 0 - 127, since SFPEXEXP does not
 * normalize).
 *
 * @param in: Input value.
 * @param c: LogPoly::C, bound outside the caller's row loop.
 * @param d: LogPoly::D, bound outside the caller's row loop.
 * @note Call @ref _init_log_ first.
 */
sfpi_inline sfpi::vFloat _calculate_log_body_on_reg_(const sfpi::vFloat in, const sfpi::vFloat c, const sfpi::vFloat d)
{
    sfpi::vFloat result = _calculate_log_series_(in, c, d);

    v_if (sfpi::exexp(in, sfpi::ExponentMode::Biased) == 0)
    {
        result = -std::numeric_limits<float>::infinity();
    }
    v_endif;

    return result;
}

/**
 * @brief ln(base) without any program constant register.
 *
 * ln2 and D come from the caller: a loop with two spare LREGs (lgamma) keeps them resident, a
 * loop with none (digamma) passes the plain floats and they are materialised at use. A, B and
 * C are always materialised at use. The zero test is bitwise; no caller delivers -0.0.
 *
 * @tparam Ln2T: sfpi::vFloat (LREG-resident) or float (materialised at use).
 * @tparam DT: sfpi::vFloat (LREG-resident) or float (materialised at use).
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
 * C, D and the base scale are bound once here and stay in LREGs across the row loop.
 * -0.0 is not mapped to -inf (bitwise zero test).
 *
 * @tparam APPROXIMATION_MODE: Unused.
 * @tparam HAS_BASE_SCALING: Multiply ln(x) by the base scale, values = <true/false>
 * @tparam ITERATIONS: Unused; the row count is the runtime iterations argument.
 * @param iterations: Number of Dest rows to process.
 * @param log_base_scale_factor: 1/ln(base) as fp16a bits; read only when HAS_BASE_SCALING.
 * @note Call @ref _init_log_ first.
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
