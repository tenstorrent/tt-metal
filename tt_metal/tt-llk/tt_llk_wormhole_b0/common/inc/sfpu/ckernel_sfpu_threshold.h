// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <type_traits>

#include "ckernel_defs.h"
#include "cmath_common.h"
#include "sfpi.h"
#include "sfpu/ckernel_sfpu_converter.h"
#include "sfpu/ckernel_sfpu_row_pair.h"

namespace ckernel::sfpu
{

template <typename T>
constexpr bool is_supported_threshold_type_v = std::is_same_v<T, float> || std::is_same_v<T, std::uint32_t>;

/**
 * Programs the threshold kernel's Dest walk (see _init_sfpu_row_pair_) and resets the Dest counters.
 * @note Call before @ref _calculate_threshold_; nothing may rewrite ADDR_MOD_6 in between.
 */
inline void _init_threshold_()
{
    _init_sfpu_row_pair_();
    math::reset_counters(p_setrwc::SET_ABD_F);
}

/**
 * Replaces every Dest value <= threshold with value, ITERATIONS rows from the current Dest row.
 * @tparam APPROXIMATION_MODE: Unused; the kernel is exact in both modes.
 * @tparam ITERATIONS: Number of Dest rows to process; must be even.
 * @tparam T: float, or std::uint32_t holding fp32 bits.
 * @note Call @ref _init_threshold_ first: the body advances Dest through ADDR_MOD_6.
 */
template <bool APPROXIMATION_MODE, int ITERATIONS, typename T>
inline void _calculate_threshold_(T threshold, T value)
{
    static_assert(is_supported_threshold_type_v<T>, "Type T must be either float or uint32_t");
    static_assert(ITERATIONS % 2 == 0, "_calculate_threshold_ processes dest rows in pairs.");

    sfpi::vFloat v_threshold;
    sfpi::vFloat v_value;
    if constexpr (std::is_same_v<T, float>)
    {
        v_threshold = threshold;
        v_value     = value;
    }
    else if constexpr (std::is_same_v<T, std::uint32_t>)
    {
        v_threshold = Converter::as_float(threshold);
        v_value     = Converter::as_float(value);
    }
    // Two rows per body: the second row's store carries the Dest increment (see ckernel_sfpu_row_pair.h),
    // which drops the per-row TTINCRWC. The compare is left in the v_if on purpose: `in <= t` lowers
    // to the SFPMAD `-1 * in + t` with the constant as the first multiplicand and a sign test on the
    // result, and the sign of a NaN produced by that MAD (t = +-inf against in = +-inf, or a NaN
    // threshold against a NaN input) depends on the operand order, so hoisting the difference into
    // `t - in` (which the compiler emits as `in * -1 + t`) changes those lanes. Kept as is, the
    // per-row instruction stream is exactly the one-row body's, so the result is bit-identical.
#pragma GCC unroll 4
    for (int d = 0; d < ITERATIONS; d += 2)
    {
        sfpi::vFloat in0 = sfpi::dst_reg[0];
        sfpi::vFloat in1 = sfpi::dst_reg[1];
        v_if (in0 <= v_threshold)
        {
            sfpi::dst_reg[0] = v_value;
        }
        v_endif;
        v_if (in1 <= v_threshold)
        {
            _sfpu_row_pair_store_second_(v_value);
        }
        v_endif;
        _sfpu_row_pair_end_();
    }
}

} // namespace ckernel::sfpu
