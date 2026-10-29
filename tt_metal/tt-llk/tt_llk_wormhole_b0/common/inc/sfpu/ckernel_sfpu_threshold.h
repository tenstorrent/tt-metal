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
 * @brief Sets up the threshold kernel's Dest walk and resets the Dest counters.
 *
 * Programs ADDR_MOD_6 through @ref _init_sfpu_row_pair_: the two-row body advances Dest from its second
 * store instead of a separate increment, so a missing init misaddresses Dest without an error.
 *
 * @note Call before @ref _calculate_threshold_; nothing may rewrite ADDR_MOD_6 in between.
 */
inline void _init_threshold_()
{
    _init_sfpu_row_pair_();
    math::reset_counters(p_setrwc::SET_ABD_F);
}

/**
 * @brief Replaces every Dest value <= threshold with value, over ITERATIONS rows from the current Dest row.
 *
 * @tparam APPROXIMATION_MODE: Unused; the kernel is exact in both modes.
 * @tparam ITERATIONS: Number of Dest rows to process; must be even.
 * @tparam T: float, or std::uint32_t holding fp32 bits.
 * @param threshold: Values at or below it are replaced.
 * @param value: The replacement.
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
    // which drops the per-row TTINCRWC. `in <= t` is tested as d = -1 * in + t >= 0: that is the SFPMAD
    // the compiler emits for the compare, with the constant -1 as the first multiplicand. Forming both
    // rows' d before either SFPSETCC lets row 1's MAD fill row 0's result shadow instead of an SFPNOP.
    // The MAD is spelled out with the builtin because the operand order matters: the sign of a NaN it
    // produces (t = +-inf against in = +-inf, or a NaN threshold against a NaN input) follows the
    // operand order, and `t - in` would be emitted as `in * -1 + t`.
    const sfpi::vFloat neg_one = -1.0f;
#pragma GCC unroll 4
    for (int d = 0; d < ITERATIONS; d += 2)
    {
        sfpi::vFloat in0 = sfpi::dst_reg[0];
        sfpi::vFloat in1 = sfpi::dst_reg[1];
        sfpi::vFloat d0  = __builtin_rvtt_sfpmad(neg_one.get(), in0.get(), v_threshold.get(), sfpi::SFPMAD_MOD1_OFFSET_NONE);
        sfpi::vFloat d1  = __builtin_rvtt_sfpmad(neg_one.get(), in1.get(), v_threshold.get(), sfpi::SFPMAD_MOD1_OFFSET_NONE);
        v_if (d0 >= 0.0f)
        {
            sfpi::dst_reg[0] = v_value;
        }
        v_endif;
        v_if (d1 >= 0.0f)
        {
            _sfpu_row_pair_store_second_(v_value);
        }
        v_endif;
        _sfpu_row_pair_end_();
    }
}

} // namespace ckernel::sfpu
