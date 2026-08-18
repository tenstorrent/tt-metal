// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_sfpu_compat.h"
#include "cmath_common.h"
#include "sfpi.h"

namespace ckernel::sfpu
{

sfpi_inline void load_value_param_float(std::uint32_t value)
{
    sfpi::vConstIntPrgm0 = value;
}

template <bool IS_MAX_OP = true, bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_unary_max_min(std::uint32_t value)
{
    const sfpi::vFloat bound = sfpi::as<sfpi::vFloat>(sfpi::vUInt(value));
    for (int d = 0; d < ITERATIONS; d++)
    {
        const sfpi::vFloat input = sfpi::dst_reg[0];
        if constexpr (IS_MAX_OP)
        {
            sfpi::dst_reg[0] = compat::fp_max(input, bound);
        }
        else
        {
            sfpi::dst_reg[0] = compat::fp_min(input, bound);
        }
        sfpi::dst_reg++;
    }
}

template <bool IS_UNSIGNED = false>
sfpi_inline void load_value_param_int(std::uint32_t value)
{
    sfpi::vConstIntPrgm0 = value;
}

template <bool IS_MAX_OP = true, bool IS_UNSIGNED = false, bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_unary_max_min_int32(std::uint32_t value)
{
    for (int d = 0; d < ITERATIONS; d++)
    {
        if constexpr (IS_UNSIGNED)
        {
            const sfpi::vUInt input = sfpi::dst_reg[0].mode<sfpi::DataLayout::I32>();
            const sfpi::vUInt bound = value;
            sfpi::vUInt result      = bound;
            if constexpr (IS_MAX_OP)
            {
                v_if (input > bound)
                {
                    result = input;
                }
                v_endif;
            }
            else
            {
                v_if (input <= bound)
                {
                    result = input;
                }
                v_endif;
            }
            sfpi::dst_reg[0].mode<sfpi::DataLayout::I32>() = result;
        }
        else
        {
            const sfpi::vInt input = sfpi::dst_reg[0].mode<sfpi::DataLayout::I32>();
            const sfpi::vInt bound = static_cast<std::int32_t>(value);
            sfpi::vInt result      = bound;
            if constexpr (IS_MAX_OP)
            {
                v_if (input > bound)
                {
                    result = input;
                }
                v_endif;
            }
            else
            {
                v_if (input <= bound)
                {
                    result = input;
                }
                v_endif;
            }
            sfpi::dst_reg[0].mode<sfpi::DataLayout::I32>() = result;
        }
        sfpi::dst_reg++;
    }
}

template <bool IS_MAX_OP = true>
inline void unary_max_min_init()
{
    math::_reset_counters_<p_setrwc::SET_ABD_F>();
}

template <bool IS_MAX_OP = true, bool IS_UNSIGNED = false>
inline void unary_max_min_int32_init()
{
    math::_reset_counters_<p_setrwc::SET_ABD_F>();
}

} // namespace ckernel::sfpu
