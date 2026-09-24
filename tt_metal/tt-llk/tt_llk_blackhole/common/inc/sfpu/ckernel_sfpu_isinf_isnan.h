// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"
#include "sfpu_compare_types.h"

namespace ckernel::sfpu
{

template <FiniteCheck CHECK, bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_sfpu_isinf_isnan_()
{
    // SFPU microcode
    for (int d = 0; d < ITERATIONS; d++)
    {
        sfpi::vFloat in  = sfpi::dst_reg[0];
        sfpi::vFloat res = 0.0f;

        if constexpr (CHECK == FiniteCheck::isinf)
        {
            v_if (sfpi::is_inf(in))
            {
                res = 1.0f;
            }
            v_endif;
        }
        else if constexpr (CHECK == FiniteCheck::isposinf)
        {
            v_if (sfpi::is_pos(in) && sfpi::is_inf(in))
            {
                res = 1.0f;
            }
            v_endif;
        }
        else if constexpr (CHECK == FiniteCheck::isneginf)
        {
            v_if (sfpi::is_neg(in) && sfpi::is_inf(in))
            {
                res = 1.0f;
            }
            v_endif;
        }
        else if constexpr (CHECK == FiniteCheck::isnan)
        {
            v_if (sfpi::is_nan(in))
            {
                res = 1.0f;
            }
            v_endif;
        }
        else if constexpr (CHECK == FiniteCheck::isfinite)
        {
            v_if (sfpi::is_finite(in))
            {
                res = 1.0f;
            }
            v_endif;
        }

        sfpi::dst_reg[0] = res;
        sfpi::dst_reg++;
    }
}

/**
 * @brief Map a legacy SfpuType isinf/isnan selector to its FiniteCheck.
 *
 * @tparam operation: Legacy selector, values = <isinf/isposinf/isneginf/isnan/isfinite>
 */
template <SfpuType operation>
constexpr FiniteCheck _sfpu_type_to_finite_check_()
{
    if constexpr (operation == SfpuType::isinf)
    {
        return FiniteCheck::isinf;
    }
    else if constexpr (operation == SfpuType::isposinf)
    {
        return FiniteCheck::isposinf;
    }
    else if constexpr (operation == SfpuType::isneginf)
    {
        return FiniteCheck::isneginf;
    }
    else if constexpr (operation == SfpuType::isnan)
    {
        return FiniteCheck::isnan;
    }
    else
    {
        static_assert(operation == SfpuType::isfinite, "SfpuType is not an isinf/isnan operation");
        return FiniteCheck::isfinite;
    }
}

// SfpuType-selected entry point, kept for existing callers. It forwards to the FiniteCheck version above.
template <SfpuType operation, bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_sfpu_isinf_isnan_()
{
    _calculate_sfpu_isinf_isnan_<_sfpu_type_to_finite_check_<operation>(), APPROXIMATION_MODE, ITERATIONS>();
}

} // namespace ckernel::sfpu
