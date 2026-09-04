// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"

namespace ckernel::sfpu
{

enum class IsInfNanMode
{
    IsInf,
    IsPosInf,
    IsNegInf,
    IsNan,
    IsFinite
};

template <IsInfNanMode operation, bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_sfpu_isinf_isnan_()
{
    // SFPU microcode
    for (int d = 0; d < ITERATIONS; d++)
    {
        sfpi::vFloat in  = sfpi::dst_reg[0];
        sfpi::vFloat res = 0.0f;

        if constexpr (operation == IsInfNanMode::IsInf)
        {
            v_if (sfpi::is_inf(in))
            {
                res = 1.0f;
            }
            v_endif;
        }
        else if constexpr (operation == IsInfNanMode::IsPosInf)
        {
            v_if (sfpi::is_pos(in) && sfpi::is_inf(in))
            {
                res = 1.0f;
            }
            v_endif;
        }
        else if constexpr (operation == IsInfNanMode::IsNegInf)
        {
            v_if (sfpi::is_neg(in) && sfpi::is_inf(in))
            {
                res = 1.0f;
            }
            v_endif;
        }
        else if constexpr (operation == IsInfNanMode::IsNan)
        {
            v_if (sfpi::is_nan(in))
            {
                res = 1.0f;
            }
            v_endif;
        }
        else if constexpr (operation == IsInfNanMode::IsFinite)
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

} // namespace ckernel::sfpu
