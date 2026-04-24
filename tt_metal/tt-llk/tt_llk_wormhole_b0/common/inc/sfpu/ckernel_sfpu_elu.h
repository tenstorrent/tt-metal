// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel_sfpu_converter.h"
#include "ckernel_sfpu_exp.h"
#include "sfpi.h"

namespace ckernel::sfpu
{

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en = false, int ITERATIONS = 8>
inline void _calculate_elu_(std::uint32_t dst_index_in, std::uint32_t dst_index_out, std::uint32_t slope)
{
    constexpr std::uint32_t SFP_DST_TILE_ROWS = 32;
    sfpi::vFloat s = Converter::as_float(slope);
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)
    {
        sfpi::vFloat v = sfpi::dst_reg[dst_index_in * SFP_DST_TILE_ROWS];
        v_if (v < 0.0f)
        {
            sfpi::vFloat v_exp = _sfpu_exp_21f_bf16_<true>(v) - sfpi::vConst1; // is_fp32_dest_acc_en set to true to avoid rounding as
                                                                               // it has to be done at the end of operation
            sfpi::vFloat result = s * v_exp;
            if constexpr (!is_fp32_dest_acc_en)
            {
                result = sfpi::reinterpret<sfpi::vFloat>(sfpi::float_to_fp16b(result, sfpi::RoundMode::NearestEven));
            }
            sfpi::dst_reg[dst_index_out * SFP_DST_TILE_ROWS] = result;
        }
        v_endif;
        sfpi::dst_reg++;
    }
}

} // namespace ckernel::sfpu
