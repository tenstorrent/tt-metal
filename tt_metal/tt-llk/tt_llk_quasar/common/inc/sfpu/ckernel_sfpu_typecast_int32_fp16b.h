// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel_trisc_common.h"
#include "cmath_common.h"

namespace ckernel
{
namespace sfpu
{
// Calculates Typecast for number of rows of output SFPU ops (Quasar = 2 rows)
//
// Unpack-to-Dest copies Int32 L1 bits as two's-complement (see _mul_int32_). Cast mode 0
// (int32 → fp32 RNE) reads sign-magnitude, so convert 2SC → SM before the cast. An Int32
// source also forces 32-bit Dest; name INT32 / FP16B explicitly instead of sfpmem::DEFAULT.
inline void _calculate_typecast_int32_to_fp16b_rows()
{
    TTI_SFPLOAD(p_sfpu::LREG0, p_sfpu::sfpmem::INT32, ADDR_MOD_7, 0, 0);              // load from dest into lreg[0], uses ADDR_MOD_7 (set to all zeroes)
    TTI_SFPCAST(p_sfpu::LREG0, p_sfpu::LREG0, p_sfpu::sfp_sfpcast_mod::TWO_SC_TO_SM); // 2's complement → sign-magnitude

    TTI_SFPCAST(p_sfpu::LREG0, p_sfpu::LREG1, 0); // convert from int32 sign+mag to fp32 using rnd nearest even
    TTI_SFP_STOCH_RND(
        p_sfpu::sfp_stochrnd_rnd_mod::NearEven,
        0,
        0,
        p_sfpu::LREG1,
        p_sfpu::LREG1,
        p_sfpu::sfp_stochrnd_mod::FP32_TO_FP16B); // convert from fp32 to fp16b using rnd nearest even

    TTI_SFPSTORE(p_sfpu::LREG1, p_sfpu::sfpmem::FP16B, ADDR_MOD_7, 0, 0); // Store from lreg[1] into dest register
}

template <int ITERATIONS = SFPU_ITERATIONS>
inline void _calculate_typecast_int32_to_fp16b_()
{
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)
    {
        _calculate_typecast_int32_to_fp16b_rows();
        ckernel::math::_incr_counters_<0x0, 0x0, ckernel::math::SFP_ROWS, 0x0>(); // does the dest_reg++ (increments by 2 rows)
    }
}

} // namespace sfpu
} // namespace ckernel
